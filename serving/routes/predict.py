"""Prediction endpoints.

POST /predict       -- single-applicant prediction with explanation summary
POST /predict-batch -- batch prediction for up to 1 000 applicants
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog
from fastapi import APIRouter, HTTPException

from serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    Decision,
    PredictRequest,
    PredictResponse,
    TopFactor,
)

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["prediction"])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    """Coerce numpy scalars / NaNs into JSON-serialisable Python types.

    Pydantic v2 cannot serialise ``numpy.int64`` / ``numpy.float64`` directly,
    so feature values pulled from a DataFrame must be unwrapped before being
    placed into a response model. ``NaN`` becomes ``None`` so the API returns
    valid JSON instead of a serializer error.
    """
    if value is None:
        return None
    if isinstance(value, (int, float, str, bool)):
        return None if (isinstance(value, float) and np.isnan(value)) else value
    if hasattr(value, "item"):
        try:
            v = value.item()
        except (ValueError, TypeError):
            return str(value)
        return None if (isinstance(v, float) and np.isnan(v)) else v
    return str(value)


def _get_bundle_and_config() -> tuple[dict[str, Any], dict, str]:
    """Retrieve the loaded bundle, config, and version from app state.

    The bundle is the canonical dict shape produced by ``_wrap_as_bundle`` in
    ``serving.app`` — it always contains ``model``, ``scaler``, ``calibrator``,
    ``feature_names``, and ``version`` keys (any of which may be ``None``).
    """
    from serving.app import get_state

    state = get_state()
    bundle = state.get("bundle")
    if bundle is None or bundle.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Service is degraded.")
    return bundle, state["config"], bundle["version"]


def _build_feature_vector(
    req: PredictRequest,
    expected_features: list[str],
    feature_lookup: pd.DataFrame | None,
) -> tuple[pd.DataFrame, str]:
    """Build the engineered feature row the model will score.

    Resolution order:
      1. ``req.engineered_features`` if present — production-style call.
      2. ``feature_lookup`` keyed by ``SK_ID_CURR`` — demo path that pulls
         pre-computed features from ``data/processed/features.parquet``.
      3. Otherwise: 422. We do not silently NaN-fill the entire vector, which
         was the prior behaviour and made every request return the same score.

    Returns ``(df, source)`` where ``source`` is one of
    ``"engineered_features" | "lookup"`` for logging/observability.
    """
    if not expected_features:
        # No schema known (e.g. naked-mock model in unit tests). Fall back to
        # the raw request payload so the existing test fixtures keep working.
        data = req.model_dump(exclude={"engineered_features"})
        return pd.DataFrame([data]), "raw_passthrough"

    if req.engineered_features:
        provided = req.engineered_features
        missing = [c for c in expected_features if c not in provided]
        if missing:
            logger.warning(
                "engineered_features_missing_columns",
                missing=missing[:10],
                count=len(missing),
            )
        row = {c: provided.get(c, np.nan) for c in expected_features}
        return pd.DataFrame([row], columns=expected_features), "engineered_features"

    if feature_lookup is not None and req.SK_ID_CURR in feature_lookup.index:
        # Loc-by-label can return a Series (1 row) or DataFrame (>1 row); coerce
        # to a single-row DataFrame regardless.
        row = feature_lookup.loc[[req.SK_ID_CURR]]
        present = [c for c in expected_features if c in row.columns]
        if len(present) < len(expected_features):
            missing = [c for c in expected_features if c not in row.columns]
            for c in missing:
                row[c] = np.nan
        return row.reindex(columns=expected_features), "lookup"

    raise HTTPException(
        status_code=422,
        detail=(
            f"Cannot score SK_ID_CURR={req.SK_ID_CURR}: no engineered_features "
            "supplied and the demo feature lookup has no row for this id. "
            "Either send the engineered feature vector under "
            "'engineered_features' or use a SK_ID_CURR present in the "
            "training set."
        ),
    )


def _map_decision(score: float, config: dict) -> tuple[Decision, str]:
    """Map a risk score to a lending decision and risk band label."""
    bands = config.get("decisioning", {}).get(
        "bands",
        {
            "approve": [0.0, 0.15],
            "review": [0.15, 0.40],
            "decline": [0.40, 1.0],
        },
    )

    approve_upper = bands["approve"][1]
    decline_lower = bands["decline"][0]

    if approve_upper >= decline_lower:
        raise ValueError(
            f"Invalid decision bands: approve upper ({approve_upper}) "
            f"must be less than decline lower ({decline_lower})"
        )

    if score < approve_upper:
        return Decision.APPROVE, "low"
    elif score < decline_lower:
        return Decision.REVIEW, "medium"
    else:
        return Decision.DECLINE, "high"


def _compute_top_factors(
    model: Any,
    df: pd.DataFrame,
    explainer: Any = None,
    n_top: int = 5,
) -> list[TopFactor]:
    """Compute top contributing features using SHAP values when available,
    falling back to ``model.feature_importances_``.

    The ``explainer`` argument is the cached ``shap.TreeExplainer`` from
    application state. If it is ``None`` (no model loaded, or SHAP could
    not bind to this estimator at startup), the function falls back to
    static feature importances.
    """
    factors: list[TopFactor] = []

    if explainer is not None:
        try:
            explanation = explainer(df)
            sv = explanation.values[0]
            if sv.ndim == 2:
                sv = sv[:, 1] if sv.shape[1] > 1 else sv.squeeze()
            feature_names = df.columns.tolist()
            indices = np.argsort(np.abs(sv))[::-1][:n_top]
            for idx in indices:
                factors.append(
                    TopFactor(
                        feature=feature_names[idx],
                        value=_jsonable(df.iloc[0, idx]),
                        contribution=round(float(sv[idx]), 6),
                    )
                )
            return factors
        except (AttributeError, TypeError, ValueError) as exc:
            logger.info("shap_fallback_to_importances", reason=str(exc))

    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        feature_names = df.columns.tolist()
        indices = np.argsort(importances)[::-1][:n_top]
        for idx in indices:
            factors.append(
                TopFactor(
                    feature=feature_names[idx],
                    value=(_jsonable(df.iloc[0, idx]) if idx < len(df.columns) else None),
                    contribution=round(float(importances[idx]), 6),
                )
            )

    return factors


def _build_confidence_note(score: float, df: pd.DataFrame) -> str:
    """Generate a confidence note based on data quality signals."""
    notes: list[str] = []
    null_frac = df.isnull().sum(axis=1).iloc[0] / max(len(df.columns), 1)

    if null_frac > 0.3:
        notes.append(
            f"High missing-feature rate ({null_frac:.0%}); prediction may be less reliable."
        )
    # NOTE: `score` here is the raw class-balanced model score, not the calibrated PD
    # (risk_score). The 0.35–0.45 boundary is referenced against the raw scale, which
    # is correct — the decision bands in config are also derived on the raw scale.
    if score > 0.35 and score < 0.45:
        notes.append("Score is near the review/decline boundary; manual review recommended.")
    if not notes:
        return "Prediction is within normal operating parameters."
    return " ".join(notes)


def _score_with_bundle(bundle: dict[str, Any], df: pd.DataFrame) -> tuple[float, float]:
    """Run the trained model, returning ``(raw_score, calibrated_pd)``.

    The model produces a raw probability; if a calibrator is present in the
    bundle, the raw probability is also mapped through it to a calibrated PD.

    Returns
    -------
    raw_score : float
        The model's raw class-balanced probability. The decision thresholds in
        ``config/train.yaml`` (``decisioning.bands``) were derived on this
        scale — see ``decisioning/THRESHOLD_JUSTIFICATION.md`` — so this is the
        score that must drive ``_map_decision``.
    calibrated_pd : float
        The Platt-calibrated probability of default — a true probability,
        returned to the caller as ``risk_score``.
    """
    model = bundle["model"]
    scaler = bundle.get("scaler")
    calibrator = bundle.get("calibrator")

    if scaler is not None:
        df_scored = pd.DataFrame(
            scaler.transform(df),
            columns=df.columns,
            index=df.index,
        )
    else:
        df_scored = df

    raw = float(model.predict_proba(df_scored)[:, 1][0])

    if calibrator is not None:
        try:
            calibrated = float(calibrator.predict_proba(np.array([[raw]]))[:, 1][0])
        except (AttributeError, ValueError):
            # Some calibrators expose ``transform`` instead of predict_proba
            # (e.g. IsotonicRegression). Try that before giving up.
            try:
                calibrated = float(calibrator.transform(np.array([raw]))[0])
            except (AttributeError, TypeError, ValueError) as exc:
                # Past this point the calibrator matches neither supported
                # interface. Raise rather than silently serve the uncalibrated
                # raw score as ``risk_score`` — a quiet fallback would violate
                # the response-field contract documented on PredictResponse.
                logger.error(
                    "calibrator_unusable",
                    error=str(exc),
                    calibrator_type=type(calibrator).__name__,
                )
                raise RuntimeError(
                    f"Calibrator of type {type(calibrator).__name__} exposes "
                    "neither predict_proba nor transform; reload the bundle."
                ) from exc
    else:
        logger.warning(
            "calibrator_absent_serving_raw_score",
            detail="No calibrator in bundle; returning raw score as calibrated_pd — PD contract not met.",
        )
        calibrated = raw

    return float(np.clip(raw, 0.0, 1.0)), float(np.clip(calibrated, 0.0, 1.0))


def _predict_single(
    req: PredictRequest,
    bundle: dict[str, Any],
    config: dict,
    version: str,
    feature_lookup: pd.DataFrame | None,
    explainer: Any = None,
    config_hash: str = "",
) -> PredictResponse:
    """Run prediction for a single applicant."""
    expected_features = bundle.get("feature_names") or []

    df, source = _build_feature_vector(req, expected_features, feature_lookup)

    raw_score, calibrated_pd = _score_with_bundle(bundle, df)

    # The decision bands (config decisioning.bands, 0.15/0.40) were derived on
    # the RAW model score in decisioning/THRESHOLD_JUSTIFICATION.md, so the
    # decision and the confidence note are taken on the raw score. The
    # risk_score returned to the caller is the calibrated PD. Platt calibration
    # is monotonic, so the decision never contradicts the ordering of risk_score.
    decision, risk_band = _map_decision(raw_score, config)
    top_factors = _compute_top_factors(bundle["model"], df, explainer=explainer)
    confidence_note = _build_confidence_note(raw_score, df)

    # Audit trail: structured event capturing the inputs to a credit decision.
    # Logged at INFO so it lands in stdout alongside the request log; the
    # serving stack's structlog config can route this to a separate sink in
    # production by filtering on ``event="audit_decision"``.
    logger.info(
        "audit_decision",
        sk_id_curr=req.SK_ID_CURR,
        raw_score=round(raw_score, 6),
        calibrated_pd=round(calibrated_pd, 6),
        decision=decision.value,
        risk_band=risk_band,
        feature_source=source,
        model_version=version,
        config_hash=config_hash,
        top_reasons=[
            {"feature": f.feature, "contribution": f.contribution} for f in top_factors[:5]
        ],
    )

    return PredictResponse(
        sk_id_curr=req.SK_ID_CURR,
        risk_score=round(calibrated_pd, 6),
        risk_band=risk_band,
        decision=decision,
        top_factors=top_factors,
        model_version=version,
        confidence_note=confidence_note,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/predict", response_model=PredictResponse)
async def predict_single(req: PredictRequest) -> PredictResponse:
    """Score a single loan applicant and return risk assessment."""
    from serving.app import get_state

    bundle, config, version = _get_bundle_and_config()
    state = get_state()
    feature_lookup = state.get("feature_lookup")
    explainer = state.get("explainer")
    config_hash = state.get("config_hash", "")
    logger.info("predict_single", sk_id_curr=req.SK_ID_CURR)
    return _predict_single(req, bundle, config, version, feature_lookup, explainer, config_hash)


@router.post("/predict-batch", response_model=BatchPredictResponse)
async def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    """Score a batch of loan applicants."""
    from serving.app import get_state

    bundle, config, version = _get_bundle_and_config()
    state = get_state()
    feature_lookup = state.get("feature_lookup")
    explainer = state.get("explainer")
    config_hash = state.get("config_hash", "")
    batch_size = len(req.applicants)
    logger.info("predict_batch", count=batch_size)
    if batch_size > 500:
        logger.warning(
            "large_batch_request", count=batch_size, detail="Batch >500 may cause high latency"
        )

    predictions = [
        _predict_single(applicant, bundle, config, version, feature_lookup, explainer, config_hash)
        for applicant in req.applicants
    ]

    return BatchPredictResponse(predictions=predictions, count=len(predictions))
