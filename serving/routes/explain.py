"""Detailed SHAP explanation endpoint.

POST /explain -- full feature-level SHAP explanation for one applicant
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from fastapi import APIRouter, HTTPException

from serving.routes.predict import _build_feature_vector, _jsonable
from serving.schemas import ExplainResponse, FeatureContribution, PredictRequest

if TYPE_CHECKING:
    import pandas as pd

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["explainability"])


def _compute_shap(explainer: Any, df: pd.DataFrame) -> tuple[float, list[FeatureContribution]]:
    """Compute SHAP values for a single observation.

    Takes a pre-built ``shap.TreeExplainer`` (cached in ``_state["explainer"]``
    at startup) so the per-request cost is one ``__call__`` rather than a
    fresh ``TreeExplainer(model)``.

    Returns
    -------
    base_value : float
        The expected model output (mean prediction on the background set).
    contributions : list[FeatureContribution]
        Per-feature contributions sorted by absolute SHAP value descending.
    """
    explanation = explainer(df)

    # Extract base value.
    base_raw = explanation.base_values[0]
    if isinstance(base_raw, np.ndarray):
        base_value = float(base_raw[1] if base_raw.size > 1 else base_raw.squeeze())
    else:
        base_value = float(base_raw)

    # explanation.values is typically (n_samples, n_features, n_classes) for
    # multiclass TreeExplainer, or (n_samples, n_features) for single-output.
    # After [0] sv is at most 2-D; reduce the per-class axis to per-feature
    # for binary classifiers. The previous code had an ``elif sv.ndim > 2``
    # branch that was unreachable (it would require values to be 4-D).
    sv = explanation.values[0]
    if sv.ndim == 2:
        sv = sv[:, 1] if sv.shape[1] > 1 else sv.squeeze()

    feature_names = df.columns.tolist()
    contributions: list[FeatureContribution] = []
    for i, fname in enumerate(feature_names):
        contributions.append(
            FeatureContribution(
                feature=fname,
                value=_jsonable(df.iloc[0, i]),
                shap_value=round(float(sv[i]), 6),
            )
        )

    # Sort by absolute contribution
    contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)
    return base_value, contributions


def _summarize(contributions: list[FeatureContribution], n_top: int = 5) -> str:
    """Build a human-readable explanation summary from SHAP contributions."""
    top = contributions[:n_top]
    parts: list[str] = []
    for c in top:
        direction = "increases" if c.shap_value > 0 else "decreases"
        parts.append(f"{c.feature} ({c.value}) {direction} risk by {abs(c.shap_value):.4f}")
    return "Top risk drivers: " + "; ".join(parts) + "."


@router.post("/explain", response_model=ExplainResponse)
async def explain(req: PredictRequest) -> ExplainResponse:
    """Return a detailed SHAP explanation for a single applicant."""
    from serving.app import get_state

    state = get_state()
    bundle = state.get("bundle")
    if bundle is None or bundle.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    logger.info("explain_request", sk_id_curr=req.SK_ID_CURR)

    expected_features = bundle.get("feature_names") or []
    feature_lookup = state.get("feature_lookup")
    df, _ = _build_feature_vector(req, expected_features, feature_lookup)

    explainer = state.get("explainer")
    if explainer is None:
        raise HTTPException(
            status_code=503,
            detail="SHAP explainer is not available for the loaded model.",
        )

    try:
        base_value, contributions = _compute_shap(explainer, df)
    except Exception as exc:
        logger.error("shap_computation_failed", error=str(exc))
        raise HTTPException(
            status_code=500,
            detail=f"SHAP explanation failed: {exc}",
        ) from exc

    summary = _summarize(contributions)

    return ExplainResponse(
        sk_id_curr=req.SK_ID_CURR,
        base_value=round(base_value, 6),
        feature_contributions=contributions,
        summary=summary,
    )
