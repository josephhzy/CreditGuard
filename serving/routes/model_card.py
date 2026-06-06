"""Model card / governance metadata endpoint.

GET /model-card -- model governance information including intended use,
                   limitations, and performance summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog
from fastapi import APIRouter

from serving.schemas import ModelCardResponse, PerformanceMetric

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["governance"])

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODEL_CARD_PATH = _PROJECT_ROOT / "governance" / "model_card.json"


def _load_model_card_file() -> dict | None:
    """Load model card JSON if it exists on disk."""
    if _MODEL_CARD_PATH.exists():
        with open(_MODEL_CARD_PATH) as f:
            data: dict = json.load(f)
        return data
    return None


def _default_model_card(config: dict, model_version: str, training_date: str) -> ModelCardResponse:
    """Build a sensible default model card from config and model metadata."""
    eval_cfg = config.get("evaluation", {})
    fairness_cfg = eval_cfg.get("fairness", {})
    disc_metrics = eval_cfg.get("discrimination", {}).get("metrics", [])

    # Try to load real metrics from evaluation artifacts
    _EVAL_PATH = _PROJECT_ROOT / "artifacts" / "evaluation_results.json"
    _real_disc: dict = {}
    if _EVAL_PATH.exists():
        with open(_EVAL_PATH) as _f:
            _eval = json.load(_f)
        _real_disc = _eval.get("champion", {}).get("discrimination", {})

    perf_metrics = [
        PerformanceMetric(
            metric=m,
            value=_real_disc.get(m, -1.0),
            dataset="validation" if _real_disc.get(m) is not None else "not_computed",
        )
        for m in disc_metrics
    ]

    return ModelCardResponse(
        model_name="CreditGuard Default Predictor",
        version=model_version,
        description=(
            "Binary classifier predicting probability of loan default within "
            "the Home Credit dataset. Outputs a calibrated risk score mapped "
            "to APPROVE / REVIEW / DECLINE decisions via configurable thresholds."
        ),
        training_date=training_date,
        intended_use=(
            "Scoring consumer loan applications for credit default risk. "
            "Intended as a decision-support tool alongside human review, "
            "not as an autonomous lending decision maker."
        ),
        non_use_policy=(
            "Must not be used as the sole basis for credit denial. "
            "Must not be applied to populations significantly different from "
            "the training distribution without revalidation. "
            "Not intended for employment, insurance, or housing decisions."
        ),
        limitations=[
            "Trained on historical Home Credit data; may not generalize to other markets.",
            "Calibration degrades if the default base rate shifts significantly.",
            "Feature drift beyond PSI > 0.20 requires retraining before production use.",
            "Does not account for macroeconomic regime changes.",
            "Fairness metrics should be monitored per regulatory requirements.",
        ],
        performance_metrics=perf_metrics,
        sensitive_features=fairness_cfg.get("sensitive_features", []),
        fairness_notes=(
            f"Monitored for demographic parity and equalized odds across "
            f"{', '.join(fairness_cfg.get('sensitive_features', []))}. "
            f"Adverse impact ratio tolerance: {fairness_cfg.get('tolerance', 0.80)}."
        ),
    )


@router.get("/model-card", response_model=ModelCardResponse)
async def model_card() -> ModelCardResponse:
    """Return model governance metadata (model card)."""
    from serving.app import get_state

    state = get_state()
    logger.info("model_card_requested")

    # Prefer a curated model card file if it exists
    card_data = _load_model_card_file()
    if card_data is not None:
        # Parse performance metrics from raw dicts
        perf = [PerformanceMetric(**m) for m in card_data.get("performance_metrics", [])]
        return ModelCardResponse(
            model_name=card_data.get("model_name", "CreditGuard"),
            version=card_data.get("version", state["model_version"]),
            description=card_data.get("description", ""),
            training_date=card_data.get("training_date", state["training_date"]),
            intended_use=card_data.get("intended_use", ""),
            non_use_policy=card_data.get("non_use_policy", ""),
            limitations=card_data.get("limitations", []),
            performance_metrics=perf,
            sensitive_features=card_data.get("sensitive_features", []),
            fairness_notes=card_data.get("fairness_notes", ""),
        )

    # Fall back to auto-generated card from config
    return _default_model_card(
        config=state["config"],
        model_version=state["model_version"],
        training_date=state["training_date"],
    )
