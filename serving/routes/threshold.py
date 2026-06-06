"""Threshold recommendation and simulation endpoints.

GET  /threshold/recommendation -- current vs recommended thresholds
POST /threshold/simulate       -- simulate impact of proposed threshold changes
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog
from fastapi import APIRouter, HTTPException

from serving.schemas import (
    ImpactSummary,
    ThresholdBand,
    ThresholdResponse,
    ThresholdSimulateRequest,
    ThresholdSimulateResponse,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/threshold", tags=["thresholds"])

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCORE_DIST_PATH = _PROJECT_ROOT / "monitoring" / "reports" / "score_distribution.json"
_OOF_PATH = _PROJECT_ROOT / "artifacts" / "oof_predictions_scored.parquet"
_score_dist_cache: dict | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bands_from_config(config: dict) -> list[ThresholdBand]:
    """Parse decision bands from config/train.yaml."""
    bands_cfg = config.get("decisioning", {}).get(
        "bands",
        {
            "approve": [0.0, 0.15],
            "review": [0.15, 0.40],
            "decline": [0.40, 1.0],
        },
    )
    return [
        ThresholdBand(band="approve", lower=bands_cfg["approve"][0], upper=bands_cfg["approve"][1]),
        ThresholdBand(band="review", lower=bands_cfg["review"][0], upper=bands_cfg["review"][1]),
        ThresholdBand(band="decline", lower=bands_cfg["decline"][0], upper=bands_cfg["decline"][1]),
    ]


def _load_score_distribution() -> dict | None:
    """Load a score distribution (``scores`` + default ``defaults``) for impact estimation.

    Prefers a monitoring snapshot at ``monitoring/reports/score_distribution.json``.
    Falls back to the committed out-of-fold predictions
    (``artifacts/oof_predictions_scored.parquet``) so the endpoint returns genuine
    impact estimates on a fresh clone instead of failing. The result is cached for
    the process lifetime.
    """
    global _score_dist_cache
    if _score_dist_cache is not None:
        return _score_dist_cache

    if _SCORE_DIST_PATH.exists():
        with open(_SCORE_DIST_PATH) as f:
            _score_dist_cache = json.load(f)
        return _score_dist_cache

    if _OOF_PATH.exists():
        import pandas as pd

        df = pd.read_parquet(_OOF_PATH, columns=["y_score", "y_true"])
        _score_dist_cache = {
            "scores": df["y_score"].tolist(),
            "defaults": df["y_true"].astype(int).tolist(),
        }
        return _score_dist_cache

    return None


def _estimate_impact(
    approve_upper: float,
    decline_lower: float,
    score_dist: dict | None,
) -> ImpactSummary:
    """Estimate population rates and loss impact for given thresholds.

    Requires a score distribution file with aligned ``scores`` and
    ``defaults`` lists. Raises HTTP 503 if the file is absent or the
    ``defaults`` list is missing or length-mismatched.
    """
    if score_dist is not None:
        # score_dist expected keys: scores (list[float]), defaults (list[int])
        scores: list[float] = score_dist.get("scores", [])
        defaults: list[int] = score_dist.get("defaults", [])

        if not defaults or len(defaults) != len(scores):
            logger.warning(
                "score_dist_defaults_missing_or_mismatched",
                scores_len=len(scores),
                defaults_len=len(defaults),
            )
            raise HTTPException(
                status_code=503,
                detail="Score distribution file is missing defaults or has a length mismatch; impact estimation unavailable.",
            )

        n = max(len(scores), 1)

        n_approve = sum(1 for s in scores if s < approve_upper)
        n_decline = sum(1 for s in scores if s >= decline_lower)
        n_review = n - n_approve - n_decline

        approve_defaults = sum(
            1 for s, d in zip(scores, defaults) if s < approve_upper and d == 1
        )
        total_defaults = sum(defaults)

        return ImpactSummary(
            approve_rate=round(n_approve / n, 4),
            review_rate=round(n_review / n, 4),
            decline_rate=round(n_decline / n, 4),
            estimated_default_rate=round(approve_defaults / max(n_approve, 1), 4),
            estimated_loss_reduction=round(1 - approve_defaults / max(total_defaults, 1), 4),
        )

    # No score distribution available — cannot produce meaningful estimates.
    raise HTTPException(
        status_code=503,
        detail="Score distribution file not found; impact estimation unavailable.",
    )


def _recommend_thresholds(config: dict, score_dist: dict | None) -> list[ThresholdBand]:
    """Generate recommended thresholds based on cost matrix and data.

    This is a simplified recommendation engine.  In production this would
    use the evaluation module's optimal-threshold search.
    """
    eval_cfg = config.get("evaluation", {}).get("threshold", {})
    # Fallback mirrors config/train.yaml (Basel-derived book averages,
    # 10:1 ratio). See decisioning/COST_MATRIX.md for derivation.
    cost_matrix = eval_cfg.get("cost_matrix", {"fp_cost": 600, "fn_cost": 6000})

    # Cost ratio drives how aggressive the decline threshold should be.
    # (target_recall from config is not currently wired in — this is a
    # heuristic recommendation, not a recall-targeted derivation; see the
    # endpoint docstring.)
    cost_ratio = cost_matrix["fn_cost"] / max(cost_matrix["fp_cost"], 1)

    # Higher cost ratio -> lower decline threshold (more conservative)
    recommended_decline = round(max(0.25, 0.50 - 0.02 * cost_ratio), 2)
    recommended_approve = round(min(0.20, recommended_decline * 0.35), 2)

    return [
        ThresholdBand(band="approve", lower=0.0, upper=recommended_approve),
        ThresholdBand(band="review", lower=recommended_approve, upper=recommended_decline),
        ThresholdBand(band="decline", lower=recommended_decline, upper=1.0),
    ]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/recommendation", response_model=ThresholdResponse)
async def threshold_recommendation() -> ThresholdResponse:
    """Return current thresholds and an *illustrative* recommendation.

    The recommendation here is a heuristic derived from the configured
    cost matrix (see ``_recommend_thresholds``); it is not the same
    artefact as the cost-optimal threshold computed on OOF predictions
    in ``scripts/evaluate_and_compare.py`` (saved to
    ``artifacts/evaluation_results.json::champion.threshold.optimal_threshold``,
    currently 0.50). The deployed decline gate is 0.40, set deliberately
    below the cost optimum per ``decisioning/THRESHOLD_JUSTIFICATION.md``.

    Use this endpoint for what-if exploration. Do not promote its output
    to production policy without re-running the OOF analysis.
    """
    from serving.app import get_state

    state = get_state()
    config = state["config"]

    logger.info("threshold_recommendation_requested")

    current = _bands_from_config(config)
    score_dist = _load_score_distribution()
    recommended = _recommend_thresholds(config, score_dist)

    # Impact summary uses the recommended thresholds
    impact = _estimate_impact(
        approve_upper=recommended[0].upper,
        decline_lower=recommended[2].lower,
        score_dist=score_dist,
    )

    return ThresholdResponse(
        current_thresholds=current,
        recommended_thresholds=recommended,
        impact_summary=impact,
    )


@router.post("/simulate", response_model=ThresholdSimulateResponse)
async def threshold_simulate(req: ThresholdSimulateRequest) -> ThresholdSimulateResponse:
    """Simulate the impact of proposed threshold changes."""
    logger.info(
        "threshold_simulate", approve_upper=req.approve_upper, decline_lower=req.decline_lower
    )

    if req.approve_upper >= req.decline_lower:
        raise HTTPException(
            status_code=422,
            detail="approve_upper must be less than decline_lower.",
        )

    proposed = [
        ThresholdBand(band="approve", lower=0.0, upper=req.approve_upper),
        ThresholdBand(band="review", lower=req.approve_upper, upper=req.decline_lower),
        ThresholdBand(band="decline", lower=req.decline_lower, upper=1.0),
    ]

    score_dist = _load_score_distribution()
    impact = _estimate_impact(req.approve_upper, req.decline_lower, score_dist)

    warnings: list[str] = []
    if impact.estimated_default_rate > 0.10:
        warnings.append("Estimated default rate among approved applicants exceeds 10%.")
    if impact.approve_rate > 0.85:
        warnings.append("Approve rate is very high; risk exposure may be elevated.")
    if impact.decline_rate > 0.50:
        warnings.append("Decline rate exceeds 50%; consider business impact on application volume.")

    return ThresholdSimulateResponse(
        proposed_thresholds=proposed,
        impact=impact,
        warnings=warnings,
    )
