"""Credit decisioning engine: maps model risk scores to business decisions.

Translates continuous raw risk scores (the class-balanced model output,
not the Platt-calibrated PD) into discrete decision bands (APPROVE /
REVIEW / DECLINE) using configurable thresholds from ``config/train.yaml``.
The configured 0.15 / 0.40 band edges live on the raw-score scale; see
``decisioning/THRESHOLD_JUSTIFICATION.md``.

Also provides SHAP-based decision explanations for regulatory
transparency and front-line underwriter context.

Typical usage::

    from decisioning.decision_engine import score_to_decision, explain_decision

    band = score_to_decision(0.12, thresholds)
    explanation = explain_decision(0.12, shap_values, feature_names)
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
import yaml

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "train.yaml"


def _load_thresholds(config_path: Path | None = None) -> dict[str, list[float]]:
    """Load decisioning band thresholds from the project config."""
    path = config_path or _CONFIG_PATH
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg["decisioning"]["bands"]


# ---------------------------------------------------------------------------
# Decision bands
# ---------------------------------------------------------------------------


class DecisionBand(str, enum.Enum):
    """Credit decision outcome."""

    APPROVE = "approve"
    REVIEW = "review"
    DECLINE = "decline"


@dataclass(frozen=True)
class Thresholds:
    """Threshold boundaries for decisioning bands.

    Parameters
    ----------
    approve_upper:
        Scores strictly below this boundary are approved.
    decline_lower:
        Scores at or above this boundary are declined.
        Scores in [approve_upper, decline_lower) go to manual review.
    """

    approve_upper: float
    decline_lower: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.approve_upper < self.decline_lower <= 1.0:
            raise ValueError(
                f"Invalid thresholds: approve_upper={self.approve_upper}, "
                f"decline_lower={self.decline_lower}. "
                "Must satisfy 0 <= approve_upper < decline_lower <= 1."
            )

    @classmethod
    def from_config(cls, config_path: Path | None = None) -> Thresholds:
        """Create thresholds from the YAML config file."""
        bands = _load_thresholds(config_path)
        return cls(
            approve_upper=bands["approve"][1],
            decline_lower=bands["decline"][0],
        )


# ---------------------------------------------------------------------------
# Core decision logic
# ---------------------------------------------------------------------------


def score_to_decision(score: float, thresholds: Thresholds) -> DecisionBand:
    """Map a single raw risk score to a decision band.

    Parameters
    ----------
    score:
        Raw class-balanced model score in [0, 1] — not the Platt-calibrated
        PD. The configured bands are tuned on the raw-score distribution.
    thresholds:
        Decision boundaries.

    Returns
    -------
    DecisionBand
        The corresponding business decision.
    """
    if not 0.0 <= score <= 1.0:
        logger.warning("score_out_of_range", score=score)
        score = float(np.clip(score, 0.0, 1.0))

    if score < thresholds.approve_upper:
        return DecisionBand.APPROVE
    elif score >= thresholds.decline_lower:
        return DecisionBand.DECLINE
    else:
        return DecisionBand.REVIEW


def batch_decide(
    scores: np.ndarray | Sequence[float],
    thresholds: Thresholds,
) -> list[DecisionBand]:
    """Map an array of raw risk scores to decision bands.

    Parameters
    ----------
    scores:
        Array-like of raw class-balanced model scores (not calibrated PDs).
    thresholds:
        Decision boundaries.

    Returns
    -------
    list[DecisionBand]
        One decision per input score.
    """
    scores_arr = np.asarray(scores, dtype=float)
    if scores_arr.ndim == 0:
        return [score_to_decision(float(scores_arr), thresholds)]

    if scores_arr.size == 0:
        logger.warning("batch_decide_empty_input")
        return []

    decisions: list[DecisionBand] = []
    for s in scores_arr.flat:
        decisions.append(score_to_decision(float(s), thresholds))

    band_counts = {band.value: 0 for band in DecisionBand}
    for d in decisions:
        band_counts[d.value] += 1
    logger.info(
        "batch_decide_complete",
        n_total=len(decisions),
        **band_counts,
    )
    return decisions


# ---------------------------------------------------------------------------
# Decision explanation (SHAP-based)
# ---------------------------------------------------------------------------

_RECOMMENDED_ACTIONS: dict[DecisionBand, str] = {
    DecisionBand.APPROVE: "Proceed to offer generation.",
    DecisionBand.REVIEW: "Route to manual underwriter for additional review.",
    DecisionBand.DECLINE: "Issue decline notice with adverse action reasons.",
}


@dataclass
class DecisionExplanation:
    """Structured explanation of a credit decision."""

    risk_score: float
    risk_band: str
    decision: str
    recommended_action: str
    top_positive_factors: list[dict[str, Any]] = field(default_factory=list)
    top_negative_factors: list[dict[str, Any]] = field(default_factory=list)
    model_version: str = "v1.0"
    confidence_note: str = ""


def explain_decision(
    score: float,
    shap_values: np.ndarray,
    feature_names: Sequence[str],
    *,
    top_k: int = 5,
    model_version: str = "v1.0",
    thresholds: Thresholds | None = None,
) -> dict[str, Any]:
    """Generate a human-readable explanation for a credit decision.

    Uses SHAP values to identify the features that contributed most
    to the risk score, enabling regulatory-compliant adverse action
    reporting and underwriter context.

    Parameters
    ----------
    score:
        Raw model risk score for this applicant — same scale as the
        decisioning bands; the calibrated PD lives on a different scale.
    shap_values:
        1-D array of SHAP values for this applicant, one per feature.
    feature_names:
        Feature names matching the SHAP values.
    top_k:
        Number of top contributing features to surface on each side.
    model_version:
        Version string to attach to the explanation.
    thresholds:
        Decision boundaries. Loaded from config if ``None``.

    Returns
    -------
    dict
        Structured explanation with risk score, band, decision,
        recommended action, and top contributing factors.
    """
    if thresholds is None:
        thresholds = Thresholds.from_config()

    shap_arr = np.asarray(shap_values, dtype=float)
    if shap_arr.shape[0] != len(feature_names):
        raise ValueError(
            f"SHAP values length ({shap_arr.shape[0]}) does not match "
            f"feature_names length ({len(feature_names)})."
        )

    decision = score_to_decision(score, thresholds)

    # Sort by absolute SHAP contribution
    sorted_indices = np.argsort(np.abs(shap_arr))[::-1]

    positive_factors: list[dict[str, Any]] = []
    negative_factors: list[dict[str, Any]] = []

    for idx in sorted_indices:
        entry = {
            "feature": feature_names[idx],
            "shap_value": round(float(shap_arr[idx]), 6),
            "direction": "increases_risk" if shap_arr[idx] > 0 else "decreases_risk",
        }
        if shap_arr[idx] > 0 and len(positive_factors) < top_k:
            positive_factors.append(entry)
        elif shap_arr[idx] <= 0 and len(negative_factors) < top_k:
            negative_factors.append(entry)

        if len(positive_factors) >= top_k and len(negative_factors) >= top_k:
            break

    # Confidence note based on how close the score is to a boundary
    dist_to_approve = abs(score - thresholds.approve_upper)
    dist_to_decline = abs(score - thresholds.decline_lower)
    min_dist = min(dist_to_approve, dist_to_decline)

    if min_dist < 0.03:
        confidence_note = (
            "Score is near a decision boundary. Small changes in inputs could change the outcome."
        )
    elif decision == DecisionBand.APPROVE and score < 0.05:
        confidence_note = "High confidence: score is well within the approval band."
    elif decision == DecisionBand.DECLINE and score > 0.7:
        confidence_note = "High confidence: score is well within the decline band."
    else:
        confidence_note = "Moderate confidence in the decision band assignment."

    explanation = DecisionExplanation(
        risk_score=round(score, 6),
        risk_band=decision.value,
        decision=decision.value.upper(),
        recommended_action=_RECOMMENDED_ACTIONS[decision],
        top_positive_factors=positive_factors,
        top_negative_factors=negative_factors,
        model_version=model_version,
        confidence_note=confidence_note,
    )

    logger.info(
        "decision_explained",
        score=round(score, 4),
        band=decision.value,
        n_positive=len(positive_factors),
        n_negative=len(negative_factors),
    )

    return {
        "risk_score": explanation.risk_score,
        "risk_band": explanation.risk_band,
        "decision": explanation.decision,
        "recommended_action": explanation.recommended_action,
        "top_factors": {
            "increases_risk": explanation.top_positive_factors,
            "decreases_risk": explanation.top_negative_factors,
        },
        "model_version": explanation.model_version,
        "confidence_note": explanation.confidence_note,
    }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CreditGuard Decision Engine demo")
    parser.add_argument(
        "--score",
        type=float,
        default=0.25,
        help="Raw risk score to classify (0..1; raw-score scale, not calibrated PD)",
    )
    parser.add_argument("--config", type=Path, default=None, help="Config YAML path")
    args = parser.parse_args()

    thresholds = Thresholds.from_config(args.config)
    band = score_to_decision(args.score, thresholds)
    print(f"Score: {args.score:.4f} -> Decision: {band.value.upper()}")

    # Demo explanation with synthetic SHAP values
    rng = np.random.default_rng(42)
    n_feats = 10
    feat_names = [f"feature_{i}" for i in range(n_feats)]
    shap_vals = rng.normal(0, 0.1, size=n_feats)

    explanation = explain_decision(args.score, shap_vals, feat_names, thresholds=thresholds)
    print(json.dumps(explanation, indent=2))
