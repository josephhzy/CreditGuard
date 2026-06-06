"""Calibration drift tracking for credit models.

Monitors whether the model's predicted probabilities remain well-calibrated
over time. A calibrated model means that when it predicts 10% default
probability, roughly 10% of those applicants actually default.

Calibration drift can occur even without feature drift -- e.g., the
economic environment changes, causing the same risk profiles to default
at different rates than the model expects.

Typical usage::

    from monitoring.calibration_drift import calibration_over_time, recalibration_needed

    cal_df = calibration_over_time(y_trues, y_scores, dates)
    needs_recal, note = recalibration_needed(y_true, y_score)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import structlog
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Calibration metrics for a single period
# ---------------------------------------------------------------------------


@dataclass
class CalibrationMetrics:
    """Calibration metrics for a single evaluation period."""

    period: str
    brier_score: float
    log_loss_value: float
    mean_predicted: float
    mean_observed: float
    calibration_gap: float  # |mean_predicted - mean_observed|
    n_samples: int
    max_bin_gap: float  # worst calibration bin deviation


def _compute_calibration_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    period: str,
    *,
    n_bins: int = 10,
) -> CalibrationMetrics:
    """Compute calibration metrics for a single period."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    # Clip scores to valid range
    y_score = np.clip(y_score, 1e-10, 1 - 1e-10)

    brier = float(brier_score_loss(y_true, y_score))
    ll = float(log_loss(y_true, y_score))

    mean_pred = float(np.mean(y_score))
    mean_obs = float(np.mean(y_true))
    cal_gap = abs(mean_pred - mean_obs)

    # Bin-level calibration via sklearn
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="uniform")
        max_bin_gap = float(np.max(np.abs(prob_true - prob_pred))) if len(prob_true) > 0 else 0.0
    except ValueError:
        max_bin_gap = 0.0

    return CalibrationMetrics(
        period=period,
        brier_score=round(brier, 6),
        log_loss_value=round(ll, 6),
        mean_predicted=round(mean_pred, 6),
        mean_observed=round(mean_obs, 6),
        calibration_gap=round(cal_gap, 6),
        n_samples=len(y_true),
        max_bin_gap=round(max_bin_gap, 6),
    )


# ---------------------------------------------------------------------------
# Calibration over time
# ---------------------------------------------------------------------------


def calibration_over_time(
    y_trues: Sequence[np.ndarray],
    y_scores: Sequence[np.ndarray],
    dates: Sequence[str],
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Track calibration metrics across multiple time periods.

    Parameters
    ----------
    y_trues:
        List of true label arrays, one per period.
    y_scores:
        List of predicted probability arrays, one per period.
    dates:
        Period labels (e.g., "2024-Q1", "2024-Q2").
    n_bins:
        Number of calibration bins.

    Returns
    -------
    pd.DataFrame
        One row per period with calibration metrics.
    """
    if not (len(y_trues) == len(y_scores) == len(dates)):
        raise ValueError("y_trues, y_scores, and dates must have the same length.")

    results: list[dict[str, Any]] = []
    for yt, ys, dt in zip(y_trues, y_scores, dates, strict=False):
        yt_arr = np.asarray(yt, dtype=int)
        ys_arr = np.asarray(ys, dtype=float)

        if yt_arr.size == 0:
            logger.warning("calibration_skip_empty_period", period=dt)
            continue

        # Need both classes present
        if len(np.unique(yt_arr)) < 2:
            logger.warning("calibration_skip_single_class", period=dt)
            continue

        metrics = _compute_calibration_metrics(yt_arr, ys_arr, dt, n_bins=n_bins)
        results.append(
            {
                "period": metrics.period,
                "brier_score": metrics.brier_score,
                "log_loss": metrics.log_loss_value,
                "mean_predicted": metrics.mean_predicted,
                "mean_observed": metrics.mean_observed,
                "calibration_gap": metrics.calibration_gap,
                "max_bin_gap": metrics.max_bin_gap,
                "n_samples": metrics.n_samples,
            }
        )

    df = pd.DataFrame(results)
    logger.info("calibration_over_time_complete", n_periods=len(df))
    return df


# ---------------------------------------------------------------------------
# Calibration drift detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationDriftResult:
    """Result of a calibration drift check."""

    reference_brier: float
    current_brier: float
    brier_delta: float
    has_drifted: bool
    severity: str  # "green", "amber", "red"
    message: str


def detect_calibration_drift(
    reference_brier: float,
    current_brier: float,
    *,
    amber_threshold: float = 0.02,
    red_threshold: float = 0.05,
) -> CalibrationDriftResult:
    """Detect whether calibration has degraded beyond a threshold.

    Parameters
    ----------
    reference_brier:
        Brier score from the reference/validation period.
    current_brier:
        Brier score from the current monitoring period.
    amber_threshold:
        Brier score increase that triggers an amber alert.
    red_threshold:
        Brier score increase that triggers a red alert.

    Returns
    -------
    CalibrationDriftResult
        Whether calibration has drifted and by how much.
    """
    delta = current_brier - reference_brier

    if delta >= red_threshold:
        severity = "red"
        has_drifted = True
        msg = (
            f"Calibration significantly degraded: Brier increased by {delta:.4f} "
            f"(ref={reference_brier:.4f}, current={current_brier:.4f}). "
            "Recalibration strongly recommended."
        )
    elif delta >= amber_threshold:
        severity = "amber"
        has_drifted = True
        msg = (
            f"Calibration moderately degraded: Brier increased by {delta:.4f} "
            f"(ref={reference_brier:.4f}, current={current_brier:.4f}). "
            "Monitor closely; recalibration may be needed."
        )
    else:
        severity = "green"
        has_drifted = False
        msg = (
            f"Calibration stable: Brier delta={delta:.4f} "
            f"(ref={reference_brier:.4f}, current={current_brier:.4f})."
        )

    logger.info(
        "calibration_drift_check",
        severity=severity,
        delta=round(delta, 4),
        reference=round(reference_brier, 4),
        current=round(current_brier, 4),
    )

    return CalibrationDriftResult(
        reference_brier=round(reference_brier, 6),
        current_brier=round(current_brier, 6),
        brier_delta=round(delta, 6),
        has_drifted=has_drifted,
        severity=severity,
        message=msg,
    )


# ---------------------------------------------------------------------------
# Recalibration recommendation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecalibrationRecommendation:
    """Whether recalibration is needed and why."""

    needs_recalibration: bool
    brier_score: float
    calibration_gap: float
    max_bin_gap: float
    recommendation: str


def recalibration_needed(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    brier_threshold: float = 0.10,
    gap_threshold: float = 0.03,
    n_bins: int = 10,
) -> RecalibrationRecommendation:
    """Assess whether the model needs recalibration on current data.

    Parameters
    ----------
    y_true:
        True binary labels.
    y_score:
        Model predicted probabilities.
    brier_threshold:
        Brier score above which recalibration is recommended.
    gap_threshold:
        Mean calibration gap above which recalibration is recommended.
    n_bins:
        Number of calibration bins.

    Returns
    -------
    RecalibrationRecommendation
        Whether to recalibrate and the supporting evidence.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.size == 0:
        return RecalibrationRecommendation(
            needs_recalibration=False,
            brier_score=0.0,
            calibration_gap=0.0,
            max_bin_gap=0.0,
            recommendation="Insufficient data to assess calibration.",
        )

    y_score = np.clip(y_score, 1e-10, 1 - 1e-10)
    metrics = _compute_calibration_metrics(y_true, y_score, "current", n_bins=n_bins)

    reasons: list[str] = []
    needs_recal = False

    if metrics.brier_score > brier_threshold:
        needs_recal = True
        reasons.append(
            f"Brier score ({metrics.brier_score:.4f}) exceeds threshold ({brier_threshold})"
        )

    if metrics.calibration_gap > gap_threshold:
        needs_recal = True
        reasons.append(
            f"Mean calibration gap ({metrics.calibration_gap:.4f}) exceeds threshold ({gap_threshold})"
        )

    if metrics.max_bin_gap > 0.15:
        needs_recal = True
        reasons.append(f"Max bin gap ({metrics.max_bin_gap:.4f}) exceeds 0.15")

    if needs_recal:
        recommendation = (
            "Recalibration recommended. "
            + "; ".join(reasons)
            + ". Consider Platt scaling or isotonic regression."
        )
    else:
        recommendation = (
            f"Calibration acceptable. Brier={metrics.brier_score:.4f}, "
            f"gap={metrics.calibration_gap:.4f}, max_bin_gap={metrics.max_bin_gap:.4f}."
        )

    logger.info(
        "recalibration_assessment",
        needs_recal=needs_recal,
        brier=metrics.brier_score,
        gap=metrics.calibration_gap,
    )

    return RecalibrationRecommendation(
        needs_recalibration=needs_recal,
        brier_score=metrics.brier_score,
        calibration_gap=metrics.calibration_gap,
        max_bin_gap=metrics.max_bin_gap,
        recommendation=recommendation,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Simulate calibration over time with gradual degradation
    n_periods = 6
    period_names = [f"2024-M{i + 1:02d}" for i in range(n_periods)]
    y_trues_list = []
    y_scores_list = []

    for i in range(n_periods):
        n = 2000
        y_t = rng.binomial(1, 0.08, size=n)
        # Add increasing miscalibration over time
        noise = rng.normal(0, 0.02 * (i + 1), size=n)
        y_s = np.clip(y_t * 0.6 + (1 - y_t) * 0.05 + noise, 0.001, 0.999)
        y_trues_list.append(y_t)
        y_scores_list.append(y_s)

    print("=== Calibration Over Time ===")
    cal_df = calibration_over_time(y_trues_list, y_scores_list, period_names)
    print(cal_df.to_string(index=False))

    print("\n=== Calibration Drift Detection ===")
    ref_brier = cal_df.iloc[0]["brier_score"]
    cur_brier = cal_df.iloc[-1]["brier_score"]
    drift_result = detect_calibration_drift(ref_brier, cur_brier)
    print(f"  Severity: {drift_result.severity.upper()}")
    print(f"  Message: {drift_result.message}")

    print("\n=== Recalibration Assessment ===")
    recal = recalibration_needed(y_trues_list[-1], y_scores_list[-1])
    print(f"  Needs recalibration: {recal.needs_recalibration}")
    print(f"  {recal.recommendation}")
