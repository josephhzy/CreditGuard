"""Probability calibration assessment and correction.

Evaluates whether predicted probabilities reflect true event rates
(reliability diagram), computes Brier score, and provides Platt
scaling and isotonic regression recalibration.

Typical usage::

    from evaluation.calibration import compute_calibration_report

    report = compute_calibration_report(y_true, y_score)
    print(report.brier_score, report.ece)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import structlog
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class CalibrationReport:
    """Complete calibration assessment report."""

    brier_score: float
    ece: float  # Expected Calibration Error
    reliability_curve: dict[str, list[float]]  # mean_predicted, empirical_freq
    platt_brier: float | None = None
    isotonic_brier: float | None = None
    comparison: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def brier_score(y_true: np.ndarray | pd.Series, y_score: np.ndarray | pd.Series) -> float:
    """Compute Brier score (mean squared error of probability estimates).

    Parameters
    ----------
    y_true:
        Binary ground truth labels.
    y_score:
        Predicted probabilities.

    Returns
    -------
    float
        Brier score (lower is better).
    """
    score = brier_score_loss(y_true, y_score)
    logger.debug("brier_score", value=round(score, 6))
    return float(score)


def expected_calibration_error(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE is the weighted average of bin-wise |predicted_rate - true_rate|,
    weighted by the fraction of samples in each bin.

    Parameters
    ----------
    y_true:
        Binary ground truth labels.
    y_score:
        Predicted probabilities.
    n_bins:
        Number of calibration bins.

    Returns
    -------
    float
        ECE (lower is better).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:], strict=False):
        mask = (y_score >= lo) & (y_score < hi)
        if mask.sum() == 0:
            continue
        bin_true = y_true[mask].mean()
        bin_pred = y_score[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_true - bin_pred)

    logger.debug("ece", value=round(ece, 6), n_bins=n_bins)
    return float(ece)


def compute_reliability_curve(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    n_bins: int = 10,
) -> dict[str, list[float]]:
    """Compute reliability diagram data.

    Parameters
    ----------
    y_true:
        Binary ground truth labels.
    y_score:
        Predicted probabilities.
    n_bins:
        Number of calibration bins.

    Returns
    -------
    dict[str, list[float]]
        Keys: ``"empirical_freq"``, ``"mean_predicted"``.
    """
    fraction_positive, mean_predicted = calibration_curve(
        y_true,
        y_score,
        n_bins=n_bins,
        strategy="uniform",
    )
    return {
        "empirical_freq": fraction_positive.tolist(),
        "mean_predicted": mean_predicted.tolist(),
    }


# ---------------------------------------------------------------------------
# Recalibration methods
# ---------------------------------------------------------------------------


def platt_scaling(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
) -> tuple[np.ndarray, LogisticRegression]:
    """Apply Platt scaling (logistic regression on raw scores).

    Parameters
    ----------
    y_true:
        Binary ground truth (used to fit the calibrator).
    y_score:
        Raw predicted probabilities.

    Returns
    -------
    tuple[np.ndarray, LogisticRegression]
        Recalibrated probabilities and the fitted calibrator.
    """
    y_score = np.asarray(y_score).reshape(-1, 1)
    lr = LogisticRegression(C=1e9, solver="lbfgs", max_iter=200)
    lr.fit(y_score, y_true)
    calibrated = lr.predict_proba(y_score)[:, 1]
    logger.info("platt_scaling_fitted")
    return calibrated, lr


def isotonic_calibration(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
) -> tuple[np.ndarray, IsotonicRegression]:
    """Apply isotonic regression calibration.

    Parameters
    ----------
    y_true:
        Binary ground truth.
    y_score:
        Raw predicted probabilities.

    Returns
    -------
    tuple[np.ndarray, IsotonicRegression]
        Recalibrated probabilities and the fitted calibrator.
    """
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(np.asarray(y_score), np.asarray(y_true))
    calibrated = iso.predict(np.asarray(y_score))
    logger.info("isotonic_calibration_fitted")
    return calibrated, iso


def compare_calibration(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    calibrator_test_size: float = 0.3,
    random_state: int = 42,
) -> dict[str, float]:
    """Compare raw, Platt, and isotonic calibration via Brier score.

    The raw Brier score is computed on the full input.  Calibrator Brier
    scores are computed on a held-out split so that the calibrator is not
    evaluated on data it was fit on.  Pass OOF or held-out predictions as
    ``y_score``; never pass training-set scores.

    Parameters
    ----------
    y_true:
        Binary ground truth.
    y_score:
        Raw predicted probabilities (OOF or held-out).
    calibrator_test_size:
        Fraction of samples withheld for calibrator evaluation.
    random_state:
        Random seed for the train/test split.

    Returns
    -------
    dict[str, float]
        Brier scores for each method.
    """
    from sklearn.model_selection import train_test_split

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    raw_brier = brier_score(y_true, y_score)

    # Nested split so calibrators are evaluated on data they were not fit on.
    y_tr, y_te, s_tr, s_te = train_test_split(
        y_true,
        y_score,
        test_size=calibrator_test_size,
        random_state=random_state,
        stratify=y_true,
    )

    _, platt_model = platt_scaling(y_tr, s_tr)
    platt_scores_te = platt_model.predict_proba(s_te.reshape(-1, 1))[:, 1]
    platt_brier = brier_score(y_te, platt_scores_te)

    _, iso_model = isotonic_calibration(y_tr, s_tr)
    iso_scores_te = iso_model.predict(s_te)
    iso_brier = brier_score(y_te, iso_scores_te)

    comparison = {
        "raw_brier": raw_brier,
        "platt_brier": platt_brier,
        "isotonic_brier": iso_brier,
    }
    logger.info("calibration_comparison", **{k: round(v, 6) for k, v in comparison.items()})
    return comparison


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------


def compute_calibration_report(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    n_bins: int = 10,
    fit_calibrators: bool = True,
) -> CalibrationReport:
    """Compute a comprehensive calibration report.

    Parameters
    ----------
    y_true:
        Binary ground truth labels.
    y_score:
        Predicted probabilities.
    n_bins:
        Number of bins for reliability diagram and ECE.
    fit_calibrators:
        Whether to fit Platt and isotonic calibrators.

    Returns
    -------
    CalibrationReport
        Full calibration assessment.
    """
    _brier = brier_score(y_true, y_score)
    _ece = expected_calibration_error(y_true, y_score, n_bins=n_bins)
    _reliability = compute_reliability_curve(y_true, y_score, n_bins=n_bins)

    platt_b = None
    iso_b = None
    comparison: dict[str, float] = {}

    if fit_calibrators:
        comparison = compare_calibration(y_true, y_score)
        platt_b = comparison["platt_brier"]
        iso_b = comparison["isotonic_brier"]

    report = CalibrationReport(
        brier_score=_brier,
        ece=_ece,
        reliability_curve=_reliability,
        platt_brier=platt_b,
        isotonic_brier=iso_b,
        comparison=comparison,
    )

    logger.info(
        "calibration_report",
        brier=round(_brier, 6),
        ece=round(_ece, 6),
        platt_brier=round(platt_b, 6) if platt_b else None,
        isotonic_brier=round(iso_b, 6) if iso_b else None,
    )
    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import json
    from dataclasses import asdict
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Compute calibration metrics from predictions")
    parser.add_argument(
        "--predictions", type=Path, required=True, help="CSV with y_true and y_score columns"
    )
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    report = compute_calibration_report(df["y_true"], df["y_score"], n_bins=args.n_bins)
    print(json.dumps(asdict(report), indent=2))
