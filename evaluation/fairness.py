"""Responsible AI fairness diagnostics for credit scoring models.

Illustrative fairness diagnostics for responsible model development.
This module is NOT a compliance tool -- it provides quantitative
signals to support human judgment during model review.

Metrics computed:
- Demographic parity (approval rate equity)
- Equalized odds (TPR/FPR equity)
- Adverse impact ratio (80% rule)
- Calibration by group
- Proxy sensitivity analysis

Typical usage::

    from evaluation.fairness import compute_fairness_report

    report = compute_fairness_report(y_true, y_pred, y_score, sensitive)
    print(report.adverse_impact_ratio)
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════
# DISCLAIMER: Illustrative fairness diagnostics for responsible model
# development. These metrics inform human review; they do not constitute
# legal compliance assessment. Always consult domain experts and legal
# counsel for actual deployment decisions.
# ═══════════════════════════════════════════════════════════════════════════
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import structlog

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class GroupMetrics:
    """Metrics for a single demographic group."""

    group: str
    n_samples: int
    positive_rate: float
    approval_rate: float
    tpr: float  # True Positive Rate (sensitivity)
    fpr: float  # False Positive Rate
    mean_score: float
    calibration_error: float  # |mean_score - positive_rate|


@dataclass
class FairnessReport:
    """Complete fairness diagnostics report.

    Illustrative fairness diagnostics for responsible model development.
    """

    sensitive_column: str
    n_groups: int
    group_metrics: list[GroupMetrics]
    demographic_parity: dict[str, float]
    equalized_odds: dict[str, dict[str, float]]
    adverse_impact_ratio: float
    calibration_by_group: dict[str, float]
    proxy_sensitivity: dict[str, float] = field(default_factory=dict)
    passes_80_pct_rule: bool = True


# ---------------------------------------------------------------------------
# Individual fairness metrics
# ---------------------------------------------------------------------------


def demographic_parity(
    y_pred: np.ndarray | pd.Series,
    sensitive: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Compute approval rate per demographic group.

    Demographic parity holds when the approval rate is equal across
    all groups.

    Parameters
    ----------
    y_pred:
        Binary predictions (1=predicted default/decline, 0=approve).
    sensitive:
        Sensitive attribute values.

    Returns
    -------
    dict[str, float]
        Approval rate per group.
    """
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)

    result: dict[str, float] = {}
    for group in sorted(set(sensitive.tolist())):
        mask = sensitive == group
        # approval = predicted non-default
        approval_rate = float((y_pred[mask] == 0).sum() / mask.sum())
        result[str(group)] = approval_rate

    logger.info("demographic_parity", rates=result)
    return result


def equalized_odds(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    sensitive: np.ndarray | pd.Series,
) -> dict[str, dict[str, float]]:
    """Compute TPR and FPR per demographic group.

    Equalized odds holds when TPR and FPR are equal across groups.

    Parameters
    ----------
    y_true:
        Binary ground truth labels.
    y_pred:
        Binary predictions.
    sensitive:
        Sensitive attribute values.

    Returns
    -------
    dict[str, dict[str, float]]
        Outer key is group; inner keys are ``"tpr"`` and ``"fpr"``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)

    result: dict[str, dict[str, float]] = {}

    for group in sorted(set(sensitive.tolist())):
        mask = sensitive == group
        yt = y_true[mask]
        yp = y_pred[mask]

        tp = ((yp == 1) & (yt == 1)).sum()
        fn = ((yp == 0) & (yt == 1)).sum()
        fp = ((yp == 1) & (yt == 0)).sum()
        tn = ((yp == 0) & (yt == 0)).sum()

        tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        result[str(group)] = {"tpr": tpr, "fpr": fpr}

    logger.info("equalized_odds", result=result)
    return result


def adverse_impact_ratio(
    y_pred: np.ndarray | pd.Series,
    sensitive: np.ndarray | pd.Series,
) -> float:
    """Compute the adverse impact ratio (four-fifths / 80% rule).

    AIR = min_group_approval_rate / max_group_approval_rate.
    Values >= 0.80 generally satisfy the 80% rule.

    Parameters
    ----------
    y_pred:
        Binary predictions (1=decline, 0=approve).
    sensitive:
        Sensitive attribute values.

    Returns
    -------
    float
        Adverse impact ratio in [0, 1].
    """
    rates = demographic_parity(y_pred, sensitive)
    if not rates:
        return 1.0

    rate_values = list(rates.values())
    max_rate = max(rate_values)
    min_rate = min(rate_values)

    air = min_rate / max_rate if max_rate > 0 else 0.0
    logger.info(
        "adverse_impact_ratio",
        air=round(air, 4),
        min_rate=round(min_rate, 4),
        max_rate=round(max_rate, 4),
    )
    return float(air)


def calibration_by_group(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    sensitive: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Compute calibration error per demographic group.

    For each group, calibration error = |mean_predicted - observed_rate|.

    Parameters
    ----------
    y_true:
        Binary ground truth labels.
    y_score:
        Predicted probabilities.
    sensitive:
        Sensitive attribute values.

    Returns
    -------
    dict[str, float]
        Calibration error per group.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    sensitive = np.asarray(sensitive)

    result: dict[str, float] = {}

    for group in sorted(set(sensitive.tolist())):
        mask = sensitive == group
        observed = float(y_true[mask].mean())
        predicted = float(y_score[mask].mean())
        result[str(group)] = abs(predicted - observed)

    logger.info("calibration_by_group", errors=result)
    return result


def proxy_sensitivity(
    X: pd.DataFrame,
    sensitive_cols: list[str],
    model: Any,
    n_perturbations: int = 100,
    random_state: int = 42,
) -> dict[str, float]:
    """Measure how much model scores change when sensitive features are perturbed.

    For each sensitive column, we randomly shuffle its values and
    measure the mean absolute change in predicted scores.  Higher
    sensitivity suggests the model may be relying on proxies.

    Parameters
    ----------
    X:
        Feature matrix.
    sensitive_cols:
        Names of sensitive columns to test.
    model:
        A fitted model with ``predict_proba``.
    n_perturbations:
        Number of random permutation trials.
    random_state:
        Random seed.

    Returns
    -------
    dict[str, float]
        Mean absolute score change per sensitive column.
    """
    rng = np.random.default_rng(random_state)
    base_scores = model.predict_proba(X)[:, 1]

    result: dict[str, float] = {}

    for col in sensitive_cols:
        if col not in X.columns:
            logger.warning("proxy_col_missing", column=col)
            continue

        total_change = 0.0
        for _ in range(n_perturbations):
            X_perm = X.copy()
            X_perm[col] = rng.permutation(X_perm[col].values)
            perm_scores = model.predict_proba(X_perm)[:, 1]
            total_change += float(np.abs(base_scores - perm_scores).mean())

        mean_change = total_change / n_perturbations
        result[col] = mean_change
        logger.info("proxy_sensitivity", column=col, mean_change=round(mean_change, 6))

    return result


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------


def compute_fairness_report(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    sensitive: pd.Series,
    tolerance: float = 0.80,
    model: Any | None = None,
    X: pd.DataFrame | None = None,
    sensitive_cols: list[str] | None = None,
) -> FairnessReport:
    """Compute a comprehensive fairness diagnostics report.

    Illustrative fairness diagnostics for responsible model development.

    Parameters
    ----------
    y_true:
        Binary ground truth labels.
    y_pred:
        Binary predictions (1=decline, 0=approve).
    y_score:
        Predicted probabilities.
    sensitive:
        Sensitive attribute series.
    tolerance:
        Adverse impact ratio threshold (default 0.80 for the 80% rule).
    model:
        Optional fitted model for proxy sensitivity analysis.
    X:
        Feature matrix (required for proxy sensitivity).
    sensitive_cols:
        Column names for proxy sensitivity (required with model).

    Returns
    -------
    FairnessReport
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_score_arr = np.asarray(y_score)
    sensitive_arr = np.asarray(sensitive)
    col_name = sensitive.name if hasattr(sensitive, "name") else "sensitive"

    # Core metrics
    dp = demographic_parity(y_pred_arr, sensitive_arr)
    eo = equalized_odds(y_true_arr, y_pred_arr, sensitive_arr)
    air = adverse_impact_ratio(y_pred_arr, sensitive_arr)
    cal_by_group = calibration_by_group(y_true_arr, y_score_arr, sensitive_arr)

    # Group-level detailed metrics
    group_metrics_list: list[GroupMetrics] = []
    for group in sorted(set(sensitive_arr.tolist())):
        mask = sensitive_arr == group
        yt = y_true_arr[mask]
        yp = y_pred_arr[mask]
        ys = y_score_arr[mask]

        n = int(mask.sum())
        pos_rate = float(yt.mean())
        approval = float((yp == 0).sum() / n) if n > 0 else 0.0
        mean_sc = float(ys.mean())
        cal_err = abs(mean_sc - pos_rate)

        tp = ((yp == 1) & (yt == 1)).sum()
        fn = ((yp == 0) & (yt == 1)).sum()
        fp = ((yp == 1) & (yt == 0)).sum()
        tn = ((yp == 0) & (yt == 0)).sum()
        tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        group_metrics_list.append(
            GroupMetrics(
                group=str(group),
                n_samples=n,
                positive_rate=pos_rate,
                approval_rate=approval,
                tpr=tpr,
                fpr=fpr,
                mean_score=mean_sc,
                calibration_error=cal_err,
            )
        )

    # Proxy sensitivity
    proxy_result: dict[str, float] = {}
    if model is not None and X is not None and sensitive_cols is not None:
        proxy_result = proxy_sensitivity(X, sensitive_cols, model)

    passes = air >= tolerance

    report = FairnessReport(
        sensitive_column=col_name,
        n_groups=len(group_metrics_list),
        group_metrics=group_metrics_list,
        demographic_parity=dp,
        equalized_odds=eo,
        adverse_impact_ratio=air,
        calibration_by_group=cal_by_group,
        proxy_sensitivity=proxy_result,
        passes_80_pct_rule=passes,
    )

    logger.info(
        "fairness_report",
        column=col_name,
        n_groups=len(group_metrics_list),
        air=round(air, 4),
        passes_80_pct=passes,
    )
    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Fairness diagnostics for credit scoring")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--sensitive-col", type=str, required=True)
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Decision threshold for y_pred"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    y_pred = (df["y_score"] >= args.threshold).astype(int)

    report = compute_fairness_report(
        df["y_true"],
        y_pred,
        df["y_score"],
        df[args.sensitive_col],
    )

    print(f"\nFairness Report: {report.sensitive_column}")
    print(f"  Adverse Impact Ratio: {report.adverse_impact_ratio:.4f}")
    print(f"  Passes 80% Rule:     {report.passes_80_pct_rule}")
    print("\n  Demographic Parity (approval rates):")
    for g, rate in report.demographic_parity.items():
        print(f"    {g:25s}  {rate:.4f}")
    print("\n  Equalized Odds:")
    for g, vals in report.equalized_odds.items():
        print(f"    {g:25s}  TPR={vals['tpr']:.4f}  FPR={vals['fpr']:.4f}")
