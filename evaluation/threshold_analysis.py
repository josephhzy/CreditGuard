"""Business-oriented threshold analysis for credit decisioning.

Evaluates precision/recall trade-offs at various thresholds, performs
cost-matrix analysis, computes approval-vs-bad-rate curves, and
identifies optimal operating points for the credit decision.

Typical usage::

    from evaluation.threshold_analysis import compute_threshold_report

    report = compute_threshold_report(y_true, y_score, fp_cost=600, fn_cost=6000)
    print(report.optimal_threshold, report.expected_cost_at_optimal)
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
class ThresholdReport:
    """Complete threshold analysis report."""

    thresholds: list[float]
    precision_at_threshold: list[float]
    recall_at_threshold: list[float]
    fpr_at_threshold: list[float]
    approval_rates: list[float]
    bad_rates: list[float]
    expected_costs: list[float]
    optimal_threshold: float
    optimal_method: str
    expected_cost_at_optimal: float
    review_queue: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Individual analyses
# ---------------------------------------------------------------------------


def precision_recall_at_thresholds(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    thresholds: list[float] | np.ndarray | None = None,
) -> dict[str, list[float]]:
    """Compute precision and recall at specified thresholds.

    Parameters
    ----------
    y_true:
        Binary ground truth labels.
    y_score:
        Predicted default probabilities.
    thresholds:
        Decision thresholds.  Defaults to 20 evenly spaced values.

    Returns
    -------
    dict[str, list[float]]
        Keys: ``"thresholds"``, ``"precision"``, ``"recall"``, ``"fpr"``.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    thresholds = np.asarray(thresholds)

    precisions, recalls, fprs = [], [], []

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        precisions.append(float(precision))
        recalls.append(float(recall))
        fprs.append(float(fpr))

    return {
        "thresholds": thresholds.tolist(),
        "precision": precisions,
        "recall": recalls,
        "fpr": fprs,
    }


def cost_matrix_analysis(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    fp_cost: float = 600.0,
    fn_cost: float = 6000.0,
    thresholds: list[float] | np.ndarray | None = None,
) -> dict[str, list[float]]:
    """Compute expected cost per threshold under a cost matrix.

    Parameters
    ----------
    y_true:
        Binary ground truth labels (1=default).
    y_score:
        Predicted default probabilities.
    fp_cost:
        Cost of wrongly declining a good applicant.
    fn_cost:
        Cost of approving a defaulter.
    thresholds:
        Decision thresholds.

    Returns
    -------
    dict[str, list[float]]
        Keys: ``"thresholds"``, ``"expected_cost"``, ``"fp_count"``,
        ``"fn_count"``.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    thresholds = np.asarray(thresholds)

    costs, fp_counts, fn_counts = [], [], []

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        fp = float(((y_pred == 1) & (y_true == 0)).sum())
        fn = float(((y_pred == 0) & (y_true == 1)).sum())
        cost = fp * fp_cost + fn * fn_cost
        costs.append(cost)
        fp_counts.append(fp)
        fn_counts.append(fn)

    return {
        "thresholds": thresholds.tolist(),
        "expected_cost": costs,
        "fp_count": fp_counts,
        "fn_count": fn_counts,
    }


def approval_rate_vs_bad_rate(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    thresholds: list[float] | np.ndarray | None = None,
) -> dict[str, list[float]]:
    """Compute approval rate and bad rate (default rate among approved) at each threshold.

    Score >= threshold means predicted-default -> decline.
    Score < threshold means predicted-good -> approve.

    Parameters
    ----------
    y_true:
        Binary ground truth labels.
    y_score:
        Predicted default probabilities.
    thresholds:
        Decision thresholds.

    Returns
    -------
    dict[str, list[float]]
        Keys: ``"thresholds"``, ``"approval_rate"``, ``"bad_rate"``.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    thresholds = np.asarray(thresholds)

    approval_rates, bad_rates = [], []

    for t in thresholds:
        approved_mask = y_score < t  # approve those below threshold
        n_approved = approved_mask.sum()
        approval_rate = n_approved / len(y_true)
        bad_rate = y_true[approved_mask].mean() if n_approved > 0 else 0.0

        approval_rates.append(float(approval_rate))
        bad_rates.append(float(bad_rate))

    return {
        "thresholds": thresholds.tolist(),
        "approval_rate": approval_rates,
        "bad_rate": bad_rates,
    }


def optimal_threshold(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    method: str = "cost_matrix",
    fp_cost: float = 600.0,
    fn_cost: float = 6000.0,
) -> tuple[float, float]:
    """Find the optimal decision threshold.

    Parameters
    ----------
    y_true:
        Binary ground truth labels.
    y_score:
        Predicted default probabilities.
    method:
        Optimization method. ``"cost_matrix"`` minimizes expected cost;
        ``"f1"`` maximizes F1 score; ``"youden"`` maximizes Youden's J
        (sensitivity + specificity - 1).
    fp_cost:
        Cost of false positive (for cost_matrix method).
    fn_cost:
        Cost of false negative (for cost_matrix method).

    Returns
    -------
    tuple[float, float]
        (optimal_threshold, metric_value_at_optimal).
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if method == "cost_matrix":
        result = cost_matrix_analysis(y_true, y_score, fp_cost, fn_cost, thresholds)
        best_idx = int(np.argmin(result["expected_cost"]))
        best_t = result["thresholds"][best_idx]
        best_val = result["expected_cost"][best_idx]
        logger.info(
            "optimal_threshold", method=method, threshold=round(best_t, 4), cost=round(best_val, 2)
        )
        return best_t, best_val

    if method == "f1":
        f1_scores = []
        for t in thresholds:
            y_pred = (y_score >= t).astype(int)
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        best_idx = int(np.argmax(f1_scores))
        best_t = float(thresholds[best_idx])
        best_val = f1_scores[best_idx]
        logger.info(
            "optimal_threshold", method=method, threshold=round(best_t, 4), f1=round(best_val, 4)
        )
        return best_t, best_val

    if method == "youden":
        j_scores = []
        for t in thresholds:
            y_pred = (y_score >= t).astype(int)
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            j_scores.append(tpr - fpr)
        best_idx = int(np.argmax(j_scores))
        best_t = float(thresholds[best_idx])
        best_val = j_scores[best_idx]
        logger.info(
            "optimal_threshold",
            method=method,
            threshold=round(best_t, 4),
            youden_j=round(best_val, 4),
        )
        return best_t, best_val

    raise ValueError(f"Unknown method: '{method}'. Use cost_matrix/f1/youden")


def review_queue_size(
    y_score: np.ndarray | pd.Series,
    threshold_low: float = 0.15,
    threshold_high: float = 0.40,
) -> dict[str, float]:
    """Compute the size of the manual review queue.

    Applicants scoring between ``threshold_low`` and ``threshold_high``
    fall into the review band.

    Parameters
    ----------
    y_score:
        Predicted default probabilities.
    threshold_low:
        Lower bound of the review band.
    threshold_high:
        Upper bound of the review band.

    Returns
    -------
    dict[str, float]
        Queue metrics: ``n_review``, ``pct_review``, ``n_approve``,
        ``n_decline``.
    """
    y_score = np.asarray(y_score)
    n = len(y_score)

    n_approve = int((y_score < threshold_low).sum())
    n_decline = int((y_score >= threshold_high).sum())
    n_review = int(n - n_approve - n_decline)

    result = {
        "n_approve": float(n_approve),
        "n_decline": float(n_decline),
        "n_review": float(n_review),
        "pct_approve": n_approve / n if n > 0 else 0.0,
        "pct_decline": n_decline / n if n > 0 else 0.0,
        "pct_review": n_review / n if n > 0 else 0.0,
    }
    logger.info("review_queue", **{k: round(v, 4) for k, v in result.items()})
    return result


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------


def compute_threshold_report(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    fp_cost: float = 600.0,
    fn_cost: float = 6000.0,
    threshold_low: float = 0.15,
    threshold_high: float = 0.40,
    optimization_method: str = "cost_matrix",
) -> ThresholdReport:
    """Compute a comprehensive threshold analysis report.

    Parameters
    ----------
    y_true:
        Binary ground truth labels.
    y_score:
        Predicted default probabilities.
    fp_cost:
        Cost of wrongly declining a good applicant.
    fn_cost:
        Cost of approving a defaulter.
    threshold_low:
        Lower bound of review band.
    threshold_high:
        Upper bound of review band.
    optimization_method:
        Method for finding optimal threshold.

    Returns
    -------
    ThresholdReport
    """
    thresholds = np.linspace(0.05, 0.95, 19)

    pr_data = precision_recall_at_thresholds(y_true, y_score, thresholds)
    cost_data = cost_matrix_analysis(y_true, y_score, fp_cost, fn_cost, thresholds)
    ar_data = approval_rate_vs_bad_rate(y_true, y_score, thresholds)
    opt_threshold, opt_cost = optimal_threshold(
        y_true,
        y_score,
        method=optimization_method,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
    )
    queue = review_queue_size(y_score, threshold_low, threshold_high)

    report = ThresholdReport(
        thresholds=pr_data["thresholds"],
        precision_at_threshold=pr_data["precision"],
        recall_at_threshold=pr_data["recall"],
        fpr_at_threshold=pr_data["fpr"],
        approval_rates=ar_data["approval_rate"],
        bad_rates=ar_data["bad_rate"],
        expected_costs=cost_data["expected_cost"],
        optimal_threshold=opt_threshold,
        optimal_method=optimization_method,
        expected_cost_at_optimal=opt_cost,
        review_queue=queue,
    )

    logger.info(
        "threshold_report",
        optimal_threshold=round(opt_threshold, 4),
        cost_at_optimal=round(opt_cost, 2),
        pct_review=round(queue["pct_review"], 4),
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

    parser = argparse.ArgumentParser(description="Threshold analysis for credit decisioning")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--fp-cost", type=float, default=50.0)
    parser.add_argument("--fn-cost", type=float, default=500.0)
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    report = compute_threshold_report(
        df["y_true"],
        df["y_score"],
        fp_cost=args.fp_cost,
        fn_cost=args.fn_cost,
    )
    print(json.dumps(asdict(report), indent=2))
