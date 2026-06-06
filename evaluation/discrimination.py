"""Core discrimination metrics for credit scoring models.

Computes ROC-AUC, PR-AUC, KS statistic, Gini coefficient, and
lift-at-k to assess how well the model separates defaulters from
non-defaulters.

Typical usage::

    from evaluation.discrimination import compute_discrimination_report

    report = compute_discrimination_report(y_true, y_score)
    print(report.roc_auc, report.ks_statistic)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import structlog
from sklearn.metrics import average_precision_score, roc_auc_score

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class DiscriminationReport:
    """Complete discrimination performance report."""

    roc_auc: float
    pr_auc: float
    ks_statistic: float
    gini: float
    lift_at_k: dict[str, float] = field(default_factory=dict)
    n_samples: int = 0
    positive_rate: float = 0.0


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------


def roc_auc(y_true: np.ndarray | pd.Series, y_score: np.ndarray | pd.Series) -> float:
    """Compute Area Under the ROC Curve.

    Parameters
    ----------
    y_true:
        Binary ground truth labels.
    y_score:
        Predicted probabilities or scores.

    Returns
    -------
    float
        ROC-AUC score in [0, 1].
    """
    score = roc_auc_score(y_true, y_score)
    logger.debug("roc_auc", value=round(score, 5))
    return float(score)


def pr_auc(y_true: np.ndarray | pd.Series, y_score: np.ndarray | pd.Series) -> float:
    """Compute Area Under the Precision-Recall Curve (average precision).

    Parameters
    ----------
    y_true:
        Binary ground truth labels.
    y_score:
        Predicted probabilities or scores.

    Returns
    -------
    float
        PR-AUC (average precision) score.
    """
    score = average_precision_score(y_true, y_score)
    logger.debug("pr_auc", value=round(score, 5))
    return float(score)


def ks_statistic(y_true: np.ndarray | pd.Series, y_score: np.ndarray | pd.Series) -> float:
    """Compute the Kolmogorov-Smirnov statistic.

    The KS statistic is the maximum absolute difference between the
    cumulative distribution functions of scores for the positive and
    negative classes.  Higher values indicate better separation.

    Parameters
    ----------
    y_true:
        Binary ground truth labels.
    y_score:
        Predicted probabilities or scores.

    Returns
    -------
    float
        KS statistic in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    pos_scores = np.sort(y_score[y_true == 1])
    neg_scores = np.sort(y_score[y_true == 0])

    # Build combined sorted thresholds
    all_scores = np.sort(np.concatenate([pos_scores, neg_scores]))

    pos_cdf = np.searchsorted(pos_scores, all_scores, side="right") / len(pos_scores)
    neg_cdf = np.searchsorted(neg_scores, all_scores, side="right") / len(neg_scores)

    ks = float(np.max(np.abs(pos_cdf - neg_cdf)))
    logger.debug("ks_statistic", value=round(ks, 5))
    return ks


def gini_coefficient(y_true: np.ndarray | pd.Series, y_score: np.ndarray | pd.Series) -> float:
    """Compute the Gini coefficient (2 * AUC - 1).

    Parameters
    ----------
    y_true:
        Binary ground truth labels.
    y_score:
        Predicted probabilities or scores.

    Returns
    -------
    float
        Gini coefficient in [-1, 1].
    """
    auc = roc_auc_score(y_true, y_score)
    gini = 2.0 * auc - 1.0
    logger.debug("gini", value=round(gini, 5))
    return float(gini)


def lift_at_k(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    k: float = 0.10,
) -> float:
    """Compute lift in the top k% of predicted scores.

    Lift measures how many times more positives are captured in the
    top-k scored population compared to random selection.

    Parameters
    ----------
    y_true:
        Binary ground truth labels.
    y_score:
        Predicted probabilities or scores.
    k:
        Fraction of population to examine (e.g. 0.10 for top 10%).

    Returns
    -------
    float
        Lift ratio.  A value of 3.0 means 3x more positives than
        random in the top-k.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    n = len(y_true)
    n_top = max(int(n * k), 1)

    # Sort descending by score
    order = np.argsort(y_score)[::-1]
    top_positives = y_true[order[:n_top]].sum()

    overall_rate = y_true.mean()
    if overall_rate == 0:
        return 0.0

    top_rate = top_positives / n_top
    lift = float(top_rate / overall_rate)
    logger.debug("lift_at_k", k=k, lift=round(lift, 3))
    return lift


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------


def compute_discrimination_report(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    lift_percentiles: list[float] | None = None,
) -> DiscriminationReport:
    """Compute a comprehensive discrimination metrics report.

    Parameters
    ----------
    y_true:
        Binary ground truth labels.
    y_score:
        Predicted probabilities or scores.
    lift_percentiles:
        List of k values for lift computation (default: [1%, 5%, 10%, 20%]).

    Returns
    -------
    DiscriminationReport
        Dataclass containing all discrimination metrics.
    """
    if lift_percentiles is None:
        lift_percentiles = [0.01, 0.05, 0.10, 0.20]

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    _roc = roc_auc(y_true, y_score)
    _pr = pr_auc(y_true, y_score)
    _ks = ks_statistic(y_true, y_score)
    _gini = gini_coefficient(y_true, y_score)

    lifts = {}
    for k in lift_percentiles:
        label = f"top_{int(k * 100)}pct"
        lifts[label] = lift_at_k(y_true, y_score, k=k)

    report = DiscriminationReport(
        roc_auc=_roc,
        pr_auc=_pr,
        ks_statistic=_ks,
        gini=_gini,
        lift_at_k=lifts,
        n_samples=len(y_true),
        positive_rate=float(y_true.mean()),
    )

    logger.info(
        "discrimination_report",
        roc_auc=round(_roc, 5),
        pr_auc=round(_pr, 5),
        ks=round(_ks, 5),
        gini=round(_gini, 5),
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

    parser = argparse.ArgumentParser(description="Compute discrimination metrics from predictions")
    parser.add_argument(
        "--predictions", type=Path, required=True, help="CSV with y_true and y_score columns"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    if "y_true" not in df.columns or "y_score" not in df.columns:
        raise ValueError("CSV must contain 'y_true' and 'y_score' columns")

    report = compute_discrimination_report(df["y_true"], df["y_score"])
    print(json.dumps(asdict(report), indent=2))
