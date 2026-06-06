"""Subgroup performance analysis for credit scoring models.

Evaluates model discrimination and calibration across population
segments (income bands, employment status, education level, product
type) to identify subgroups where the model under- or over-performs.

Typical usage::

    from evaluation.segment_analysis import compute_segment_report

    report = compute_segment_report(y_true, y_score, segments)
    for seg in report.segments:
        print(seg.segment_value, seg.roc_auc, seg.pr_auc)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import structlog
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class SegmentMetrics:
    """Performance metrics for a single segment."""

    segment_column: str
    segment_value: str
    n_samples: int
    positive_rate: float
    roc_auc: float | None
    pr_auc: float | None
    brier_score: float
    mean_score: float
    approval_rate: float  # approval rate at the configured approval_threshold


@dataclass
class SegmentReport:
    """Complete segment analysis report."""

    segment_column: str
    n_segments: int
    segments: list[SegmentMetrics]
    overall_roc_auc: float
    overall_pr_auc: float
    worst_segment: str
    best_segment: str
    performance_gap: float  # max segment AUC - min segment AUC


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def segment_metrics(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    segment_column: pd.Series,
    approval_threshold: float = 0.15,
) -> list[SegmentMetrics]:
    """Compute performance metrics per segment.

    Parameters
    ----------
    y_true:
        Binary ground truth labels.
    y_score:
        Predicted default probabilities.
    segment_column:
        Categorical series defining segments.
    approval_threshold:
        Threshold below which an applicant is approved.

    Returns
    -------
    list[SegmentMetrics]
        Metrics for each unique segment value, sorted by segment name.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    segments = np.asarray(segment_column)
    col_name = segment_column.name if hasattr(segment_column, "name") else "segment"

    results: list[SegmentMetrics] = []

    for seg_val in sorted(set(segments.tolist())):
        mask = segments == seg_val
        yt = y_true[mask]
        ys = y_score[mask]
        n = int(mask.sum())

        if n < 5:
            logger.warning("segment_too_small", segment=str(seg_val), n=n)
            continue

        pos_rate = float(yt.mean())
        _brier = float(brier_score_loss(yt, ys))
        mean_score = float(ys.mean())
        approval_rate = float((ys < approval_threshold).sum() / n)

        # AUC requires both classes present
        _roc = None
        _pr = None
        if yt.sum() > 0 and yt.sum() < n:
            _roc = float(roc_auc_score(yt, ys))
            _pr = float(average_precision_score(yt, ys))

        seg_metrics = SegmentMetrics(
            segment_column=col_name,
            segment_value=str(seg_val),
            n_samples=n,
            positive_rate=pos_rate,
            roc_auc=_roc,
            pr_auc=_pr,
            brier_score=_brier,
            mean_score=mean_score,
            approval_rate=approval_rate,
        )
        results.append(seg_metrics)

        logger.debug(
            "segment_metrics",
            segment=str(seg_val),
            n=n,
            roc_auc=round(_roc, 5) if _roc else None,
            pos_rate=round(pos_rate, 4),
        )

    return results


# ---------------------------------------------------------------------------
# Segment binning helpers
# ---------------------------------------------------------------------------


def bin_income(income_series: pd.Series, n_bins: int = 5) -> pd.Series:
    """Bin a continuous income column into labeled quantile bands.

    Parameters
    ----------
    income_series:
        Numeric income series.
    n_bins:
        Number of quantile bins.

    Returns
    -------
    pd.Series
        Categorical series with bin labels.
    """
    return pd.qcut(income_series, q=n_bins, labels=False, duplicates="drop").astype(str)


def bin_employment(days_employed: pd.Series) -> pd.Series:
    """Bin employment duration into labeled bands.

    Parameters
    ----------
    days_employed:
        Number of days employed (negative in Home Credit = employed,
        positive 365243 = retired/unemployed).

    Returns
    -------
    pd.Series
        Categorical employment band.
    """

    def _classify(x: float) -> str:
        if x >= 365243:
            return "retired_or_unemployed"
        years = abs(x) / 365.25
        if years < 1:
            return "0-1y"
        if years < 3:
            return "1-3y"
        if years < 7:
            return "3-7y"
        return "7y+"

    return days_employed.apply(_classify)


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------


def compute_segment_report(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    segment_column: pd.Series,
    approval_threshold: float = 0.15,
) -> SegmentReport:
    """Compute a comprehensive segment performance report.

    Parameters
    ----------
    y_true:
        Binary ground truth labels.
    y_score:
        Predicted default probabilities.
    segment_column:
        Categorical series defining segments.
    approval_threshold:
        Approval threshold for rate computation.

    Returns
    -------
    SegmentReport
    """
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    col_name = segment_column.name if hasattr(segment_column, "name") else "segment"

    # Overall metrics
    overall_roc = float(roc_auc_score(y_true_arr, y_score_arr))
    overall_pr = float(average_precision_score(y_true_arr, y_score_arr))

    # Per-segment metrics
    seg_list = segment_metrics(y_true, y_score, segment_column, approval_threshold)

    # Identify best/worst by ROC-AUC (among segments with valid AUC)
    valid_segments = [s for s in seg_list if s.roc_auc is not None]

    best: SegmentMetrics | None
    worst: SegmentMetrics | None
    if valid_segments:
        # roc_auc has been narrowed to non-None by the filter above; the cast
        # avoids mypy seeing `float | None` inside the lambda.
        best = max(valid_segments, key=lambda s: s.roc_auc or 0.0)
        worst = min(valid_segments, key=lambda s: s.roc_auc or 0.0)
        gap = (best.roc_auc or 0.0) - (worst.roc_auc or 0.0)
    else:
        best = seg_list[0] if seg_list else None
        worst = seg_list[0] if seg_list else None
        gap = 0.0

    report = SegmentReport(
        segment_column=col_name,
        n_segments=len(seg_list),
        segments=seg_list,
        overall_roc_auc=overall_roc,
        overall_pr_auc=overall_pr,
        worst_segment=worst.segment_value if worst else "N/A",
        best_segment=best.segment_value if best else "N/A",
        performance_gap=gap,
    )

    logger.info(
        "segment_report",
        column=col_name,
        n_segments=len(seg_list),
        best=report.best_segment,
        worst=report.worst_segment,
        gap=round(gap, 5),
    )
    return report


def compute_multi_segment_reports(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    df: pd.DataFrame,
    segment_columns: list[str],
    approval_threshold: float = 0.15,
) -> list[SegmentReport]:
    """Run segment analysis across multiple segment columns.

    Parameters
    ----------
    y_true:
        Binary ground truth labels.
    y_score:
        Predicted default probabilities.
    df:
        DataFrame containing the segment columns.
    segment_columns:
        List of column names to segment by.
    approval_threshold:
        Approval threshold.

    Returns
    -------
    list[SegmentReport]
    """
    reports = []
    for col in segment_columns:
        if col not in df.columns:
            logger.warning("segment_column_missing", column=col)
            continue
        report = compute_segment_report(y_true, y_score, df[col], approval_threshold)
        reports.append(report)
    return reports


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Segment performance analysis")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--segment-col", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    report = compute_segment_report(
        df["y_true"],
        df["y_score"],
        df[args.segment_col],
    )

    print(f"\nSegment Analysis: {report.segment_column}")
    print(f"  Segments: {report.n_segments}")
    print(f"  Best:     {report.best_segment} | Worst: {report.worst_segment}")
    print(f"  Gap:      {report.performance_gap:.5f}")
    for seg in report.segments:
        roc_str = f"{seg.roc_auc:.5f}" if seg.roc_auc else "N/A"
        print(
            f"    {seg.segment_value:25s}  n={seg.n_samples:6d}  ROC={roc_str}  pos={seg.positive_rate:.3%}"
        )
