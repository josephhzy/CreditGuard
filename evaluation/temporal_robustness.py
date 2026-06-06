"""Temporal robustness analysis for credit scoring models.

Assesses model stability over time by computing performance metrics
per time period, running out-of-time validation, and tracking score
distribution shifts.

**Project status:** The Home Credit aggregated feature matrix has no
canonical application-date column, so this module is available but not
exercised against that dataset. See ``governance/model_card.md``
(Out-of-time validation row) for the written justification.

Typical usage::

    from evaluation.temporal_robustness import compute_temporal_report

    report = compute_temporal_report(y_true, y_score, dates)
    print(report.performance_by_period)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
class PeriodMetrics:
    """Performance metrics for a single time period."""

    period: str
    n_samples: int
    positive_rate: float
    roc_auc: float
    pr_auc: float
    mean_score: float
    std_score: float


@dataclass
class TemporalReport:
    """Complete temporal robustness report."""

    performance_by_period: list[PeriodMetrics]
    oot_roc_auc: float | None = None
    oot_pr_auc: float | None = None
    oot_n_train: int | None = None
    oot_n_test: int | None = None
    score_distribution_by_period: dict[str, dict[str, float]] = field(default_factory=dict)
    stability_summary: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Performance by period
# ---------------------------------------------------------------------------


def performance_by_period(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    dates: pd.Series,
    period: str = "month",
) -> list[PeriodMetrics]:
    """Compute discrimination metrics for each time period.

    Parameters
    ----------
    y_true:
        Binary ground truth labels.
    y_score:
        Predicted default probabilities.
    dates:
        Datetime series for each observation.
    period:
        Grouping period: ``"month"``, ``"quarter"``, or ``"year"``.

    Returns
    -------
    list[PeriodMetrics]
        Metrics per period, sorted chronologically.
    """
    df = pd.DataFrame(
        {
            "y_true": np.asarray(y_true),
            "y_score": np.asarray(y_score),
            "date": pd.to_datetime(dates),
        }
    )

    freq_map = {"month": "ME", "quarter": "QE", "year": "YE"}
    freq = freq_map.get(period)
    if freq is None:
        raise ValueError(f"Unknown period: '{period}'. Use month/quarter/year")

    df["period"] = df["date"].dt.to_period(freq[0])

    results: list[PeriodMetrics] = []

    for period_label, group in df.groupby("period", sort=True):
        n = len(group)
        pos_rate = float(group["y_true"].mean())

        # Need at least 2 classes to compute AUC
        if group["y_true"].nunique() < 2 or n < 10:
            logger.warning("period_skipped_insufficient_data", period=str(period_label), n=n)
            continue

        _roc = roc_auc_score(group["y_true"], group["y_score"])
        _pr = average_precision_score(group["y_true"], group["y_score"])

        metrics = PeriodMetrics(
            period=str(period_label),
            n_samples=n,
            positive_rate=pos_rate,
            roc_auc=float(_roc),
            pr_auc=float(_pr),
            mean_score=float(group["y_score"].mean()),
            std_score=float(group["y_score"].std()),
        )
        results.append(metrics)
        logger.debug(
            "period_metrics",
            period=str(period_label),
            n=n,
            roc_auc=round(_roc, 5),
            pr_auc=round(_pr, 5),
        )

    logger.info("performance_by_period", n_periods=len(results), period_type=period)
    return results


# ---------------------------------------------------------------------------
# Out-of-time validation
# ---------------------------------------------------------------------------


def out_of_time_validation(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_oot: pd.DataFrame,
    y_oot: pd.Series,
) -> dict[str, float]:
    """Run out-of-time validation: train on historical, evaluate on future.

    Parameters
    ----------
    model:
        An unfitted scikit-learn compatible estimator.
    X_train:
        Training feature matrix (historical period).
    y_train:
        Training labels.
    X_oot:
        Out-of-time test feature matrix.
    y_oot:
        Out-of-time test labels.

    Returns
    -------
    dict[str, float]
        Keys: ``"roc_auc"``, ``"pr_auc"``, ``"n_train"``, ``"n_test"``,
        ``"train_pos_rate"``, ``"test_pos_rate"``.
    """
    logger.info("oot_validation_start", n_train=len(X_train), n_test=len(X_oot))

    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_oot)[:, 1]

    _roc = roc_auc_score(y_oot, y_score)
    _pr = average_precision_score(y_oot, y_score)

    result = {
        "roc_auc": float(_roc),
        "pr_auc": float(_pr),
        "n_train": len(X_train),
        "n_test": len(X_oot),
        "train_pos_rate": float(y_train.mean()),
        "test_pos_rate": float(y_oot.mean()),
    }

    logger.info(
        "oot_validation_result",
        roc_auc=round(_roc, 5),
        pr_auc=round(_pr, 5),
    )
    return result


# ---------------------------------------------------------------------------
# Score distribution by period
# ---------------------------------------------------------------------------


def score_distribution_by_period(
    y_score: np.ndarray | pd.Series,
    dates: pd.Series,
    period: str = "month",
) -> dict[str, dict[str, float]]:
    """Compute score distribution statistics by time period.

    Parameters
    ----------
    y_score:
        Predicted scores.
    dates:
        Datetime series.
    period:
        Grouping period.

    Returns
    -------
    dict[str, dict[str, float]]
        Outer key is period label; inner keys are ``"mean"``, ``"std"``,
        ``"p25"``, ``"p50"``, ``"p75"``, ``"n"``.
    """
    df = pd.DataFrame(
        {
            "y_score": np.asarray(y_score),
            "date": pd.to_datetime(dates),
        }
    )

    freq_map = {"month": "ME", "quarter": "QE", "year": "YE"}
    freq = freq_map.get(period)
    if freq is None:
        raise ValueError(f"Unknown period: '{period}'")

    df["period"] = df["date"].dt.to_period(freq[0])

    result: dict[str, dict[str, float]] = {}

    for period_label, group in df.groupby("period", sort=True):
        scores = group["y_score"]
        result[str(period_label)] = {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "p25": float(scores.quantile(0.25)),
            "p50": float(scores.quantile(0.50)),
            "p75": float(scores.quantile(0.75)),
            "n": float(len(scores)),
        }

    logger.info("score_distribution_by_period", n_periods=len(result))
    return result


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------


def compute_temporal_report(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    dates: pd.Series,
    period: str = "month",
    model: Any | None = None,
    X_train: pd.DataFrame | None = None,
    y_train: pd.Series | None = None,
    X_oot: pd.DataFrame | None = None,
    y_oot: pd.Series | None = None,
) -> TemporalReport:
    """Compute a comprehensive temporal robustness report.

    Parameters
    ----------
    y_true:
        Binary ground truth labels.
    y_score:
        Predicted default probabilities.
    dates:
        Datetime series.
    period:
        Grouping period.
    model:
        Optional unfitted model for OOT validation.
    X_train, y_train:
        Training data for OOT.
    X_oot, y_oot:
        Out-of-time test data for OOT.

    Returns
    -------
    TemporalReport
    """
    period_metrics = performance_by_period(y_true, y_score, dates, period=period)
    dist_by_period = score_distribution_by_period(y_score, dates, period=period)

    # OOT validation if data provided
    oot_roc = None
    oot_pr = None
    oot_n_train = None
    oot_n_test = None

    if model is not None and X_train is not None and X_oot is not None:
        oot_result = out_of_time_validation(model, X_train, y_train, X_oot, y_oot)
        oot_roc = oot_result["roc_auc"]
        oot_pr = oot_result["pr_auc"]
        oot_n_train = int(oot_result["n_train"])
        oot_n_test = int(oot_result["n_test"])

    # Stability summary
    if len(period_metrics) >= 2:
        roc_values = [m.roc_auc for m in period_metrics]
        pr_values = [m.pr_auc for m in period_metrics]
        stability = {
            "roc_auc_std": float(np.std(roc_values)),
            "roc_auc_range": float(max(roc_values) - min(roc_values)),
            "pr_auc_std": float(np.std(pr_values)),
            "pr_auc_range": float(max(pr_values) - min(pr_values)),
            "n_periods": float(len(period_metrics)),
        }
    else:
        stability = {}

    report = TemporalReport(
        performance_by_period=period_metrics,
        oot_roc_auc=oot_roc,
        oot_pr_auc=oot_pr,
        oot_n_train=oot_n_train,
        oot_n_test=oot_n_test,
        score_distribution_by_period=dist_by_period,
        stability_summary=stability,
    )

    logger.info(
        "temporal_report",
        n_periods=len(period_metrics),
        has_oot=oot_roc is not None,
        stability=stability,
    )
    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # NOTE: This CLI requires a CSV with columns y_true, y_score, and date.
    # It is NOT runnable against the project's Home Credit feature matrix —
    # the aggregated feature matrix has no canonical application-date column
    # and out-of-time validation was not performed on this dataset.
    # See governance/model_card.md (Out-of-time validation row) for context.
    # To use this script, supply predictions from a dataset that preserves
    # application dates (e.g. a future real-world deployment dataset).
    import argparse
    import json
    from dataclasses import asdict
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Temporal robustness analysis")
    parser.add_argument(
        "--predictions", type=Path, required=True, help="CSV with y_true, y_score, date columns"
    )
    parser.add_argument("--period", type=str, default="month")
    args = parser.parse_args()

    df = pd.read_csv(args.predictions, parse_dates=["date"])
    report = compute_temporal_report(df["y_true"], df["y_score"], df["date"], period=args.period)

    # Custom serialization for PeriodMetrics
    output = {
        "n_periods": len(report.performance_by_period),
        "stability_summary": report.stability_summary,
        "periods": [asdict(m) for m in report.performance_by_period],
    }
    print(json.dumps(output, indent=2))
