"""Portfolio-level risk analytics for credit decisioning.

Provides aggregate risk analysis across the applicant pool and existing
portfolio. Covers risk score distributions, concentration risk by
segment, temporal drift in portfolio composition, and exposure
sensitivity analysis.

These metrics support portfolio management, risk committee reporting,
and stress-testing exercises.

Typical usage::

    from decisioning.portfolio_analytics import risk_distribution, concentration_risk

    dist = risk_distribution(scores)
    conc = concentration_risk(scores, segments)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import structlog

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Risk distribution
# ---------------------------------------------------------------------------


@dataclass
class RiskDistribution:
    """Histogram of risk scores across the applicant pool."""

    bin_edges: list[float]
    bin_counts: list[int]
    bin_pcts: list[float]
    mean_score: float
    median_score: float
    std_score: float
    p5: float
    p25: float
    p75: float
    p95: float


def risk_distribution(
    scores: np.ndarray,
    *,
    n_bins: int = 20,
) -> RiskDistribution:
    """Compute the risk score distribution across a pool.

    Parameters
    ----------
    scores:
        Array of raw risk scores in [0, 1] (raw-score scale — same as
        ``decisioning/decision_engine.py``, not the calibrated PD).
    n_bins:
        Number of histogram bins.

    Returns
    -------
    RiskDistribution
        Summary statistics and histogram of the score distribution.
    """
    scores = np.asarray(scores, dtype=float)
    scores = scores[~np.isnan(scores)]

    if scores.size == 0:
        logger.warning("risk_distribution_empty_input")
        return RiskDistribution(
            bin_edges=[],
            bin_counts=[],
            bin_pcts=[],
            mean_score=0.0,
            median_score=0.0,
            std_score=0.0,
            p5=0.0,
            p25=0.0,
            p75=0.0,
            p95=0.0,
        )

    counts, edges = np.histogram(scores, bins=n_bins, range=(0.0, 1.0))
    total = counts.sum()
    pcts = (counts / total * 100).tolist() if total > 0 else [0.0] * len(counts)

    dist = RiskDistribution(
        bin_edges=[round(float(e), 4) for e in edges],
        bin_counts=counts.tolist(),
        bin_pcts=[round(p, 2) for p in pcts],
        mean_score=round(float(np.mean(scores)), 6),
        median_score=round(float(np.median(scores)), 6),
        std_score=round(float(np.std(scores)), 6),
        p5=round(float(np.percentile(scores, 5)), 6),
        p25=round(float(np.percentile(scores, 25)), 6),
        p75=round(float(np.percentile(scores, 75)), 6),
        p95=round(float(np.percentile(scores, 95)), 6),
    )

    logger.info(
        "risk_distribution_computed",
        n_scores=len(scores),
        mean=dist.mean_score,
        median=dist.median_score,
    )
    return dist


# ---------------------------------------------------------------------------
# Concentration risk
# ---------------------------------------------------------------------------


@dataclass
class ConcentrationResult:
    """Risk concentration across portfolio segments."""

    segment_stats: pd.DataFrame  # columns: segment, count, pct, mean_score, median_score, std_score
    herfindahl_index: float  # portfolio concentration measure
    top_segment: str
    top_segment_share: float


def concentration_risk(
    scores: np.ndarray,
    segments: np.ndarray | Sequence[str],
) -> ConcentrationResult:
    """Analyse risk concentration by portfolio segment.

    Parameters
    ----------
    scores:
        Raw risk scores for each applicant.
    segments:
        Segment labels (e.g., industry, region, credit grade) per applicant.

    Returns
    -------
    ConcentrationResult
        Per-segment statistics and overall concentration measure.
    """
    scores = np.asarray(scores, dtype=float)
    segments = np.asarray(segments)

    if scores.shape[0] != segments.shape[0]:
        raise ValueError("scores and segments must have the same length.")
    if scores.size == 0:
        logger.warning("concentration_risk_empty_input")
        return ConcentrationResult(
            segment_stats=pd.DataFrame(),
            herfindahl_index=0.0,
            top_segment="",
            top_segment_share=0.0,
        )

    df = pd.DataFrame({"score": scores, "segment": segments})
    agg = df.groupby("segment")["score"].agg(["count", "mean", "median", "std"]).reset_index()
    agg.columns = ["segment", "count", "mean_score", "median_score", "std_score"]
    agg["std_score"] = agg["std_score"].fillna(0.0)
    total = agg["count"].sum()
    agg["pct"] = (agg["count"] / total * 100).round(2)
    agg = agg.sort_values("count", ascending=False).reset_index(drop=True)

    # Herfindahl-Hirschman Index: sum of squared shares
    shares = agg["count"].values / total
    hhi = float(np.sum(shares**2))

    top_row = agg.iloc[0]

    result = ConcentrationResult(
        segment_stats=agg,
        herfindahl_index=round(hhi, 6),
        top_segment=str(top_row["segment"]),
        top_segment_share=round(float(top_row["pct"]), 2),
    )

    logger.info(
        "concentration_risk_computed",
        n_segments=len(agg),
        hhi=result.herfindahl_index,
        top_segment=result.top_segment,
    )
    return result


# ---------------------------------------------------------------------------
# Segment deterioration
# ---------------------------------------------------------------------------


@dataclass
class DeteriorationResult:
    """Segments with the largest score increases between two periods."""

    segment_changes: pd.DataFrame  # segment, mean_t0, mean_t1, delta, pct_change
    top_deteriorating: list[str]


def top_deteriorating_segments(
    scores_t0: np.ndarray,
    scores_t1: np.ndarray,
    segments_t0: np.ndarray | Sequence[str],
    segments_t1: np.ndarray | Sequence[str],
    *,
    top_k: int = 5,
) -> DeteriorationResult:
    """Find segments with the biggest score increases between two periods.

    Parameters
    ----------
    scores_t0, scores_t1:
        Raw risk scores at time 0 and time 1.
    segments_t0, segments_t1:
        Segment labels at time 0 and time 1.
    top_k:
        Number of top deteriorating segments to surface.

    Returns
    -------
    DeteriorationResult
        Per-segment change and ranked deterioration list.
    """
    df0 = pd.DataFrame(
        {"score": np.asarray(scores_t0, dtype=float), "segment": np.asarray(segments_t0)}
    )
    df1 = pd.DataFrame(
        {"score": np.asarray(scores_t1, dtype=float), "segment": np.asarray(segments_t1)}
    )

    mean0 = df0.groupby("segment")["score"].mean().rename("mean_t0")
    mean1 = df1.groupby("segment")["score"].mean().rename("mean_t1")

    merged = pd.concat([mean0, mean1], axis=1).dropna()
    merged["delta"] = merged["mean_t1"] - merged["mean_t0"]
    merged["pct_change"] = np.where(
        merged["mean_t0"] > 0,
        (merged["delta"] / merged["mean_t0"] * 100).round(2),
        0.0,
    )
    merged = merged.sort_values("delta", ascending=False).reset_index()
    merged.columns = ["segment", "mean_t0", "mean_t1", "delta", "pct_change"]

    top_segments = merged.head(top_k)["segment"].tolist()

    logger.info(
        "deterioration_analysis_complete",
        n_segments=len(merged),
        top_deteriorating=top_segments,
    )

    return DeteriorationResult(
        segment_changes=merged,
        top_deteriorating=top_segments,
    )


# ---------------------------------------------------------------------------
# Portfolio drift over time
# ---------------------------------------------------------------------------


@dataclass
class PortfolioDrift:
    """How overall portfolio risk changes across periods."""

    period_stats: pd.DataFrame  # period, mean_score, median_score, std_score, n
    mean_trend_direction: str  # "increasing", "decreasing", "stable"
    total_mean_shift: float


def portfolio_drift(
    scores_by_period: dict[str, np.ndarray],
) -> PortfolioDrift:
    """Track portfolio risk score distribution changes over time.

    Parameters
    ----------
    scores_by_period:
        Mapping of period label -> array of raw risk scores for that period.

    Returns
    -------
    PortfolioDrift
        Per-period summary and overall trend direction.
    """
    if not scores_by_period:
        logger.warning("portfolio_drift_no_periods")
        return PortfolioDrift(
            period_stats=pd.DataFrame(),
            mean_trend_direction="stable",
            total_mean_shift=0.0,
        )

    rows: list[dict[str, Any]] = []
    for period, scores in scores_by_period.items():
        arr = np.asarray(scores, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            continue
        rows.append(
            {
                "period": period,
                "mean_score": round(float(np.mean(arr)), 6),
                "median_score": round(float(np.median(arr)), 6),
                "std_score": round(float(np.std(arr)), 6),
                "n": len(arr),
            }
        )

    if not rows:
        return PortfolioDrift(
            period_stats=pd.DataFrame(),
            mean_trend_direction="stable",
            total_mean_shift=0.0,
        )

    df = pd.DataFrame(rows)
    means = df["mean_score"].values
    total_shift = float(means[-1] - means[0])

    if abs(total_shift) < 0.01:
        direction = "stable"
    elif total_shift > 0:
        direction = "increasing"
    else:
        direction = "decreasing"

    logger.info(
        "portfolio_drift_computed",
        n_periods=len(df),
        direction=direction,
        total_shift=round(total_shift, 4),
    )

    return PortfolioDrift(
        period_stats=df,
        mean_trend_direction=direction,
        total_mean_shift=round(total_shift, 6),
    )


# ---------------------------------------------------------------------------
# Exposure at threshold
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExposureResult:
    """Exposure impact if threshold changes."""

    threshold: float
    n_above: int
    n_below: int
    exposure_above: float
    exposure_below: float
    total_exposure: float
    pct_exposure_above: float


def exposure_at_threshold(
    scores: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
) -> ExposureResult:
    """Calculate total exposure above and below a score threshold.

    Parameters
    ----------
    scores:
        Raw risk scores per applicant.
    amounts:
        Loan/credit amounts per applicant.
    threshold:
        Score threshold to split on.

    Returns
    -------
    ExposureResult
        Exposure on each side of the threshold.
    """
    scores = np.asarray(scores, dtype=float)
    amounts = np.asarray(amounts, dtype=float)

    if scores.shape[0] != amounts.shape[0]:
        raise ValueError("scores and amounts must have the same length.")

    above_mask = scores >= threshold
    below_mask = ~above_mask

    exposure_above = float(amounts[above_mask].sum())
    exposure_below = float(amounts[below_mask].sum())
    total = exposure_above + exposure_below

    result = ExposureResult(
        threshold=threshold,
        n_above=int(above_mask.sum()),
        n_below=int(below_mask.sum()),
        exposure_above=round(exposure_above, 2),
        exposure_below=round(exposure_below, 2),
        total_exposure=round(total, 2),
        pct_exposure_above=round(exposure_above / total * 100, 2) if total > 0 else 0.0,
    )

    logger.info(
        "exposure_at_threshold",
        threshold=threshold,
        pct_above=result.pct_exposure_above,
        total=result.total_exposure,
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 3000

    scores_demo = rng.beta(2, 8, size=n)
    # Illustrative placeholders only — not Home Credit dataset attributes.
    segments_demo = rng.choice(["SME", "Retail", "Corporate", "Mortgage"], size=n)
    amounts_demo = rng.lognormal(10, 1.5, size=n)

    print("=== Risk Distribution ===")
    dist = risk_distribution(scores_demo)
    print(f"  Mean: {dist.mean_score}, Median: {dist.median_score}, Std: {dist.std_score}")
    print(f"  P5-P95: [{dist.p5}, {dist.p95}]")

    print("\n=== Concentration Risk ===")
    conc = concentration_risk(scores_demo, segments_demo)
    print(conc.segment_stats.to_string(index=False))
    print(f"  HHI: {conc.herfindahl_index}")

    print("\n=== Exposure at Threshold ===")
    exp = exposure_at_threshold(scores_demo, amounts_demo, threshold=0.30)
    print(f"  Above: {exp.n_above} ({exp.pct_exposure_above}% exposure)")
    print(f"  Below: {exp.n_below}")

    print("\n=== Portfolio Drift ===")
    drift = portfolio_drift(
        {
            "2024-Q1": rng.beta(2, 8, size=1000),
            "2024-Q2": rng.beta(2, 7.5, size=1000),
            "2024-Q3": rng.beta(2.2, 7, size=1000),
            "2024-Q4": rng.beta(2.5, 6.5, size=1000),
        }
    )
    print(drift.period_stats.to_string(index=False))
    print(f"  Trend: {drift.mean_trend_direction} (shift: {drift.total_mean_shift})")
