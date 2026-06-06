"""Feature quality validation.

Provides automated checks for common feature-engineering pitfalls:
infinite values, constant features, high-correlation pairs, suspicious
target correlations, and completeness reporting. Returns a structured
``FeatureQualityReport`` so the pipeline can fail loudly on data issues.

All thresholds are configurable but ship with sensible defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import structlog

if TYPE_CHECKING:
    import pandas as pd

log = structlog.get_logger(__name__)


@dataclass
class FeatureQualityReport:
    """Container for quality-check results."""

    infinite_features: list[str] = field(default_factory=list)
    constant_features: list[str] = field(default_factory=list)
    high_corr_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    suspicious_target_corr: list[tuple[str, float]] = field(default_factory=list)
    completeness: dict[str, float] = field(default_factory=dict)
    n_features: int = 0
    passed: bool = True

    def summary(self) -> str:
        """Human-readable summary of the report."""
        lines = [
            f"Feature Quality Report ({self.n_features} features)",
            "=" * 50,
            f"  Infinite values:        {len(self.infinite_features)} features",
            f"  Constant/near-constant: {len(self.constant_features)} features",
            f"  Highly correlated pairs: {len(self.high_corr_pairs)} pairs",
            f"  Suspicious target corr:  {len(self.suspicious_target_corr)} features",
            f"  Overall pass:            {'YES' if self.passed else 'NO'}",
        ]
        if self.infinite_features:
            lines.append(f"\n  Infinite: {self.infinite_features[:10]}")
        if self.constant_features:
            lines.append(f"  Constant: {self.constant_features[:10]}")
        if self.high_corr_pairs:
            top_pairs = [(a, b, f"{r:.3f}") for a, b, r in self.high_corr_pairs[:3]]
            lines.append(f"  Top corr pairs: {top_pairs}")
        if self.suspicious_target_corr:
            top_target = [(f, f"{r:.3f}") for f, r in self.suspicious_target_corr[:3]]
            lines.append(f"  Suspicious target corr: {top_target}")
        return "\n".join(lines)


def run_quality_checks(
    features: pd.DataFrame,
    target: pd.Series | None = None,
    *,
    variance_threshold: float = 0.001,
    correlation_threshold: float = 0.95,
    target_corr_threshold: float = 0.5,
) -> FeatureQualityReport:
    """Run all feature quality checks.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix (no ID or target columns).
    target : pd.Series, optional
        Binary target vector aligned with *features*. If provided,
        feature-target correlation checks are run.
    variance_threshold : float
        Features with variance below this are flagged as constant.
    correlation_threshold : float
        Feature pairs with absolute correlation above this are flagged.
    target_corr_threshold : float
        Features with absolute target correlation above this are flagged
        as suspicious (possible leakage).

    Returns
    -------
    FeatureQualityReport
    """
    report = FeatureQualityReport(n_features=len(features.columns))

    # Only check numeric columns
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()

    report.infinite_features = _check_infinite(features[numeric_cols])
    report.constant_features = _check_constant(features[numeric_cols], threshold=variance_threshold)
    report.high_corr_pairs = _check_high_correlation(
        features[numeric_cols], threshold=correlation_threshold
    )
    if target is not None:
        report.suspicious_target_corr = _check_target_correlation(
            features[numeric_cols], target, threshold=target_corr_threshold
        )
    report.completeness = _check_completeness(features)

    # Mark as failed if there are infinite values (they break most models)
    if report.infinite_features:
        report.passed = False
        log.warning(
            "quality check FAILED: infinite values detected",
            features=report.infinite_features[:5],
        )
    else:
        log.info("quality checks passed", n_features=report.n_features)

    return report


def _check_infinite(df: pd.DataFrame) -> list[str]:
    """Return column names that contain infinite values."""
    inf_mask = np.isinf(df.select_dtypes(include=[np.number]))
    cols_with_inf = inf_mask.any(axis=0)
    result = cols_with_inf[cols_with_inf].index.tolist()
    if result:
        log.warning("infinite values found", count=len(result), features=result[:5])
    return result


def _check_constant(df: pd.DataFrame, threshold: float) -> list[str]:
    """Return column names with variance below *threshold*."""
    variances = df.var(numeric_only=True)
    constant = variances[variances < threshold]
    result = constant.index.tolist()
    if result:
        log.info("constant/near-constant features", count=len(result), features=result[:5])
    return result


def _check_high_correlation(
    df: pd.DataFrame,
    threshold: float,
) -> list[tuple[str, str, float]]:
    """Return pairs of features with absolute correlation above *threshold*.

    Only numeric columns are checked. Pairs are deduplicated (A,B not B,A).
    Returns at most 50 pairs to avoid flooding logs on wide datasets.
    """
    if df.shape[1] < 2:
        return []

    # Sample if too many features (correlation matrix is O(n^2))
    cols = df.columns.tolist()
    if len(cols) > 300:
        log.info("sampling 300 features for correlation check", total=len(cols))
        rng = np.random.default_rng(42)
        cols = rng.choice(cols, size=300, replace=False).tolist()

    # Deduplicate column names before computing correlation to avoid
    # ambiguous indexing when duplicate feature names exist.
    deduped = df[cols].copy()
    seen: dict[str, int] = {}
    new_names: list[str] = []
    for c in deduped.columns:
        if c in seen:
            seen[c] += 1
            new_names.append(f"{c}__dup{seen[c]}")
        else:
            seen[c] = 0
            new_names.append(c)
    deduped.columns = new_names

    corr = deduped.corr().abs()
    # Extract upper triangle pairs above threshold
    pairs: list[tuple[str, str, float]] = []
    corr_vals = corr.values
    col_names = corr.columns.tolist()
    n = len(col_names)
    for i in range(n):
        for j in range(i + 1, n):
            val = corr_vals[i, j]
            if not np.isnan(val) and val > threshold:
                pairs.append((col_names[i], col_names[j], round(float(val), 4)))
    pairs.sort(key=lambda x: x[2], reverse=True)
    if pairs:
        log.info("high correlation pairs", count=len(pairs))
    return pairs[:50]


def _check_target_correlation(
    df: pd.DataFrame,
    target: pd.Series,
    threshold: float,
) -> list[tuple[str, float]]:
    """Return features with absolute target correlation above *threshold*.

    High correlation with the target may indicate data leakage rather
    than a genuinely predictive feature.
    """
    # Align indices
    common = df.index.intersection(target.index)
    if len(common) == 0:
        return []

    correlations = df.loc[common].corrwith(target.loc[common]).abs()
    suspicious = correlations[correlations > threshold].sort_values(ascending=False)
    result = [(col, round(float(val), 4)) for col, val in suspicious.items()]
    if result:
        log.warning(
            "suspicious target correlations (possible leakage)",
            count=len(result),
            features=result[:5],
        )
    return result


def _check_completeness(df: pd.DataFrame) -> dict[str, float]:
    """Return per-column completeness as fraction of non-null values.

    A completeness of 1.0 means no missing data; 0.0 means entirely null.
    """
    n_rows = len(df)
    if n_rows == 0:
        return {}
    completeness = (df.notna().sum() / n_rows).round(4)
    low = completeness[completeness < 0.5]
    if len(low) > 0:
        log.info(
            "low-completeness features (<50%)",
            count=len(low),
            features=low.index.tolist()[:5],
        )
    return completeness.to_dict()
