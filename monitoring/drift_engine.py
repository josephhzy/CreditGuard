"""Multi-method drift detection engine for model monitoring.

Combines Population Stability Index (PSI), Kolmogorov-Smirnov (KS)
test, and Jensen-Shannon (JS) divergence to detect distribution shifts
across features between a reference and current dataset.

Produces a structured drift report with per-feature scores, ranked
drifting features, overall severity, and root-cause notes.

Typical usage::

    from monitoring.drift_engine import detect_drift, DriftReport

    report = detect_drift(X_reference, X_current)
    print(report.overall_severity)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import structlog
from scipy import stats

from monitoring.psi import calculate_psi
from monitoring.psi import classify_drift as classify_psi

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)

_EPSILON = 1e-10


# ---------------------------------------------------------------------------
# Divergence / test functions
# ---------------------------------------------------------------------------


def js_divergence(
    p: np.ndarray,
    q: np.ndarray,
    *,
    n_bins: int = 50,
) -> float:
    """Jensen-Shannon divergence between two sample distributions.

    JS divergence is the symmetric, bounded version of KL divergence.
    Range: [0, ln(2)] for natural log, or [0, 1] for log base 2.
    This implementation uses natural log.

    Parameters
    ----------
    p, q:
        Sample arrays from two distributions.
    n_bins:
        Number of histogram bins for the density estimate.

    Returns
    -------
    float
        JS divergence value.
    """
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)

    p_arr = p_arr[~np.isnan(p_arr)]
    q_arr = q_arr[~np.isnan(q_arr)]

    if p_arr.size == 0 or q_arr.size == 0:
        return 0.0

    # Common bin edges
    combined = np.concatenate([p_arr, q_arr])
    bin_edges = np.linspace(combined.min(), combined.max(), n_bins + 1)

    p_hist, _ = np.histogram(p_arr, bins=bin_edges, density=True)
    q_hist, _ = np.histogram(q_arr, bins=bin_edges, density=True)

    # Normalise to probability distributions
    p_dist = (p_hist + _EPSILON) / (p_hist.sum() + _EPSILON * len(p_hist))
    q_dist = (q_hist + _EPSILON) / (q_hist.sum() + _EPSILON * len(q_hist))

    m = 0.5 * (p_dist + q_dist)

    kl_pm = float(np.sum(p_dist * np.log(p_dist / m)))
    kl_qm = float(np.sum(q_dist * np.log(q_dist / m)))

    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    return round(max(jsd, 0.0), 6)


def ks_test(
    sample1: np.ndarray,
    sample2: np.ndarray,
) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test.

    Parameters
    ----------
    sample1, sample2:
        Two sample arrays to compare.

    Returns
    -------
    tuple[float, float]
        (KS statistic, p-value).
    """
    s1 = np.asarray(sample1, dtype=float)
    s2 = np.asarray(sample2, dtype=float)

    s1 = s1[~np.isnan(s1)]
    s2 = s2[~np.isnan(s2)]

    if s1.size == 0 or s2.size == 0:
        return 0.0, 1.0

    ks_stat, p_value = stats.ks_2samp(s1, s2)
    return round(float(ks_stat), 6), round(float(p_value), 6)


# ---------------------------------------------------------------------------
# Drift report structures
# ---------------------------------------------------------------------------


@dataclass
class FeatureDrift:
    """Drift scores for a single feature across all methods."""

    feature: str
    psi: float = 0.0
    psi_severity: str = "green"
    ks_statistic: float = 0.0
    ks_pvalue: float = 1.0
    js_divergence: float = 0.0
    mean_shift: float = 0.0
    direction: str = "stable"  # "shift_up", "shift_down", "stable"


@dataclass
class DriftReport:
    """Comprehensive drift detection report."""

    per_feature: list[FeatureDrift]
    top_drifting_features: list[str]
    overall_severity: str  # "green", "amber", "red"
    n_features_green: int = 0
    n_features_amber: int = 0
    n_features_red: int = 0
    root_cause_notes: list[str] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert per-feature drift to a DataFrame."""
        rows = []
        for fd in self.per_feature:
            rows.append(
                {
                    "feature": fd.feature,
                    "psi": fd.psi,
                    "psi_severity": fd.psi_severity,
                    "ks_statistic": fd.ks_statistic,
                    "ks_pvalue": fd.ks_pvalue,
                    "js_divergence": fd.js_divergence,
                    "mean_shift": fd.mean_shift,
                    "direction": fd.direction,
                }
            )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main drift detection
# ---------------------------------------------------------------------------


def detect_drift(
    X_reference: np.ndarray | pd.DataFrame,
    X_current: np.ndarray | pd.DataFrame,
    *,
    feature_names: list[str] | None = None,
    methods: Sequence[str] = ("psi", "ks", "js"),
    n_bins: int = 10,
    top_k: int = 5,
) -> DriftReport:
    """Run multi-method drift detection across all features.

    Parameters
    ----------
    X_reference:
        Reference dataset (training / baseline period).
    X_current:
        Current dataset (production / monitoring period).
    feature_names:
        Feature names. Inferred from DataFrame columns or auto-generated.
    methods:
        Which drift methods to run: "psi", "ks", "js".
    n_bins:
        Number of bins for PSI and JS divergence.
    top_k:
        Number of top drifting features to highlight.

    Returns
    -------
    DriftReport
        Comprehensive drift report with per-feature scores and overall severity.
    """
    # Convert to numpy
    if isinstance(X_reference, pd.DataFrame):
        if feature_names is None:
            feature_names = list(X_reference.columns)
        X_ref = X_reference.values.astype(float)
    else:
        X_ref = np.asarray(X_reference, dtype=float)

    if isinstance(X_current, pd.DataFrame):
        X_cur = X_current.values.astype(float)
    else:
        X_cur = np.asarray(X_current, dtype=float)

    if X_ref.ndim == 1:
        X_ref = X_ref.reshape(-1, 1)
    if X_cur.ndim == 1:
        X_cur = X_cur.reshape(-1, 1)

    if X_ref.shape[1] != X_cur.shape[1]:
        raise ValueError(
            f"Feature count mismatch: reference={X_ref.shape[1]}, current={X_cur.shape[1]}"
        )

    n_features = X_ref.shape[1]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    per_feature: list[FeatureDrift] = []
    n_green = n_amber = n_red = 0

    for i, fname in enumerate(feature_names):
        ref_col = X_ref[:, i]
        cur_col = X_cur[:, i]

        fd = FeatureDrift(feature=fname)

        if "psi" in methods:
            fd.psi = calculate_psi(ref_col, cur_col, n_bins=n_bins)
            fd.psi_severity = classify_psi(fd.psi)

        if "ks" in methods:
            fd.ks_statistic, fd.ks_pvalue = ks_test(ref_col, cur_col)

        if "js" in methods:
            fd.js_divergence = js_divergence(ref_col, cur_col, n_bins=n_bins * 5)

        # Mean shift
        ref_clean = ref_col[~np.isnan(ref_col)]
        cur_clean = cur_col[~np.isnan(cur_col)]
        if ref_clean.size > 0 and cur_clean.size > 0:
            ref_mean = float(np.mean(ref_clean))
            cur_mean = float(np.mean(cur_clean))
            fd.mean_shift = round(cur_mean - ref_mean, 6)
            if abs(fd.mean_shift) < 0.01 * max(abs(ref_mean), 1e-6):
                fd.direction = "stable"
            elif fd.mean_shift > 0:
                fd.direction = "shift_up"
            else:
                fd.direction = "shift_down"

        # Classify severity based on PSI
        if fd.psi_severity == "red":
            n_red += 1
        elif fd.psi_severity == "amber":
            n_amber += 1
        else:
            n_green += 1

        per_feature.append(fd)

    # Rank by PSI for top drifting features
    sorted_features = sorted(per_feature, key=lambda x: x.psi, reverse=True)
    top_drifting = [f.feature for f in sorted_features[:top_k] if f.psi > 0]

    # Overall severity: worst of any feature
    if n_red > 0:
        overall = "red"
    elif n_amber > 0:
        overall = "amber"
    else:
        overall = "green"

    # Root-cause notes
    root_cause_notes: list[str] = []
    for fd in sorted_features[:top_k]:
        if fd.psi_severity in ("amber", "red"):
            root_cause_notes.append(
                f"{fd.feature}: PSI={fd.psi:.4f} [{fd.psi_severity}], "
                f"mean shifted {fd.direction} by {fd.mean_shift:+.4f}"
            )

    report = DriftReport(
        per_feature=per_feature,
        top_drifting_features=top_drifting,
        overall_severity=overall,
        n_features_green=n_green,
        n_features_amber=n_amber,
        n_features_red=n_red,
        root_cause_notes=root_cause_notes,
    )

    logger.info(
        "drift_detection_complete",
        overall=overall,
        n_features=n_features,
        n_red=n_red,
        n_amber=n_amber,
        top_drifting=top_drifting,
    )
    return report


# ---------------------------------------------------------------------------
# Subgroup drift
# ---------------------------------------------------------------------------


def subgroup_drift(
    X_reference: pd.DataFrame,
    X_current: pd.DataFrame,
    group_col: str,
    *,
    score_col: str | None = None,
    n_bins: int = 10,
) -> dict[str, DriftReport]:
    """Run drift detection separately per subgroup.

    Parameters
    ----------
    X_reference:
        Reference data with a grouping column.
    X_current:
        Current data with the same grouping column.
    group_col:
        Column name for subgroup splitting.
    score_col:
        If provided, only compute drift on this column. Otherwise all numeric columns.
    n_bins:
        Number of bins for drift methods.

    Returns
    -------
    dict[str, DriftReport]
        One DriftReport per subgroup value.
    """
    if group_col not in X_reference.columns or group_col not in X_current.columns:
        raise ValueError(f"Group column '{group_col}' not found in data.")

    groups = set(X_reference[group_col].dropna().unique()) & set(
        X_current[group_col].dropna().unique()
    )

    if score_col:
        feat_cols = [score_col]
    else:
        feat_cols = [
            c
            for c in X_reference.columns
            if c != group_col and np.issubdtype(X_reference[c].dtype, np.number)
        ]

    results: dict[str, DriftReport] = {}

    for grp in sorted(groups):
        ref_grp = X_reference.loc[X_reference[group_col] == grp, feat_cols]
        cur_grp = X_current.loc[X_current[group_col] == grp, feat_cols]

        if ref_grp.empty or cur_grp.empty:
            continue

        report = detect_drift(
            ref_grp,
            cur_grp,
            feature_names=feat_cols,
            n_bins=n_bins,
        )
        results[str(grp)] = report

    logger.info("subgroup_drift_complete", n_groups=len(results))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    n_ref = 5000
    n_cur = 5000
    n_feats = 6

    X_ref_demo = rng.normal(0, 1, size=(n_ref, n_feats))
    X_cur_demo = X_ref_demo.copy()[:n_cur, :]

    # Inject drift in features 1 and 3
    X_cur_demo[:, 1] += rng.normal(0.5, 0.2, size=n_cur)
    X_cur_demo[:, 3] *= 1.5

    feat_names = ["income", "age", "debt_ratio", "tenure", "credit_util", "enquiry_count"]

    print("=== Multi-Method Drift Detection ===")
    report = detect_drift(
        X_ref_demo,
        X_cur_demo,
        feature_names=feat_names,
        methods=["psi", "ks", "js"],
    )
    print(f"Overall severity: {report.overall_severity.upper()}")
    print(f"Top drifting: {report.top_drifting_features}")
    print(
        f"Green/Amber/Red: {report.n_features_green}/{report.n_features_amber}/{report.n_features_red}"
    )
    print("\nRoot cause notes:")
    for note in report.root_cause_notes:
        print(f"  - {note}")

    print("\n=== Per-Feature Detail ===")
    df = report.to_dataframe()
    print(df.to_string(index=False))
