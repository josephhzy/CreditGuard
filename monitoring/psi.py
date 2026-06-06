"""Population Stability Index (PSI) for distribution monitoring.

PSI measures how much a distribution has shifted between a reference
(training) population and a current (production) population. It is
a widely used metric for credit model monitoring, referenced in model-risk
frameworks such as SR 11-7, SS1/23, and MAS FEAT.

Traffic-light classification:
  - Green:  PSI < 0.10 (no significant shift)
  - Amber:  0.10 <= PSI < 0.20 (moderate shift, investigate)
  - Red:    PSI >= 0.20 (significant shift, action required)

Typical usage::

    from monitoring.psi import calculate_psi, psi_per_feature, classify_drift

    psi_value = calculate_psi(train_scores, prod_scores)
    severity = classify_drift(psi_value)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import structlog
import yaml

logger = structlog.get_logger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "train.yaml"
_EPSILON = 1e-10


def _load_psi_thresholds(config_path: Path | None = None) -> dict[str, float]:
    """Load PSI traffic-light thresholds from config."""
    path = config_path or _CONFIG_PATH
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg["monitoring"]["psi"]


# ---------------------------------------------------------------------------
# Core PSI calculation
# ---------------------------------------------------------------------------


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    n_bins: int = 10,
    method: Literal["fixed", "quantile"] = "fixed",
) -> float:
    """Calculate the Population Stability Index between two distributions.

    PSI = SUM( (actual_pct - expected_pct) * ln(actual_pct / expected_pct) )

    Parameters
    ----------
    expected:
        Reference distribution (typically training data).
    actual:
        Current distribution (typically production data).
    n_bins:
        Number of bins for the histogram comparison.
    method:
        Binning strategy:
        - "fixed": equal-width bins across the combined range
        - "quantile": bins based on quantiles of the expected distribution

    Returns
    -------
    float
        PSI value. Lower is better (less shift).
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    # Remove NaN
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if expected.size == 0 or actual.size == 0:
        logger.warning("psi_empty_input", expected_size=expected.size, actual_size=actual.size)
        return 0.0

    # Determine bin edges
    if method == "quantile":
        bin_edges = np.unique(np.percentile(expected, np.linspace(0, 100, n_bins + 1)))
        if len(bin_edges) < 3:
            # Fall back to fixed if quantiles collapse
            logger.warning("psi_quantile_fallback", n_unique_edges=len(bin_edges))
            combined = np.concatenate([expected, actual])
            bin_edges = np.linspace(combined.min(), combined.max(), n_bins + 1)
    else:
        combined = np.concatenate([expected, actual])
        bin_edges = np.linspace(combined.min(), combined.max(), n_bins + 1)

    # Compute bin proportions
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = (expected_counts + _EPSILON) / (
        expected_counts.sum() + _EPSILON * len(expected_counts)
    )
    actual_pct = (actual_counts + _EPSILON) / (actual_counts.sum() + _EPSILON * len(actual_counts))

    psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))

    logger.info(
        "psi_calculated",
        method=method,
        n_bins=n_bins,
        psi=round(psi, 6),
        expected_n=len(expected),
        actual_n=len(actual),
    )
    return round(psi, 6)


# ---------------------------------------------------------------------------
# Per-feature PSI
# ---------------------------------------------------------------------------


def psi_per_feature(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    feature_names: list[str] | None = None,
    n_bins: int = 10,
    method: Literal["fixed", "quantile"] = "fixed",
) -> dict[str, float]:
    """Calculate PSI for each feature column.

    Parameters
    ----------
    X_train:
        Training data (n_samples, n_features).
    X_test:
        Test/production data (n_samples, n_features).
    feature_names:
        Optional feature names. Auto-generated if not provided.
    n_bins:
        Number of bins per feature.
    method:
        Binning method ("fixed" or "quantile").

    Returns
    -------
    dict[str, float]
        PSI value per feature, sorted descending by PSI.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"Feature count mismatch: train has {X_train.shape[1]}, test has {X_test.shape[1]}."
        )

    n_features = X_train.shape[1]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) does not match "
            f"number of features ({n_features})."
        )

    results: dict[str, float] = {}
    for i, name in enumerate(feature_names):
        results[name] = calculate_psi(
            X_train[:, i],
            X_test[:, i],
            n_bins=n_bins,
            method=method,
        )

    # Sort by PSI descending
    results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    logger.info(
        "psi_per_feature_complete",
        n_features=n_features,
        n_red=sum(1 for v in results.values() if v >= 0.20),
        n_amber=sum(1 for v in results.values() if 0.10 <= v < 0.20),
    )
    return results


# ---------------------------------------------------------------------------
# Traffic-light classification
# ---------------------------------------------------------------------------


def classify_drift(
    psi_value: float,
    *,
    config_path: Path | None = None,
) -> str:
    """Classify a PSI value into a traffic-light severity.

    Parameters
    ----------
    psi_value:
        The PSI value to classify.
    config_path:
        Optional config path to load thresholds.

    Returns
    -------
    str
        "green", "amber", or "red".
    """
    thresholds = _load_psi_thresholds(config_path)
    green_limit = thresholds.get("threshold_green", 0.10)
    amber_limit = thresholds.get("threshold_amber", 0.20)

    if psi_value < green_limit:
        return "green"
    elif psi_value < amber_limit:
        return "amber"
    else:
        return "red"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Simulate a moderate distribution shift
    expected_demo = rng.normal(0.2, 0.1, size=5000)
    actual_demo = rng.normal(0.25, 0.12, size=5000)

    print("=== PSI Calculation ===")
    for method in ("fixed", "quantile"):
        psi_val = calculate_psi(expected_demo, actual_demo, method=method)  # type: ignore[arg-type]
        severity = classify_drift(psi_val)
        print(f"  Method={method}: PSI={psi_val:.6f} [{severity.upper()}]")

    print("\n=== Per-feature PSI ===")
    X_train_demo = rng.normal(0, 1, size=(3000, 5))
    X_test_demo = X_train_demo + rng.normal(0, 0.1, size=(3000, 5))
    X_test_demo[:, 2] += 0.8  # Inject drift in feature 2

    feat_psi = psi_per_feature(
        X_train_demo,
        X_test_demo,
        feature_names=["income", "age", "debt_ratio", "tenure", "credit_util"],
    )
    for name, val in feat_psi.items():
        sev = classify_drift(val)
        print(f"  {name}: PSI={val:.6f} [{sev.upper()}]")
