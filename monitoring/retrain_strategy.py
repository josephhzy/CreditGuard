"""Retraining strategy recommendations for credit models.

Provides experiments and analysis to determine the optimal retraining
frequency: rolling window vs. expanding window, performance decay
curves, and feature stability across retrains.

The goal is to answer: "How often should we retrain, and what evidence
supports that cadence?"

Typical usage::

    from monitoring.retrain_strategy import (
        rolling_window_experiment,
        recommend_retrain_frequency,
    )

    results = rolling_window_experiment(data, window_months=6)
    recommendation = recommend_retrain_frequency(decay_curve)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import structlog
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data windowing helpers
# ---------------------------------------------------------------------------


def _get_period_boundaries(
    dates: pd.Series,
    window_months: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Generate non-overlapping period boundaries from a date series."""
    dates = pd.to_datetime(dates)
    min_date = dates.min()
    max_date = dates.max()

    boundaries: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    current = min_date
    while current < max_date:
        end = current + pd.DateOffset(months=window_months)
        if end > max_date:
            end = max_date + pd.Timedelta(days=1)
        boundaries.append((current, end))
        current = end

    return boundaries


# ---------------------------------------------------------------------------
# Experiment results
# ---------------------------------------------------------------------------


@dataclass
class WindowExperimentResult:
    """Results from a rolling or expanding window experiment."""

    strategy: str  # "rolling" or "expanding"
    window_months: int
    period_metrics: (
        pd.DataFrame
    )  # train_start, train_end, test_start, test_end, roc_auc, pr_auc, brier, n_train, n_test
    mean_roc_auc: float
    mean_pr_auc: float
    std_pr_auc: float
    trend: str  # "stable", "degrading", "improving"


def rolling_window_experiment(
    X: pd.DataFrame,
    y: np.ndarray | pd.Series,
    dates: pd.Series,
    *,
    window_months: int = 6,
    test_months: int = 3,
    model_factory: Any = None,
) -> WindowExperimentResult:
    """Simulate rolling-window retraining and evaluate on subsequent period.

    Uses a simple logistic regression if no model factory is provided,
    to measure how discrimination degrades over time.

    Parameters
    ----------
    X:
        Feature DataFrame.
    y:
        Binary target array.
    dates:
        Date column for temporal splitting.
    window_months:
        Training window size in months.
    test_months:
        Test window size after each training window.
    model_factory:
        Callable that returns a fitted model with ``.predict_proba()``.
        If ``None``, uses LogisticRegression.

    Returns
    -------
    WindowExperimentResult
        Per-period metrics and overall summary.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    dates = pd.to_datetime(dates)
    y_arr = np.asarray(y, dtype=int)
    boundaries = _get_period_boundaries(dates, window_months)

    results: list[dict[str, Any]] = []

    for i in range(len(boundaries) - 1):
        train_start, train_end = boundaries[i]
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)

        train_mask = (dates >= train_start) & (dates < train_end)
        test_mask = (dates >= test_start) & (dates < test_end)

        X_train = X.loc[train_mask]
        y_train = y_arr[train_mask]
        X_test = X.loc[test_mask]
        y_test = y_arr[test_mask]

        if len(X_train) < 50 or len(X_test) < 50:
            continue
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        # Fit model
        if model_factory is not None:
            model = model_factory()
            model.fit(X_train, y_train)
        else:
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train.select_dtypes(include=[np.number]).fillna(0))
            X_test_sc = scaler.transform(X_test.select_dtypes(include=[np.number]).fillna(0))

            model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            model.fit(X_train_sc, y_train)
            y_prob = model.predict_proba(X_test_sc)[:, 1]

            results.append(
                {
                    "train_start": train_start.strftime("%Y-%m-%d"),
                    "train_end": train_end.strftime("%Y-%m-%d"),
                    "test_start": test_start.strftime("%Y-%m-%d"),
                    "test_end": test_end.strftime("%Y-%m-%d"),
                    "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
                    "pr_auc": round(average_precision_score(y_test, y_prob), 4),
                    "brier": round(brier_score_loss(y_test, y_prob), 4),
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                }
            )
            continue

        y_prob = model.predict_proba(X_test)[:, 1]
        results.append(
            {
                "train_start": train_start.strftime("%Y-%m-%d"),
                "train_end": train_end.strftime("%Y-%m-%d"),
                "test_start": test_start.strftime("%Y-%m-%d"),
                "test_end": test_end.strftime("%Y-%m-%d"),
                "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
                "pr_auc": round(average_precision_score(y_test, y_prob), 4),
                "brier": round(brier_score_loss(y_test, y_prob), 4),
                "n_train": len(X_train),
                "n_test": len(X_test),
            }
        )

    df = pd.DataFrame(results)
    if df.empty:
        return WindowExperimentResult(
            strategy="rolling",
            window_months=window_months,
            period_metrics=df,
            mean_roc_auc=0.0,
            mean_pr_auc=0.0,
            std_pr_auc=0.0,
            trend="stable",
        )

    mean_roc = float(df["roc_auc"].mean())
    mean_pr = float(df["pr_auc"].mean())
    std_pr = float(df["pr_auc"].std())

    # Trend detection
    if len(df) >= 3:
        pr_vals = df["pr_auc"].values
        diffs = np.diff(pr_vals)
        if np.mean(diffs) < -0.005:
            trend = "degrading"
        elif np.mean(diffs) > 0.005:
            trend = "improving"
        else:
            trend = "stable"
    else:
        trend = "stable"

    logger.info(
        "rolling_window_experiment_complete",
        n_periods=len(df),
        mean_pr_auc=round(mean_pr, 4),
        trend=trend,
    )

    return WindowExperimentResult(
        strategy="rolling",
        window_months=window_months,
        period_metrics=df,
        mean_roc_auc=round(mean_roc, 4),
        mean_pr_auc=round(mean_pr, 4),
        std_pr_auc=round(std_pr, 4),
        trend=trend,
    )


def expanding_window_experiment(
    X: pd.DataFrame,
    y: np.ndarray | pd.Series,
    dates: pd.Series,
    *,
    initial_months: int = 6,
    test_months: int = 3,
    model_factory: Any = None,
) -> WindowExperimentResult:
    """Simulate expanding-window retraining.

    Each successive training window includes all data from the start,
    expanding as new periods arrive.

    Parameters
    ----------
    X:
        Feature DataFrame.
    y:
        Binary target.
    dates:
        Date column.
    initial_months:
        Minimum initial training period.
    test_months:
        Test window after each training cutoff.
    model_factory:
        Optional model constructor.

    Returns
    -------
    WindowExperimentResult
        Per-period metrics.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    dates = pd.to_datetime(dates)
    y_arr = np.asarray(y, dtype=int)

    min_date = dates.min()
    max_date = dates.max()

    results: list[dict[str, Any]] = []
    cutoff = min_date + pd.DateOffset(months=initial_months)

    while cutoff < max_date:
        test_end = cutoff + pd.DateOffset(months=test_months)

        train_mask = dates < cutoff
        test_mask = (dates >= cutoff) & (dates < test_end)

        X_train = X.loc[train_mask]
        y_train = y_arr[train_mask]
        X_test = X.loc[test_mask]
        y_test = y_arr[test_mask]

        if len(X_train) < 50 or len(X_test) < 50:
            cutoff += pd.DateOffset(months=test_months)
            continue
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            cutoff += pd.DateOffset(months=test_months)
            continue

        if model_factory is not None:
            model = model_factory()
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train.select_dtypes(include=[np.number]).fillna(0))
            X_test_sc = scaler.transform(X_test.select_dtypes(include=[np.number]).fillna(0))

            model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            model.fit(X_train_sc, y_train)
            y_prob = model.predict_proba(X_test_sc)[:, 1]

        results.append(
            {
                "train_start": min_date.strftime("%Y-%m-%d"),
                "train_end": cutoff.strftime("%Y-%m-%d"),
                "test_start": cutoff.strftime("%Y-%m-%d"),
                "test_end": test_end.strftime("%Y-%m-%d"),
                "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
                "pr_auc": round(average_precision_score(y_test, y_prob), 4),
                "brier": round(brier_score_loss(y_test, y_prob), 4),
                "n_train": len(X_train),
                "n_test": len(X_test),
            }
        )

        cutoff += pd.DateOffset(months=test_months)

    df = pd.DataFrame(results)
    if df.empty:
        return WindowExperimentResult(
            strategy="expanding",
            window_months=initial_months,
            period_metrics=df,
            mean_roc_auc=0.0,
            mean_pr_auc=0.0,
            std_pr_auc=0.0,
            trend="stable",
        )

    mean_roc = float(df["roc_auc"].mean())
    mean_pr = float(df["pr_auc"].mean())
    std_pr = float(df["pr_auc"].std())

    if len(df) >= 3:
        diffs = np.diff(df["pr_auc"].values)
        trend = (
            "degrading"
            if np.mean(diffs) < -0.005
            else ("improving" if np.mean(diffs) > 0.005 else "stable")
        )
    else:
        trend = "stable"

    logger.info(
        "expanding_window_experiment_complete",
        n_periods=len(df),
        mean_pr_auc=round(mean_pr, 4),
        trend=trend,
    )

    return WindowExperimentResult(
        strategy="expanding",
        window_months=initial_months,
        period_metrics=df,
        mean_roc_auc=round(mean_roc, 4),
        mean_pr_auc=round(mean_pr, 4),
        std_pr_auc=round(std_pr, 4),
        trend=trend,
    )


# ---------------------------------------------------------------------------
# Model decay curve
# ---------------------------------------------------------------------------


@dataclass
class DecayCurve:
    """Performance degradation curve over time."""

    periods: list[str]
    pr_auc_values: list[float]
    roc_auc_values: list[float]
    brier_values: list[float]
    months_since_train: list[int]
    half_life_months: float | None  # months until PR-AUC drops by 50% of initial
    degradation_rate: float  # PR-AUC drop per month


def model_decay_curve(
    metrics_over_time: pd.DataFrame,
) -> DecayCurve:
    """Compute the performance decay curve from longitudinal metrics.

    Parameters
    ----------
    metrics_over_time:
        DataFrame with columns: period, pr_auc, roc_auc, brier, months_since_train.

    Returns
    -------
    DecayCurve
        Decay statistics and trend data.
    """
    required_cols = {"period", "pr_auc", "months_since_train"}
    if not required_cols.issubset(metrics_over_time.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    df = metrics_over_time.sort_values("months_since_train").reset_index(drop=True)

    if df.empty:
        return DecayCurve(
            periods=[],
            pr_auc_values=[],
            roc_auc_values=[],
            brier_values=[],
            months_since_train=[],
            half_life_months=None,
            degradation_rate=0.0,
        )

    pr_values = df["pr_auc"].values
    months = df["months_since_train"].values

    # Degradation rate: linear fit
    if len(pr_values) >= 2 and months[-1] > months[0]:
        degradation_rate = float((pr_values[-1] - pr_values[0]) / (months[-1] - months[0]))
    else:
        degradation_rate = 0.0

    # Half-life: when does PR-AUC drop to 50% of initial value?
    initial_pr = pr_values[0] if len(pr_values) > 0 else 0
    half_target = initial_pr * 0.5
    half_life: float | None = None

    if degradation_rate < 0 and initial_pr > 0:
        # Extrapolate: months_to_half = (initial - half_target) / |degradation_rate|
        half_life = round((initial_pr - half_target) / abs(degradation_rate), 1)

    logger.info(
        "decay_curve_computed",
        n_periods=len(df),
        degradation_rate=round(degradation_rate, 6),
        half_life=half_life,
    )

    return DecayCurve(
        periods=df["period"].tolist(),
        pr_auc_values=df["pr_auc"].tolist(),
        roc_auc_values=df.get("roc_auc", pd.Series(dtype=float)).tolist(),
        brier_values=df.get("brier", pd.Series(dtype=float)).tolist(),
        months_since_train=df["months_since_train"].tolist(),
        half_life_months=half_life,
        degradation_rate=round(degradation_rate, 6),
    )


# ---------------------------------------------------------------------------
# Retrain frequency recommendation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrainRecommendation:
    """Recommended retrain frequency and supporting evidence."""

    recommended_interval_months: int
    confidence: str  # "high", "medium", "low"
    rationale: str
    degradation_rate: float
    half_life_months: float | None


def recommend_retrain_frequency(
    decay_curve: DecayCurve,
    *,
    max_acceptable_pr_auc_drop: float = 0.03,
) -> RetrainRecommendation:
    """Recommend the optimal retrain interval based on the decay curve.

    Parameters
    ----------
    decay_curve:
        Output from ``model_decay_curve()``.
    max_acceptable_pr_auc_drop:
        Maximum PR-AUC drop before retrain is necessary.

    Returns
    -------
    RetrainRecommendation
        Interval and confidence level.
    """
    rate = decay_curve.degradation_rate

    if abs(rate) < 1e-6:
        return RetrainRecommendation(
            recommended_interval_months=12,
            confidence="low",
            rationale="No measurable degradation detected. Default to annual retrain.",
            degradation_rate=rate,
            half_life_months=decay_curve.half_life_months,
        )

    if rate >= 0:
        # Performance is stable or improving
        return RetrainRecommendation(
            recommended_interval_months=12,
            confidence="medium",
            rationale="Performance stable or improving. Annual retrain sufficient.",
            degradation_rate=rate,
            half_life_months=decay_curve.half_life_months,
        )

    # Performance is degrading: how many months until we exceed the drop limit?
    months_to_threshold = max_acceptable_pr_auc_drop / abs(rate)
    recommended = max(1, int(np.floor(months_to_threshold)))
    recommended = min(recommended, 12)  # Cap at annual

    if len(decay_curve.periods) >= 6:
        confidence = "high"
    elif len(decay_curve.periods) >= 3:
        confidence = "medium"
    else:
        confidence = "low"

    rationale = (
        f"PR-AUC degrades at ~{abs(rate):.4f}/month. "
        f"At threshold of {max_acceptable_pr_auc_drop}, "
        f"retrain every {recommended} month(s) is recommended."
    )

    if decay_curve.half_life_months is not None:
        rationale += f" Model half-life: ~{decay_curve.half_life_months:.0f} months."

    logger.info(
        "retrain_recommendation",
        interval_months=recommended,
        confidence=confidence,
        degradation_rate=round(rate, 4),
    )

    return RetrainRecommendation(
        recommended_interval_months=recommended,
        confidence=confidence,
        rationale=rationale,
        degradation_rate=rate,
        half_life_months=decay_curve.half_life_months,
    )


# ---------------------------------------------------------------------------
# Feature stability across retrains
# ---------------------------------------------------------------------------


@dataclass
class FeatureStability:
    """Feature selection stability across multiple retrains."""

    feature_frequency: pd.DataFrame  # feature, n_selected, pct_selected
    always_selected: list[str]
    never_selected: list[str]
    unstable_features: list[str]  # selected in 30-70% of retrains
    stability_index: float  # Jaccard-based average stability


def feature_stability_across_retrains(
    selected_features_per_retrain: list[list[str]],
    *,
    all_features: list[str] | None = None,
) -> FeatureStability:
    """Analyse which features are consistently selected across retrains.

    Parameters
    ----------
    selected_features_per_retrain:
        List of feature lists, one per retrain cycle.
    all_features:
        Complete feature universe. Inferred from data if not provided.

    Returns
    -------
    FeatureStability
        Per-feature selection rates and stability metrics.
    """
    n_retrains = len(selected_features_per_retrain)

    if n_retrains == 0:
        return FeatureStability(
            feature_frequency=pd.DataFrame(),
            always_selected=[],
            never_selected=[],
            unstable_features=[],
            stability_index=0.0,
        )

    # Build feature universe
    if all_features is None:
        all_features_set: set[str] = set()
        for fl in selected_features_per_retrain:
            all_features_set.update(fl)
        all_features = sorted(all_features_set)

    # Count how often each feature is selected
    freq: dict[str, int] = {f: 0 for f in all_features}
    for fl in selected_features_per_retrain:
        for f in fl:
            if f in freq:
                freq[f] += 1

    rows = [
        {"feature": f, "n_selected": c, "pct_selected": round(c / n_retrains * 100, 1)}
        for f, c in sorted(freq.items(), key=lambda x: -x[1])
    ]
    df = pd.DataFrame(rows)

    always = [f for f, c in freq.items() if c == n_retrains]
    never = [f for f, c in freq.items() if c == 0]
    unstable = [f for f, c in freq.items() if 0.3 * n_retrains <= c <= 0.7 * n_retrains]

    # Jaccard stability index: average pairwise Jaccard similarity
    if n_retrains >= 2:
        jaccard_scores: list[float] = []
        sets = [set(fl) for fl in selected_features_per_retrain]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                union = sets[i] | sets[j]
                intersection = sets[i] & sets[j]
                if union:
                    jaccard_scores.append(len(intersection) / len(union))
        stability_idx = float(np.mean(jaccard_scores)) if jaccard_scores else 0.0
    else:
        stability_idx = 1.0

    logger.info(
        "feature_stability_computed",
        n_retrains=n_retrains,
        n_always=len(always),
        n_unstable=len(unstable),
        stability_index=round(stability_idx, 4),
    )

    return FeatureStability(
        feature_frequency=df,
        always_selected=sorted(always),
        never_selected=sorted(never),
        unstable_features=sorted(unstable),
        stability_index=round(stability_idx, 4),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Demo: decay curve and recommendation
    print("=== Model Decay Curve ===")
    decay_data = pd.DataFrame(
        {
            "period": [f"M+{i}" for i in range(1, 13)],
            "pr_auc": [
                0.42 - 0.003 * i + np.random.default_rng(42 + i).normal(0, 0.002) for i in range(12)
            ],
            "roc_auc": [0.79 - 0.002 * i for i in range(12)],
            "brier": [0.065 + 0.001 * i for i in range(12)],
            "months_since_train": list(range(1, 13)),
        }
    )

    curve = model_decay_curve(decay_data)
    print(f"  Degradation rate: {curve.degradation_rate:.4f} PR-AUC/month")
    print(f"  Half-life: {curve.half_life_months} months")

    print("\n=== Retrain Recommendation ===")
    rec = recommend_retrain_frequency(curve, max_acceptable_pr_auc_drop=0.03)
    print(f"  Interval: {rec.recommended_interval_months} months")
    print(f"  Confidence: {rec.confidence}")
    print(f"  Rationale: {rec.rationale}")

    print("\n=== Feature Stability ===")
    retrains = [
        ["income", "debt_ratio", "tenure", "credit_util", "age"],
        ["income", "debt_ratio", "tenure", "enquiry_count", "age"],
        ["income", "debt_ratio", "num_accounts", "credit_util", "age"],
        ["income", "debt_ratio", "tenure", "credit_util", "employment_years"],
    ]
    stability = feature_stability_across_retrains(retrains)
    print(stability.feature_frequency.to_string(index=False))
    print(f"  Always selected: {stability.always_selected}")
    print(f"  Unstable: {stability.unstable_features}")
    print(f"  Jaccard stability: {stability.stability_index:.4f}")
