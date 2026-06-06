"""WoE-based interpretable scorecard for credit decisioning.

Implements a traditional Weight-of-Evidence (WoE) scorecard using
logistic regression on WoE-transformed features. This serves as
the interpretable baseline model for regulatory governance, while
gradient boosting serves as the high-performing challenger.

The scorecard produces credit score points on a familiar scale
(e.g., 300-850) that can be understood by underwriters and
explained to applicants.

Typical usage::

    from decisioning.scorecard import build_scorecard, woe_binning

    woe_result = woe_binning(feature, target, n_bins=10)
    scorecard_model = build_scorecard(X, y, features)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import structlog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# WoE binning
# ---------------------------------------------------------------------------

_EPSILON = 1e-10  # avoid log(0); per-bin zeros yield extreme WoE, clipped to ±_WOE_CLIP
_WOE_CLIP = 10.0  # bins beyond ±10 are numerically unreliable


@dataclass
class WoEBin:
    """A single WoE bin."""

    bin_label: str
    lower: float
    upper: float
    n_total: int
    n_event: int
    n_non_event: int
    event_rate: float
    woe: float
    iv_contribution: float


@dataclass
class WoEResult:
    """Full WoE binning result for a single feature."""

    feature_name: str
    bins: list[WoEBin]
    information_value: float
    is_predictive: str  # "strong" (>0.3), "medium" (0.1-0.3), "weak" (0.02-0.1), "useless" (<0.02)


def woe_binning(
    feature: np.ndarray | pd.Series,
    target: np.ndarray | pd.Series,
    *,
    n_bins: int = 10,
    feature_name: str = "feature",
) -> WoEResult:
    """Calculate Weight-of-Evidence per bin and Information Value.

    Parameters
    ----------
    feature:
        Numeric feature values.
    target:
        Binary target (1 = event/default, 0 = non-event).
    n_bins:
        Number of quantile-based bins.
    feature_name:
        Name for logging and result labels.

    Returns
    -------
    WoEResult
        Per-bin WoE values and total IV for the feature.
    """
    feat = np.asarray(feature, dtype=float)
    tgt = np.asarray(target, dtype=int)

    if feat.shape[0] != tgt.shape[0]:
        raise ValueError("feature and target must have the same length.")

    # Remove NaN
    valid_mask = ~(np.isnan(feat) | np.isnan(tgt.astype(float)))
    feat = feat[valid_mask]
    tgt = tgt[valid_mask]

    if feat.size == 0:
        logger.warning("woe_binning_empty_input", feature=feature_name)
        return WoEResult(
            feature_name=feature_name, bins=[], information_value=0.0, is_predictive="useless"
        )

    # Quantile-based binning (handle duplicates by dropping)
    try:
        bin_labels, bin_edges = pd.qcut(
            feat, q=n_bins, retbins=True, duplicates="drop", labels=False
        )
    except ValueError:
        # If all values are the same
        logger.warning("woe_binning_constant_feature", feature=feature_name)
        return WoEResult(
            feature_name=feature_name, bins=[], information_value=0.0, is_predictive="useless"
        )

    total_events = tgt.sum()
    total_non_events = len(tgt) - total_events

    if total_events == 0 or total_non_events == 0:
        logger.warning("woe_binning_single_class", feature=feature_name)
        return WoEResult(
            feature_name=feature_name, bins=[], information_value=0.0, is_predictive="useless"
        )

    woe_bins: list[WoEBin] = []
    total_iv = 0.0

    unique_bins = np.sort(np.unique(bin_labels[~np.isnan(bin_labels)]))

    for b in unique_bins:
        b_int = int(b)
        mask = bin_labels == b

        n_total = int(mask.sum())
        n_event = int(tgt[mask].sum())
        n_non_event = n_total - n_event

        dist_event = (n_event + _EPSILON) / (total_events + _EPSILON)
        dist_non_event = (n_non_event + _EPSILON) / (total_non_events + _EPSILON)

        woe = float(np.log(dist_non_event / dist_event))

        # Clip extreme WoE from zero-count bins; values beyond ±_WOE_CLIP are
        # numerically unreliable and destabilise logistic regression inputs.
        if abs(woe) > _WOE_CLIP:
            logger.warning(
                "woe_bin_extreme_clipped",
                feature=feature_name,
                bin=b_int,
                n_event=n_event,
                n_non_event=n_non_event,
                raw_woe=round(woe, 4),
                clipped_to=_WOE_CLIP,
            )
            woe = float(np.clip(woe, -_WOE_CLIP, _WOE_CLIP))

        iv_contrib = (dist_non_event - dist_event) * woe

        lower = float(bin_edges[b_int]) if b_int < len(bin_edges) else float(bin_edges[-2])
        upper = float(bin_edges[b_int + 1]) if b_int + 1 < len(bin_edges) else float(bin_edges[-1])

        woe_bins.append(
            WoEBin(
                bin_label=f"[{lower:.4f}, {upper:.4f})",
                lower=lower,
                upper=upper,
                n_total=n_total,
                n_event=n_event,
                n_non_event=n_non_event,
                event_rate=round(n_event / n_total, 4) if n_total > 0 else 0.0,
                woe=round(woe, 6),
                iv_contribution=round(iv_contrib, 6),
            )
        )
        total_iv += iv_contrib

    # Classify predictive power
    if total_iv > 0.3:
        predictive = "strong"
    elif total_iv > 0.1:
        predictive = "medium"
    elif total_iv > 0.02:
        predictive = "weak"
    else:
        predictive = "useless"

    logger.info(
        "woe_binning_complete",
        feature=feature_name,
        iv=round(total_iv, 4),
        predictive_power=predictive,
        n_bins=len(woe_bins),
    )

    return WoEResult(
        feature_name=feature_name,
        bins=woe_bins,
        information_value=round(total_iv, 6),
        is_predictive=predictive,
    )


# ---------------------------------------------------------------------------
# WoE transformation
# ---------------------------------------------------------------------------


def _woe_transform_feature(
    feature: np.ndarray,
    woe_result: WoEResult,
) -> np.ndarray:
    """Transform a feature into its WoE values using precomputed bins."""
    feat = np.asarray(feature, dtype=float)
    transformed = np.zeros_like(feat)

    for woe_bin in woe_result.bins:
        mask = (feat >= woe_bin.lower) & (feat < woe_bin.upper)
        transformed[mask] = woe_bin.woe

    # Handle values at or beyond the last upper edge
    if woe_result.bins:
        last_bin = woe_result.bins[-1]
        mask_upper = feat >= last_bin.upper
        transformed[mask_upper] = last_bin.woe

    return transformed


# ---------------------------------------------------------------------------
# Scorecard builder
# ---------------------------------------------------------------------------


@dataclass
class ScorecardModel:
    """Trained WoE-based scorecard.

    Attributes
    ----------
    roc_auc, pr_auc, brier_score:
        **In-sample** (training-data) metrics — optimistically biased
        because WoE bins and the logistic model are fitted on the same
        data.  Use these only as a rough sanity check; obtain unbiased
        figures by evaluating on a held-out split.
    """

    feature_names: list[str]
    woe_results: dict[str, WoEResult]
    logistic_model: LogisticRegression
    intercept: float
    coefficients: dict[str, float]
    pdo: float
    base_score: float
    base_odds: float
    roc_auc: float
    pr_auc: float
    brier_score: float


def build_scorecard(
    X: pd.DataFrame,
    y: np.ndarray | pd.Series,
    features: list[str],
    *,
    n_bins: int = 10,
    pdo: float = 20.0,
    base_score: float = 600.0,
    base_odds: float = 50.0,
    C: float = 1.0,
) -> ScorecardModel:
    """Build a WoE-based logistic regression scorecard.

    Parameters
    ----------
    X:
        Feature DataFrame.
    y:
        Binary target (1 = default).
    features:
        List of feature columns to include in the scorecard.
    n_bins:
        Number of WoE bins per feature.
    pdo:
        Points to Double the Odds.
    base_score:
        Base score at base_odds.
    base_odds:
        Base odds ratio (non-default:default).
    C:
        Logistic regression regularisation strength.

    Returns
    -------
    ScorecardModel
        Trained scorecard with all components.

    Notes
    -----
    The ``roc_auc``, ``pr_auc``, and ``brier_score`` fields on the returned
    ``ScorecardModel`` are **in-sample** (training-data) metrics.  WoE bin
    boundaries are also derived from the same data, so these figures will be
    optimistically biased.  Pass a held-out split to
    ``compare_interpretable_vs_boosting`` for an unbiased comparison, or
    evaluate independently on a test set.
    """
    y_arr = np.asarray(y, dtype=int)

    # Step 1: WoE binning for each feature
    woe_results: dict[str, WoEResult] = {}
    woe_transformed = pd.DataFrame(index=X.index)

    for feat_name in features:
        if feat_name not in X.columns:
            logger.warning("scorecard_feature_missing", feature=feat_name)
            continue

        woe_res = woe_binning(
            X[feat_name].values,
            y_arr,
            n_bins=n_bins,
            feature_name=feat_name,
        )
        woe_results[feat_name] = woe_res

        if woe_res.bins:
            woe_transformed[feat_name] = _woe_transform_feature(X[feat_name].values, woe_res)

    if woe_transformed.empty:
        raise ValueError("No features produced valid WoE bins.")

    used_features = list(woe_transformed.columns)

    # Step 2: Logistic regression on WoE-transformed features
    lr = LogisticRegression(C=C, solver="lbfgs", max_iter=1000, random_state=42)
    lr.fit(woe_transformed[used_features].values, y_arr)

    # Step 3: Evaluate in-sample (training data only — metrics are optimistically biased)
    y_prob = lr.predict_proba(woe_transformed[used_features].values)[:, 1]
    roc = roc_auc_score(y_arr, y_prob)
    pr = average_precision_score(y_arr, y_prob)
    brier = brier_score_loss(y_arr, y_prob)

    coef_dict = {
        feat: round(float(c), 6) for feat, c in zip(used_features, lr.coef_[0], strict=False)
    }

    model = ScorecardModel(
        feature_names=used_features,
        woe_results=woe_results,
        logistic_model=lr,
        intercept=round(float(lr.intercept_[0]), 6),
        coefficients=coef_dict,
        pdo=pdo,
        base_score=base_score,
        base_odds=base_odds,
        roc_auc=round(roc, 4),
        pr_auc=round(pr, 4),
        brier_score=round(brier, 4),
    )

    logger.info(
        "scorecard_built",
        n_features=len(used_features),
        roc_auc=model.roc_auc,
        pr_auc=model.pr_auc,
        brier=model.brier_score,
    )
    return model


# ---------------------------------------------------------------------------
# Score-to-points mapping
# ---------------------------------------------------------------------------


def score_to_points(
    woe_score: float,
    *,
    pdo: float = 20.0,
    base_score: float = 600.0,
    base_odds: float = 50.0,
) -> float:
    """Convert a WoE log-odds score to credit score points.

    Uses the standard PDO (Points to Double Odds) scaling formula:
        Score = Offset - Factor * log(odds)
    where:
        Factor = PDO / ln(2)
        Offset = Base_Score - Factor * ln(Base_Odds)

    Parameters
    ----------
    woe_score:
        Log-odds score from the logistic regression.
    pdo:
        Points to Double the Odds.
    base_score:
        Score at the base odds.
    base_odds:
        The reference odds ratio.

    Returns
    -------
    float
        Credit score in points.
    """
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    points = offset - factor * woe_score
    return round(float(points), 1)


# ---------------------------------------------------------------------------
# Feature monotonicity check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonotonicityResult:
    """Result of a feature monotonicity test."""

    feature_name: str
    is_monotonic: bool
    direction: str  # "increasing", "decreasing", "non_monotonic"
    correlation: float
    n_inversions: int


def feature_monotonicity_check(
    feature: np.ndarray | pd.Series,
    target: np.ndarray | pd.Series,
    *,
    n_bins: int = 10,
    feature_name: str = "feature",
) -> MonotonicityResult:
    """Check whether the relationship between feature and target is monotonic.

    Bins the feature and checks whether the event rate is consistently
    increasing or decreasing across bins.

    Parameters
    ----------
    feature:
        Numeric feature values.
    target:
        Binary target values.
    n_bins:
        Number of bins for the monotonicity test.
    feature_name:
        Feature name for reporting.

    Returns
    -------
    MonotonicityResult
        Whether the feature-target relationship is monotonic.
    """
    feat = np.asarray(feature, dtype=float)
    tgt = np.asarray(target, dtype=int)

    valid = ~(np.isnan(feat) | np.isnan(tgt.astype(float)))
    feat, tgt = feat[valid], tgt[valid]

    if feat.size < n_bins:
        return MonotonicityResult(
            feature_name=feature_name,
            is_monotonic=False,
            direction="non_monotonic",
            correlation=0.0,
            n_inversions=0,
        )

    try:
        bin_labels = pd.qcut(feat, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        return MonotonicityResult(
            feature_name=feature_name,
            is_monotonic=False,
            direction="non_monotonic",
            correlation=0.0,
            n_inversions=0,
        )

    df = pd.DataFrame({"bin": bin_labels, "target": tgt})
    rates = df.groupby("bin")["target"].mean().values

    if len(rates) < 2:
        return MonotonicityResult(
            feature_name=feature_name,
            is_monotonic=False,
            direction="non_monotonic",
            correlation=0.0,
            n_inversions=0,
        )

    diffs = np.diff(rates)
    n_positive = (diffs > 0).sum()
    n_negative = (diffs < 0).sum()
    n_inversions = min(n_positive, n_negative)

    corr = float(np.corrcoef(np.arange(len(rates)), rates)[0, 1])

    if n_inversions == 0:
        is_mono = True
        direction = "increasing" if n_positive >= n_negative else "decreasing"
    else:
        is_mono = False
        direction = "non_monotonic"

    logger.info(
        "monotonicity_check",
        feature=feature_name,
        monotonic=is_mono,
        direction=direction,
        inversions=n_inversions,
    )

    return MonotonicityResult(
        feature_name=feature_name,
        is_monotonic=is_mono,
        direction=direction,
        correlation=round(corr, 4) if not np.isnan(corr) else 0.0,
        n_inversions=int(n_inversions),
    )


# ---------------------------------------------------------------------------
# Interpretable vs. boosting comparison
# ---------------------------------------------------------------------------


@dataclass
class ModelComparison:
    """Side-by-side comparison of scorecard vs. boosting model."""

    comparison_table: pd.DataFrame
    narrative: str
    scorecard_advantage: str
    boosting_advantage: str


def compare_interpretable_vs_boosting(
    scorecard_model: ScorecardModel,
    boosting_metrics: dict[str, float],
    X: pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> ModelComparison:
    """Compare the interpretable scorecard against a boosting model.

    Parameters
    ----------
    scorecard_model:
        Trained WoE scorecard.
    boosting_metrics:
        Dict with keys like "roc_auc", "pr_auc", "brier_score" for the booster.
    X:
        Feature DataFrame (used for scorecard prediction).
    y:
        True labels.

    Returns
    -------
    ModelComparison
        Comparison table and governance narrative.
    """
    y_arr = np.asarray(y, dtype=int)

    # Scorecard metrics (re-evaluate on given data)
    woe_df = pd.DataFrame(index=X.index)
    for feat_name in scorecard_model.feature_names:
        if feat_name in X.columns and feat_name in scorecard_model.woe_results:
            woe_df[feat_name] = _woe_transform_feature(
                X[feat_name].values,
                scorecard_model.woe_results[feat_name],
            )

    if not woe_df.empty:
        sc_proba = scorecard_model.logistic_model.predict_proba(woe_df.values)[:, 1]
        sc_roc = roc_auc_score(y_arr, sc_proba)
        sc_pr = average_precision_score(y_arr, sc_proba)
        sc_brier = brier_score_loss(y_arr, sc_proba)
    else:
        sc_roc = scorecard_model.roc_auc
        sc_pr = scorecard_model.pr_auc
        sc_brier = scorecard_model.brier_score

    rows = [
        {
            "metric": "ROC-AUC",
            "scorecard": round(sc_roc, 4),
            "boosting": round(boosting_metrics.get("roc_auc", 0), 4),
        },
        {
            "metric": "PR-AUC",
            "scorecard": round(sc_pr, 4),
            "boosting": round(boosting_metrics.get("pr_auc", 0), 4),
        },
        {
            "metric": "Brier Score",
            "scorecard": round(sc_brier, 4),
            "boosting": round(boosting_metrics.get("brier_score", 0), 4),
        },
        {
            "metric": "Interpretability",
            "scorecard": "Full (WoE + LR)",
            "boosting": "Partial (SHAP)",
        },
        {"metric": "Regulatory Suitability", "scorecard": "High", "boosting": "Medium"},
        {
            "metric": "Feature Count",
            "scorecard": len(scorecard_model.feature_names),
            "boosting": boosting_metrics.get("n_features", "N/A"),
        },
    ]

    table = pd.DataFrame(rows)

    narrative = (
        "The WoE scorecard serves as the interpretable baseline for governance and "
        "regulatory compliance. Every feature contribution is transparent and auditable. "
        "The gradient boosting model serves as the high-performing challenger, capturing "
        "non-linear interactions at the cost of reduced interpretability. "
        "Both models should be maintained: the scorecard for regulatory reporting and "
        "adverse action explanations, the booster for operational decisioning where "
        "permitted by policy."
    )

    return ModelComparison(
        comparison_table=table,
        narrative=narrative,
        scorecard_advantage="Full interpretability, regulatory compliance, adverse action reporting.",
        boosting_advantage="Higher discrimination, captures non-linear patterns and interactions.",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 5000

    # Synthetic features with some predictive signal
    X_demo = pd.DataFrame(
        {
            "income": rng.lognormal(10, 1, size=n),
            "debt_ratio": rng.beta(2, 5, size=n),
            "credit_history_months": rng.poisson(60, size=n).astype(float),
            "num_delinquencies": rng.poisson(0.5, size=n).astype(float),
        }
    )

    # Target with signal from features
    logit = (
        -3
        + 0.5 * (X_demo["debt_ratio"] - 0.3)
        + 0.3 * (X_demo["num_delinquencies"])
        - 0.001 * (X_demo["credit_history_months"] - 60)
    )
    prob = 1 / (1 + np.exp(-logit))
    y_demo = rng.binomial(1, prob)

    print("=== WoE Binning (debt_ratio) ===")
    woe_res = woe_binning(X_demo["debt_ratio"].values, y_demo, feature_name="debt_ratio")
    print(f"  IV: {woe_res.information_value} ({woe_res.is_predictive})")
    for b in woe_res.bins[:3]:
        print(f"  {b.bin_label}: WoE={b.woe}, event_rate={b.event_rate}")

    print("\n=== Monotonicity Check ===")
    mono = feature_monotonicity_check(
        X_demo["debt_ratio"].values, y_demo, feature_name="debt_ratio"
    )
    print(f"  Monotonic: {mono.is_monotonic}, Direction: {mono.direction}")

    print("\n=== Build Scorecard ===")
    features = ["income", "debt_ratio", "credit_history_months", "num_delinquencies"]
    sc = build_scorecard(X_demo, y_demo, features, n_bins=8)
    print(f"  ROC-AUC: {sc.roc_auc}, PR-AUC: {sc.pr_auc}, Brier: {sc.brier_score}")
    print(f"  Coefficients: {sc.coefficients}")

    print("\n=== Score-to-Points ===")
    sample_woe = 1.5
    pts = score_to_points(sample_woe)
    print(f"  WoE score {sample_woe} -> {pts} points")
