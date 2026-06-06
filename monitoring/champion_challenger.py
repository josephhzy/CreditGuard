"""Champion-challenger model promotion governance.

Implements the decision framework for promoting a challenger model
to champion status in production. Ensures that any model promotion
meets discrimination, calibration, fairness, and interpretability
criteria before deployment.

This is a critical governance gate -- no model reaches production
without passing these checks.

Typical usage::

    from monitoring.champion_challenger import compare_models, promote_decision

    report = compare_models(champion_metrics, challenger_metrics)
    decision = promote_decision(report)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Metrics structures
# ---------------------------------------------------------------------------


@dataclass
class ModelMetrics:
    """Standard metrics for a credit model."""

    model_name: str
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    brier_score: float = 0.0
    ks_statistic: float = 0.0
    gini: float = 0.0
    calibration_gap: float = 0.0
    adverse_impact_ratio: float = 1.0  # 1.0 = perfect parity
    demographic_parity_diff: float = 0.0
    equalized_odds_diff: float = 0.0
    interpretability_tier: str = "high"  # "high", "medium", "low"
    n_features: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to dict."""
        return {
            "model_name": self.model_name,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "brier_score": self.brier_score,
            "ks_statistic": self.ks_statistic,
            "gini": self.gini,
            "calibration_gap": self.calibration_gap,
            "adverse_impact_ratio": self.adverse_impact_ratio,
            "demographic_parity_diff": self.demographic_parity_diff,
            "equalized_odds_diff": self.equalized_odds_diff,
            "interpretability_tier": self.interpretability_tier,
            "n_features": self.n_features,
        }


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------


@dataclass
class ComparisonReport:
    """Side-by-side champion vs. challenger comparison."""

    champion: ModelMetrics
    challenger: ModelMetrics
    comparison_table: pd.DataFrame
    pr_auc_improvement: float
    calibration_degradation: float
    fairness_within_tolerance: bool
    interpretability_acceptable: bool
    notes: list[str] = field(default_factory=list)


def compare_models(
    champion_metrics: ModelMetrics,
    challenger_metrics: ModelMetrics,
    *,
    fairness_tolerance: float = 0.80,
    calibration_tolerance: float = 0.02,
) -> ComparisonReport:
    """Compare champion and challenger models across all governance dimensions.

    Parameters
    ----------
    champion_metrics:
        Current production model metrics.
    challenger_metrics:
        Candidate model metrics.
    fairness_tolerance:
        Minimum adverse impact ratio (80% rule).
    calibration_tolerance:
        Maximum acceptable calibration degradation.

    Returns
    -------
    ComparisonReport
        Full comparison with governance checks.
    """
    champ = champion_metrics
    chall = challenger_metrics

    # Build comparison table
    metrics_list = [
        ("ROC-AUC", champ.roc_auc, chall.roc_auc),
        ("PR-AUC", champ.pr_auc, chall.pr_auc),
        ("Brier Score", champ.brier_score, chall.brier_score),
        ("KS Statistic", champ.ks_statistic, chall.ks_statistic),
        ("Gini", champ.gini, chall.gini),
        ("Calibration Gap", champ.calibration_gap, chall.calibration_gap),
        ("Adverse Impact Ratio", champ.adverse_impact_ratio, chall.adverse_impact_ratio),
        ("Demographic Parity Diff", champ.demographic_parity_diff, chall.demographic_parity_diff),
        ("Equalized Odds Diff", champ.equalized_odds_diff, chall.equalized_odds_diff),
        ("Interpretability", champ.interpretability_tier, chall.interpretability_tier),
        ("Feature Count", champ.n_features, chall.n_features),
    ]

    rows = []
    for metric_name, champ_val, chall_val in metrics_list:
        delta: float | str
        if isinstance(champ_val, (int, float)) and isinstance(chall_val, (int, float)):
            delta = round(chall_val - champ_val, 6)
        else:
            delta = "N/A"
        rows.append(
            {
                "metric": metric_name,
                "champion": champ_val,
                "challenger": chall_val,
                "delta": delta,
            }
        )

    table = pd.DataFrame(rows)

    # Key checks
    pr_auc_improvement = chall.pr_auc - champ.pr_auc
    cal_degradation = chall.brier_score - champ.brier_score
    fairness_ok = chall.adverse_impact_ratio >= fairness_tolerance

    # Interpretability check: low is only acceptable if current is also low
    interp_order = {"high": 3, "medium": 2, "low": 1}
    chall_interp = interp_order.get(chall.interpretability_tier, 0)
    champ_interp = interp_order.get(champ.interpretability_tier, 0)
    interp_ok = chall_interp >= champ_interp or chall_interp >= 2  # at least medium

    notes: list[str] = []
    if pr_auc_improvement > 0:
        notes.append(f"Challenger PR-AUC improved by {pr_auc_improvement:+.4f}")
    else:
        notes.append(f"Challenger PR-AUC did not improve ({pr_auc_improvement:+.4f})")

    if cal_degradation > calibration_tolerance:
        notes.append(
            f"Calibration degraded beyond tolerance: "
            f"delta={cal_degradation:+.4f} > {calibration_tolerance}"
        )

    if not fairness_ok:
        notes.append(
            f"Fairness violation: AIR={chall.adverse_impact_ratio:.3f} "
            f"< tolerance={fairness_tolerance}"
        )

    if not interp_ok:
        notes.append(
            f"Interpretability concern: challenger is '{chall.interpretability_tier}' "
            f"vs champion '{champ.interpretability_tier}'"
        )

    report = ComparisonReport(
        champion=champ,
        challenger=chall,
        comparison_table=table,
        pr_auc_improvement=round(pr_auc_improvement, 6),
        calibration_degradation=round(cal_degradation, 6),
        fairness_within_tolerance=fairness_ok,
        interpretability_acceptable=interp_ok,
        notes=notes,
    )

    logger.info(
        "model_comparison_complete",
        pr_auc_delta=round(pr_auc_improvement, 4),
        cal_delta=round(cal_degradation, 4),
        fairness_ok=fairness_ok,
        interp_ok=interp_ok,
    )
    return report


# ---------------------------------------------------------------------------
# Promotion decision
# ---------------------------------------------------------------------------


def promote_decision(
    comparison: ComparisonReport,
    *,
    min_pr_auc_improvement: float = 0.0,
    max_calibration_degradation: float = 0.02,
) -> str:
    """Decide whether to promote the challenger model.

    Promotion criteria:
    1. Challenger must beat champion on PR-AUC
    2. Must not degrade calibration beyond threshold
    3. Must stay within fairness tolerance band
    4. Must be interpretable enough for deployment tier

    Parameters
    ----------
    comparison:
        Output from ``compare_models()``.
    min_pr_auc_improvement:
        Minimum PR-AUC improvement required. Zero means any improvement suffices.
    max_calibration_degradation:
        Maximum acceptable Brier score increase.

    Returns
    -------
    str
        "promote", "keep_champion", or "investigate".
    """
    passes_performance = comparison.pr_auc_improvement > min_pr_auc_improvement
    passes_calibration = comparison.calibration_degradation <= max_calibration_degradation
    passes_fairness = comparison.fairness_within_tolerance
    passes_interpretability = comparison.interpretability_acceptable

    if passes_performance and passes_calibration and passes_fairness and passes_interpretability:
        decision = "promote"
    elif not passes_performance:
        decision = "keep_champion"
    else:
        # Failed on calibration, fairness, or interpretability -- needs investigation
        decision = "investigate"

    logger.info(
        "promotion_decision",
        decision=decision,
        perf=passes_performance,
        cal=passes_calibration,
        fair=passes_fairness,
        interp=passes_interpretability,
    )
    return decision


# ---------------------------------------------------------------------------
# Rollback check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RollbackResult:
    """Whether the current model should be rolled back to baseline."""

    should_rollback: bool
    reason: str
    current_pr_auc: float
    baseline_pr_auc: float
    degradation: float


def rollback_check(
    current_metrics: ModelMetrics,
    baseline_metrics: ModelMetrics,
    *,
    pr_auc_drop_threshold: float = 0.03,
    brier_increase_threshold: float = 0.05,
) -> RollbackResult:
    """Determine whether the current production model should be rolled back.

    Parameters
    ----------
    current_metrics:
        Metrics from the current monitoring period.
    baseline_metrics:
        Metrics from the model's initial validation (or last known good state).
    pr_auc_drop_threshold:
        PR-AUC drop that triggers rollback.
    brier_increase_threshold:
        Brier score increase that triggers rollback.

    Returns
    -------
    RollbackResult
        Whether to rollback and why.
    """
    pr_auc_drop = baseline_metrics.pr_auc - current_metrics.pr_auc
    brier_increase = current_metrics.brier_score - baseline_metrics.brier_score

    reasons: list[str] = []
    should_rollback = False

    if pr_auc_drop > pr_auc_drop_threshold:
        should_rollback = True
        reasons.append(
            f"PR-AUC dropped by {pr_auc_drop:.4f} "
            f"(baseline={baseline_metrics.pr_auc:.4f}, current={current_metrics.pr_auc:.4f})"
        )

    if brier_increase > brier_increase_threshold:
        should_rollback = True
        reasons.append(
            f"Brier score increased by {brier_increase:.4f} "
            f"(baseline={baseline_metrics.brier_score:.4f}, current={current_metrics.brier_score:.4f})"
        )

    if current_metrics.adverse_impact_ratio < 0.80:
        should_rollback = True
        reasons.append(f"Fairness violation: AIR={current_metrics.adverse_impact_ratio:.3f} < 0.80")

    reason_str = "; ".join(reasons) if reasons else "All metrics within acceptable bounds."

    logger.info(
        "rollback_check",
        should_rollback=should_rollback,
        pr_auc_drop=round(pr_auc_drop, 4),
        brier_increase=round(brier_increase, 4),
    )

    return RollbackResult(
        should_rollback=should_rollback,
        reason=reason_str,
        current_pr_auc=current_metrics.pr_auc,
        baseline_pr_auc=baseline_metrics.pr_auc,
        degradation=round(pr_auc_drop, 6),
    )


# ---------------------------------------------------------------------------
# CLI  --  NOTE: numbers below are ILLUSTRATIVE only.
# Real champion metrics: ROC-AUC 0.762, PR-AUC 0.247, n_features=15
# See artifacts/evaluation_results.json for authoritative values.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    champion = ModelMetrics(
        model_name="lgb_v3_champion",
        roc_auc=0.785,
        pr_auc=0.420,
        brier_score=0.065,
        ks_statistic=0.45,
        gini=0.57,
        calibration_gap=0.012,
        adverse_impact_ratio=0.88,
        demographic_parity_diff=0.04,
        equalized_odds_diff=0.03,
        interpretability_tier="medium",
        n_features=35,
    )

    challenger = ModelMetrics(
        model_name="lgb_v4_challenger",
        roc_auc=0.798,
        pr_auc=0.445,
        brier_score=0.062,
        ks_statistic=0.48,
        gini=0.60,
        calibration_gap=0.015,
        adverse_impact_ratio=0.85,
        demographic_parity_diff=0.05,
        equalized_odds_diff=0.04,
        interpretability_tier="medium",
        n_features=40,
    )

    print("=== Champion vs. Challenger ===")
    report = compare_models(champion, challenger)
    print(report.comparison_table.to_string(index=False))
    print("\nNotes:")
    for note in report.notes:
        print(f"  - {note}")

    print(f"\n=== Promotion Decision: {promote_decision(report).upper()} ===")

    print("\n=== Rollback Check ===")
    degraded = ModelMetrics(
        model_name="lgb_v3_degraded",
        roc_auc=0.745,
        pr_auc=0.375,
        brier_score=0.085,
        ks_statistic=0.38,
        gini=0.49,
        adverse_impact_ratio=0.82,
        interpretability_tier="medium",
    )
    rb = rollback_check(degraded, champion)
    print(f"  Should rollback: {rb.should_rollback}")
    print(f"  Reason: {rb.reason}")
