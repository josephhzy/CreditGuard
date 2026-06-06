"""CreditGuard decisioning layer.

Maps model risk scores to business decisions, simulates threshold impacts,
provides portfolio-level analytics, and builds interpretable scorecards.
"""

from decisioning.business_simulation import (
    OptimalPoint,
    SimulationResult,
    find_optimal_operating_point,
    simulate,
    sweep_thresholds,
)
from decisioning.decision_engine import (
    DecisionBand,
    DecisionExplanation,
    Thresholds,
    batch_decide,
    explain_decision,
    score_to_decision,
)
from decisioning.portfolio_analytics import (
    ConcentrationResult,
    DeteriorationResult,
    ExposureResult,
    PortfolioDrift,
    RiskDistribution,
    concentration_risk,
    exposure_at_threshold,
    portfolio_drift,
    risk_distribution,
    top_deteriorating_segments,
)
from decisioning.scorecard import (
    MonotonicityResult,
    ScorecardModel,
    WoEResult,
    build_scorecard,
    compare_interpretable_vs_boosting,
    feature_monotonicity_check,
    score_to_points,
    woe_binning,
)

__all__ = [
    # decision_engine
    "DecisionBand",
    "DecisionExplanation",
    "Thresholds",
    "batch_decide",
    "explain_decision",
    "score_to_decision",
    # business_simulation
    "OptimalPoint",
    "SimulationResult",
    "find_optimal_operating_point",
    "simulate",
    "sweep_thresholds",
    # portfolio_analytics
    "ConcentrationResult",
    "DeteriorationResult",
    "ExposureResult",
    "PortfolioDrift",
    "RiskDistribution",
    "concentration_risk",
    "exposure_at_threshold",
    "portfolio_drift",
    "risk_distribution",
    "top_deteriorating_segments",
    # scorecard
    "MonotonicityResult",
    "ScorecardModel",
    "WoEResult",
    "build_scorecard",
    "compare_interpretable_vs_boosting",
    "feature_monotonicity_check",
    "score_to_points",
    "woe_binning",
]
