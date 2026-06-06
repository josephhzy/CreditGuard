"""CreditGuard monitoring suite.

Tracks model health in production via distribution drift (PSI, KS, JS),
calibration stability, alert management, champion-challenger governance,
and retraining strategy recommendations.
"""

from monitoring.alert_manager import (
    Alert,
    AlertSeverity,
    format_alert_summary,
    generate_alerts,
    retrain_trigger,
)
from monitoring.calibration_drift import (
    CalibrationDriftResult,
    RecalibrationRecommendation,
    calibration_over_time,
    detect_calibration_drift,
    recalibration_needed,
)
from monitoring.champion_challenger import (
    ComparisonReport,
    ModelMetrics,
    RollbackResult,
    compare_models,
    promote_decision,
    rollback_check,
)
from monitoring.drift_engine import (
    DriftReport,
    FeatureDrift,
    detect_drift,
    js_divergence,
    ks_test,
    subgroup_drift,
)
from monitoring.psi import (
    calculate_psi,
    classify_drift,
    psi_per_feature,
)
from monitoring.retrain_strategy import (
    DecayCurve,
    FeatureStability,
    RetrainRecommendation,
    WindowExperimentResult,
    expanding_window_experiment,
    feature_stability_across_retrains,
    model_decay_curve,
    recommend_retrain_frequency,
    rolling_window_experiment,
)

__all__ = [
    # psi
    "calculate_psi",
    "classify_drift",
    "psi_per_feature",
    # drift_engine
    "DriftReport",
    "FeatureDrift",
    "detect_drift",
    "js_divergence",
    "ks_test",
    "subgroup_drift",
    # calibration_drift
    "CalibrationDriftResult",
    "RecalibrationRecommendation",
    "calibration_over_time",
    "detect_calibration_drift",
    "recalibration_needed",
    # alert_manager
    "Alert",
    "AlertSeverity",
    "format_alert_summary",
    "generate_alerts",
    "retrain_trigger",
    # champion_challenger
    "ComparisonReport",
    "ModelMetrics",
    "RollbackResult",
    "compare_models",
    "promote_decision",
    "rollback_check",
    # retrain_strategy
    "DecayCurve",
    "FeatureStability",
    "RetrainRecommendation",
    "WindowExperimentResult",
    "expanding_window_experiment",
    "feature_stability_across_retrains",
    "model_decay_curve",
    "recommend_retrain_frequency",
    "rolling_window_experiment",
]
