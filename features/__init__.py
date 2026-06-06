"""CreditGuard feature engineering layer.

Provides modular feature families that transform the joined Home Credit
dataset into a model-ready feature matrix. Each module exports a builder
function: ``build_*_features(df) -> pd.DataFrame``.

Usage::

    from features.build_all import build_features, load_joined_dataset

    df = load_joined_dataset()
    features = build_features(df)
"""

from features.application_profile import build_application_features
from features.build_all import build_features, load_joined_dataset
from features.bureau_history import build_bureau_features
from features.payment_behaviour import build_payment_features
from features.previous_apps import build_previous_app_features
from features.quality_checks import FeatureQualityReport, run_quality_checks
from features.risk_ratios import build_risk_ratio_features
from features.temporal_trends import build_temporal_features
from features.utilization import build_utilization_features

__all__ = [
    "build_application_features",
    "build_bureau_features",
    "build_features",
    "build_payment_features",
    "build_previous_app_features",
    "build_risk_ratio_features",
    "build_temporal_features",
    "build_utilization_features",
    "load_joined_dataset",
    "run_quality_checks",
    "FeatureQualityReport",
]
