"""Tests for the feature engineering layer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.application_profile import build_application_features
from features.build_all import build_features
from features.bureau_history import build_bureau_features
from features.payment_behaviour import build_payment_features
from features.previous_apps import build_previous_app_features
from features.quality_checks import run_quality_checks
from features.risk_ratios import build_risk_ratio_features
from features.temporal_trends import build_temporal_features
from features.utilization import build_utilization_features

# ═══════════════════════════════════════════════════════════════════════════
# Individual feature families
# ═══════════════════════════════════════════════════════════════════════════


class TestApplicationProfile:
    """Tests for application_profile features."""

    def test_application_profile_creates_features(
        self,
        sample_application_df: pd.DataFrame,
    ) -> None:
        """Builder should return a DataFrame of feat_* columns."""
        result = build_application_features(sample_application_df)
        assert len(result) == len(sample_application_df)
        feat_cols = [c for c in result.columns if c.startswith("feat_")]
        assert len(feat_cols) >= 5
        assert "feat_income" in result.columns
        assert "feat_age_years" in result.columns


class TestBureauHistory:
    """Tests for bureau_history features."""

    def test_bureau_history_creates_features(
        self,
        sample_application_df: pd.DataFrame,
    ) -> None:
        """Bureau builder should produce features even when source columns are absent."""
        result = build_bureau_features(sample_application_df)
        assert len(result) == len(sample_application_df)
        assert "feat_bureau_total_count" in result.columns
        assert "feat_bureau_active_ratio" in result.columns


class TestRiskRatios:
    """Tests for risk_ratios features."""

    def test_risk_ratios_creates_features(
        self,
        sample_application_df: pd.DataFrame,
    ) -> None:
        """Risk ratios should create debt burden and external source features."""
        result = build_risk_ratio_features(sample_application_df)
        assert len(result) == len(sample_application_df)
        assert "feat_debt_burden" in result.columns
        assert "feat_annuity_to_income" in result.columns
        # External source features should exist since the fixture has EXT_SOURCE_*
        assert "feat_ext_source_1" in result.columns


class TestPaymentBehaviour:
    """Tests for payment_behaviour features."""

    def test_payment_behaviour_creates_features(
        self,
        sample_application_df: pd.DataFrame,
    ) -> None:
        """Builder should produce features even when source columns are absent."""
        result = build_payment_features(sample_application_df)
        assert len(result) == len(sample_application_df)
        feat_cols = [c for c in result.columns if c.startswith("feat_")]
        assert len(feat_cols) >= 1
        assert "feat_pay_lateness_mean" in result.columns


class TestUtilization:
    """Tests for utilization features."""

    def test_utilization_creates_features(
        self,
        sample_application_df: pd.DataFrame,
    ) -> None:
        """Builder should produce features even when source columns are absent."""
        result = build_utilization_features(sample_application_df)
        assert len(result) == len(sample_application_df)
        feat_cols = [c for c in result.columns if c.startswith("feat_")]
        assert len(feat_cols) >= 1
        assert "feat_cc_utilization_mean" in result.columns


class TestTemporalTrends:
    """Tests for temporal_trends features."""

    def test_temporal_trends_creates_features(
        self,
        sample_application_df: pd.DataFrame,
    ) -> None:
        """Builder should produce features even when source columns are absent."""
        result = build_temporal_features(sample_application_df)
        assert len(result) == len(sample_application_df)
        feat_cols = [c for c in result.columns if c.startswith("feat_")]
        assert len(feat_cols) >= 1
        assert "feat_trend_worsening_score" in result.columns


class TestPreviousApps:
    """Tests for previous_apps features."""

    def test_previous_apps_creates_features(
        self,
        sample_application_df: pd.DataFrame,
    ) -> None:
        """Builder should produce features even when source columns are absent."""
        result = build_previous_app_features(sample_application_df)
        assert len(result) == len(sample_application_df)
        feat_cols = [c for c in result.columns if c.startswith("feat_")]
        assert len(feat_cols) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Quality checks
# ═══════════════════════════════════════════════════════════════════════════


class TestQualityChecks:
    """Tests for the feature quality checker."""

    def test_quality_checks_detects_constant_feature(self) -> None:
        """A constant column should appear in the constant_features list."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "good": rng.standard_normal(100),
                "constant": np.ones(100),
            }
        )
        report = run_quality_checks(df, variance_threshold=0.001)
        assert "constant" in report.constant_features

    def test_quality_checks_detects_high_correlation(self) -> None:
        """Perfectly correlated pair should be flagged."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        df = pd.DataFrame({"a": x, "b": x * 1.0 + 0.0001 * rng.standard_normal(200)})
        report = run_quality_checks(df, correlation_threshold=0.95)
        assert len(report.high_corr_pairs) >= 1
        pair_names = {(a, b) for a, b, _ in report.high_corr_pairs}
        assert ("a", "b") in pair_names or ("b", "a") in pair_names


# ═══════════════════════════════════════════════════════════════════════════
# build_all orchestrator
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildAll:
    """Tests for the feature orchestrator."""

    def test_build_all_runs_without_error(
        self,
        sample_application_df: pd.DataFrame,
    ) -> None:
        """build_features should succeed on a well-formed application DataFrame."""
        result = build_features(sample_application_df, run_checks=False)
        assert len(result) == len(sample_application_df)
        assert "SK_ID_CURR" in result.columns

    def test_features_have_no_infinities(
        self,
        sample_application_df: pd.DataFrame,
    ) -> None:
        """Engineered features should contain no infinite values."""
        result = build_features(sample_application_df, run_checks=False)
        numeric = result.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any(), "Infinite values found in features"
