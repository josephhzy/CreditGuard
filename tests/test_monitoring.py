"""Tests for the monitoring layer: PSI, drift engine, calibration drift, alerts, champion-challenger."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from monitoring.alert_manager import (
    Alert,
    AlertSeverity,
    generate_alerts,
    retrain_trigger,
)
from monitoring.calibration_drift import detect_calibration_drift
from monitoring.champion_challenger import (
    ModelMetrics,
    compare_models,
    promote_decision,
)
from monitoring.drift_engine import detect_drift
from monitoring.psi import calculate_psi

# ═══════════════════════════════════════════════════════════════════════════
# PSI
# ═══════════════════════════════════════════════════════════════════════════


class TestPSI:
    """Tests for Population Stability Index calculation."""

    def test_psi_identical_distributions(self) -> None:
        """PSI of identical distributions should be very close to zero."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(1000)
        psi = calculate_psi(data, data.copy(), n_bins=10)
        assert psi < 0.01

    def test_psi_shifted_distribution(self) -> None:
        """PSI should be substantial when the distribution is shifted."""
        rng = np.random.default_rng(42)
        expected = rng.normal(0.0, 1.0, size=2000)
        actual = rng.normal(1.0, 1.0, size=2000)  # mean shift of 1.0
        psi = calculate_psi(expected, actual, n_bins=10)
        assert psi > 0.10  # should clearly detect the shift

    @patch("monitoring.psi._load_psi_thresholds")
    def test_psi_classify_drift_green(self, mock_thresh: ...) -> None:
        """PSI below threshold should classify as green."""
        mock_thresh.return_value = {"threshold_green": 0.10, "threshold_amber": 0.20}
        from monitoring.psi import classify_drift

        assert classify_drift(0.05) == "green"

    @patch("monitoring.psi._load_psi_thresholds")
    def test_psi_classify_drift_amber(self, mock_thresh: ...) -> None:
        """PSI between green and amber thresholds should classify as amber."""
        mock_thresh.return_value = {"threshold_green": 0.10, "threshold_amber": 0.20}
        from monitoring.psi import classify_drift

        assert classify_drift(0.15) == "amber"

    @patch("monitoring.psi._load_psi_thresholds")
    def test_psi_classify_drift_red(self, mock_thresh: ...) -> None:
        """PSI above amber threshold should classify as red."""
        mock_thresh.return_value = {"threshold_green": 0.10, "threshold_amber": 0.20}
        from monitoring.psi import classify_drift

        assert classify_drift(0.25) == "red"


# ═══════════════════════════════════════════════════════════════════════════
# Drift engine
# ═══════════════════════════════════════════════════════════════════════════


class TestDriftEngine:
    """Tests for the multi-method drift detection engine."""

    @patch("monitoring.psi._load_psi_thresholds")
    def test_drift_engine_detects_drift(self, mock_thresh: ...) -> None:
        """Drift engine should flag a significantly shifted feature as red/amber."""
        mock_thresh.return_value = {"threshold_green": 0.10, "threshold_amber": 0.20}
        rng = np.random.default_rng(42)
        n = 1000
        X_ref = rng.standard_normal((n, 3))
        X_cur = X_ref.copy()
        X_cur[:, 1] += 2.0  # inject heavy drift in feature 1

        report = detect_drift(
            X_ref,
            X_cur,
            feature_names=["a", "b", "c"],
            methods=["psi"],
            n_bins=10,
        )
        assert report.overall_severity in ("amber", "red")
        assert "b" in report.top_drifting_features

    @patch("monitoring.psi._load_psi_thresholds")
    def test_drift_engine_no_drift(self, mock_thresh: ...) -> None:
        """Identical data should produce a green overall severity."""
        mock_thresh.return_value = {"threshold_green": 0.10, "threshold_amber": 0.20}
        rng = np.random.default_rng(42)
        data = rng.standard_normal((500, 3))
        report = detect_drift(
            data,
            data.copy(),
            feature_names=["x", "y", "z"],
            methods=["psi"],
            n_bins=10,
        )
        assert report.overall_severity == "green"


# ═══════════════════════════════════════════════════════════════════════════
# Calibration drift
# ═══════════════════════════════════════════════════════════════════════════


class TestCalibrationDrift:
    """Tests for calibration drift detection."""

    def test_calibration_drift_detection(self) -> None:
        """A large Brier score increase should trigger a red alert."""
        result = detect_calibration_drift(
            reference_brier=0.05,
            current_brier=0.12,
            amber_threshold=0.02,
            red_threshold=0.05,
        )
        assert result.has_drifted is True
        assert result.severity == "red"
        assert result.brier_delta > 0

    def test_calibration_drift_stable(self) -> None:
        """A tiny Brier change should be classified as green."""
        result = detect_calibration_drift(
            reference_brier=0.06,
            current_brier=0.065,
            amber_threshold=0.02,
            red_threshold=0.05,
        )
        assert result.has_drifted is False
        assert result.severity == "green"


# ═══════════════════════════════════════════════════════════════════════════
# Alert manager
# ═══════════════════════════════════════════════════════════════════════════


class TestAlertManager:
    """Tests for alert generation and retrain triggers."""

    def test_alert_manager_generates_alerts(self) -> None:
        """PSI values exceeding threshold should produce alerts."""
        psi_values = {"income": 0.25, "age": 0.05, "debt": 0.15}
        alerts = generate_alerts(psi_values=psi_values)
        assert len(alerts) >= 2  # income (red) + debt (amber)
        severities = {a.severity for a in alerts}
        assert AlertSeverity.RED in severities

    def test_alert_manager_no_alerts_when_clean(self) -> None:
        """All-green PSI values should generate no alerts."""
        psi_values = {"income": 0.02, "age": 0.01}
        alerts = generate_alerts(psi_values=psi_values)
        assert len(alerts) == 0

    def test_retrain_trigger_on_red(self) -> None:
        """A single RED alert should trigger retrain (default thresholds)."""
        alerts = [
            Alert(
                feature="income",
                metric="psi",
                value=0.25,
                severity=AlertSeverity.RED,
                message="test",
            ),
        ]
        assert retrain_trigger(alerts) is True

    def test_retrain_trigger_no_retrain_on_green(self) -> None:
        """No alerts should not trigger retrain."""
        assert retrain_trigger([]) is False


# ═══════════════════════════════════════════════════════════════════════════
# Champion-challenger
# ═══════════════════════════════════════════════════════════════════════════


class TestChampionChallenger:
    """Tests for model promotion governance."""

    @staticmethod
    def _champion() -> ModelMetrics:
        return ModelMetrics(
            model_name="champion_v1",
            roc_auc=0.78,
            pr_auc=0.42,
            brier_score=0.065,
            ks_statistic=0.45,
            gini=0.56,
            calibration_gap=0.01,
            adverse_impact_ratio=0.88,
            interpretability_tier="medium",
            n_features=35,
        )

    @staticmethod
    def _better_challenger() -> ModelMetrics:
        return ModelMetrics(
            model_name="challenger_v2",
            roc_auc=0.80,
            pr_auc=0.46,
            brier_score=0.060,
            ks_statistic=0.50,
            gini=0.60,
            calibration_gap=0.012,
            adverse_impact_ratio=0.85,
            interpretability_tier="medium",
            n_features=40,
        )

    @staticmethod
    def _worse_challenger() -> ModelMetrics:
        return ModelMetrics(
            model_name="challenger_bad",
            roc_auc=0.75,
            pr_auc=0.38,
            brier_score=0.080,
            ks_statistic=0.40,
            gini=0.50,
            calibration_gap=0.03,
            adverse_impact_ratio=0.82,
            interpretability_tier="medium",
            n_features=30,
        )

    def test_champion_challenger_promote(self) -> None:
        """A better challenger should receive a 'promote' decision."""
        report = compare_models(self._champion(), self._better_challenger())
        decision = promote_decision(report)
        assert decision == "promote"

    def test_champion_challenger_keep(self) -> None:
        """A worse challenger should receive a 'keep_champion' decision."""
        report = compare_models(self._champion(), self._worse_challenger())
        decision = promote_decision(report)
        assert decision == "keep_champion"
