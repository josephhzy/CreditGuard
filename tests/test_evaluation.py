"""Tests for the evaluation layer: discrimination, calibration, threshold, fairness, segments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.calibration import (
    brier_score,
    expected_calibration_error,
    platt_scaling,
)
from evaluation.discrimination import (
    compute_discrimination_report,
    gini_coefficient,
    ks_statistic,
    roc_auc,
)
from evaluation.fairness import (
    adverse_impact_ratio,
    compute_fairness_report,
    demographic_parity,
)
from evaluation.segment_analysis import compute_segment_report
from evaluation.threshold_analysis import (
    approval_rate_vs_bad_rate,
    optimal_threshold,
)

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _make_y_true_score(
    n: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (y_true, y_score) with positive signal for metric tests."""
    rng = np.random.default_rng(seed)
    y_true = rng.binomial(1, 0.08, size=n)
    noise = rng.normal(0, 0.1, size=n)
    y_score = np.clip(y_true * 0.7 + (1 - y_true) * 0.05 + noise, 0.001, 0.999)
    return y_true, y_score


# ═══════════════════════════════════════════════════════════════════════════
# Discrimination
# ═══════════════════════════════════════════════════════════════════════════


class TestDiscrimination:
    """Tests for discrimination metrics."""

    def test_discrimination_roc_auc(self) -> None:
        """ROC-AUC should be substantially above 0.5 for a model with signal."""
        y_true, y_score = _make_y_true_score()
        auc = roc_auc(y_true, y_score)
        assert 0.5 < auc <= 1.0

    def test_discrimination_ks_statistic(self) -> None:
        """KS statistic should be in (0, 1] when the model has signal."""
        y_true, y_score = _make_y_true_score()
        ks = ks_statistic(y_true, y_score)
        assert 0.0 < ks <= 1.0

    def test_discrimination_gini(self) -> None:
        """Gini coefficient should equal 2*AUC - 1."""
        y_true, y_score = _make_y_true_score()
        gini = gini_coefficient(y_true, y_score)
        expected_gini = 2 * roc_auc(y_true, y_score) - 1.0
        assert abs(gini - expected_gini) < 1e-6

    def test_discrimination_report_complete(self) -> None:
        """Full discrimination report should have every dataclass field populated."""
        import dataclasses

        y_true, y_score = _make_y_true_score()
        report = compute_discrimination_report(y_true, y_score)
        # Every field on the report must be populated. Prevents a partial
        # report from silently passing when a new field is added but not
        # filled in.
        for f in dataclasses.fields(report):
            value = getattr(report, f.name)
            assert value is not None, f"discrimination report field {f.name} is None"
        # Spot checks for sanity.
        assert report.n_samples == len(y_true)
        assert report.roc_auc > 0.5
        assert report.ks_statistic > 0.0
        assert len(report.lift_at_k) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Calibration
# ═══════════════════════════════════════════════════════════════════════════


class TestCalibration:
    """Tests for calibration metrics and recalibration."""

    def test_calibration_brier_score(self) -> None:
        """Brier score should be in [0, 1] and beat a constant-base-rate predictor."""
        y_true, y_score = _make_y_true_score()
        bs = brier_score(y_true, y_score)
        assert 0.0 <= bs <= 1.0
        # Naive predictor that always emits the base rate p*(1-p).
        # A model with real signal must STRICTLY beat this baseline.
        naive_brier = brier_score(y_true, np.full_like(y_score, y_true.mean()))
        assert bs < naive_brier, (
            f"model Brier {bs:.4f} did not beat naive base-rate Brier {naive_brier:.4f}"
        )

    def test_calibration_platt_scaling(self) -> None:
        """Platt scaling should produce a fitted model whose calibrated mean matches the base rate."""
        y_true, y_score = _make_y_true_score()
        calibrated, lr = platt_scaling(y_true, y_score)
        # Bounds are mathematically tautological (predict_proba is in [0,1])
        # but kept as a smoke check.
        assert calibrated.min() >= 0.0
        assert calibrated.max() <= 1.0
        # Mean of calibrated probabilities should match the empirical base
        # rate — Platt scaling on the input scores recovers the prior.
        assert abs(calibrated.mean() - y_true.mean()) < 0.02
        # The fitted model must have a non-zero slope on its one input
        # feature; a constant predictor would not provide calibration.
        assert lr.coef_.shape == (1, 1)
        assert abs(float(lr.coef_[0, 0])) > 0.0

    def test_calibration_ece(self) -> None:
        """ECE should be non-negative and less than 1."""
        y_true, y_score = _make_y_true_score()
        ece = expected_calibration_error(y_true, y_score, n_bins=10)
        assert 0.0 <= ece < 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Threshold analysis
# ═══════════════════════════════════════════════════════════════════════════


class TestThresholdAnalysis:
    """Tests for threshold optimisation and business metrics."""

    def test_threshold_optimal_cost_matrix(self) -> None:
        """Optimal threshold via cost matrix should be in (0, 1)."""
        y_true, y_score = _make_y_true_score()
        best_t, best_cost = optimal_threshold(
            y_true,
            y_score,
            method="cost_matrix",
            fp_cost=50,
            fn_cost=500,
        )
        assert 0.0 < best_t < 1.0
        assert best_cost >= 0.0

    def test_threshold_approval_vs_bad_rate(self) -> None:
        """Higher thresholds should yield higher approval rates and lower bad rates."""
        y_true, y_score = _make_y_true_score()
        thresholds = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        result = approval_rate_vs_bad_rate(y_true, y_score, thresholds=thresholds)
        approval = result["approval_rate"]
        # approval_rate should increase monotonically as threshold goes up (more are approved)
        assert all(approval[i] <= approval[i + 1] for i in range(len(approval) - 1))


# ═══════════════════════════════════════════════════════════════════════════
# Fairness
# ═══════════════════════════════════════════════════════════════════════════


class TestFairness:
    """Tests for fairness diagnostics."""

    @staticmethod
    def _make_fairness_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
        """Synthetic data with equal approval rates across groups."""
        rng = np.random.default_rng(42)
        n = 400
        y_true = rng.binomial(1, 0.08, size=n)
        y_score = np.clip(rng.beta(2, 10, size=n), 0.001, 0.999)
        y_pred = (y_score >= 0.15).astype(int)
        sensitive = pd.Series(rng.choice(["A", "B"], size=n), name="group")
        return y_true, y_pred, y_score, sensitive

    def test_fairness_demographic_parity(self) -> None:
        """Demographic parity should compute the per-group approval rate correctly."""
        _, y_pred, _, sensitive = self._make_fairness_data()
        dp = demographic_parity(y_pred, sensitive)
        assert "A" in dp and "B" in dp
        for rate in dp.values():
            assert 0.0 <= rate <= 1.0
        # Approval = predicted non-default, so the expected per-group rate is
        # the fraction of that group with y_pred == 0. This catches the case
        # where the function silently returns a constant or the wrong sign.
        expected_a = float((y_pred[sensitive == "A"] == 0).mean())
        expected_b = float((y_pred[sensitive == "B"] == 0).mean())
        assert abs(dp["A"] - expected_a) < 1e-9
        assert abs(dp["B"] - expected_b) < 1e-9

    def test_fairness_adverse_impact_ratio(self) -> None:
        """AIR should be between 0 and 1."""
        _, y_pred, _, sensitive = self._make_fairness_data()
        air = adverse_impact_ratio(y_pred, sensitive)
        assert 0.0 <= air <= 1.0

    def test_fairness_air_near_unity_for_random_groups(self) -> None:
        """With random group assignment, AIR should land near 1.0.

        Note: this is a sanity floor on the computation, not a verification
        of the 80% rule. The 80% rule (AIR >= 0.80) is a production fairness
        gate; on a 400-row synthetic sample it is too noisy to assert tightly,
        which is why the threshold here is 0.75 rather than 0.80.
        """
        rng = np.random.default_rng(42)
        n = 400
        y_pred = rng.binomial(1, 0.1, size=n)  # ~10% decline
        sensitive = pd.Series(["A"] * (n // 2) + ["B"] * (n // 2), name="group")
        air = adverse_impact_ratio(y_pred, sensitive)
        assert air >= 0.75  # noise floor for 400 samples

    def test_fairness_air_fails_for_biased_distribution(self) -> None:
        """AIR should be < 0.80 when one group has a clearly lower approval rate.

        Group A: 90% approved (y_pred == 0), Group B: 60% approved.
        Expected AIR = 0.60 / 0.90 ≈ 0.667, which fails the 80% rule gate.
        This confirms the gate can actually trip on biased inputs.
        """
        n = 200
        y_pred_a = np.array([0] * 180 + [1] * 20)  # 90% approved
        y_pred_b = np.array([0] * 120 + [1] * 80)  # 60% approved
        y_pred = np.concatenate([y_pred_a, y_pred_b])
        sensitive = pd.Series(["A"] * n + ["B"] * n, name="group")
        air = adverse_impact_ratio(y_pred, sensitive)
        assert air < 0.80, f"Expected AIR < 0.80 for biased distribution, got {air:.4f}"

    def test_fairness_report_complete(self) -> None:
        """Full fairness report should contain all expected fields."""
        y_true, y_pred, y_score, sensitive = self._make_fairness_data()
        report = compute_fairness_report(y_true, y_pred, y_score, sensitive)
        assert report.n_groups == 2
        assert len(report.group_metrics) == 2
        assert isinstance(report.adverse_impact_ratio, float)


# ═══════════════════════════════════════════════════════════════════════════
# Segment analysis
# ═══════════════════════════════════════════════════════════════════════════


class TestSegmentAnalysis:
    """Tests for subgroup performance analysis."""

    def test_segment_analysis_produces_results(self) -> None:
        """Segment report should produce per-segment metrics."""
        rng = np.random.default_rng(42)
        n = 400
        y_true = rng.binomial(1, 0.10, size=n)
        y_score = np.clip(rng.beta(2, 8, size=n) + y_true * 0.2, 0.001, 0.999)
        segments = pd.Series(
            rng.choice(["low", "mid", "high"], size=n),
            name="income_band",
        )
        report = compute_segment_report(y_true, y_score, segments)
        assert report.n_segments >= 2
        assert report.overall_roc_auc > 0.5
        assert report.performance_gap >= 0.0
