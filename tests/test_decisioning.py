"""Tests for the decisioning layer: decision engine, simulation, scorecard, portfolio."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from decisioning.decision_engine import (
    DecisionBand,
    Thresholds,
    batch_decide,
    explain_decision,
    score_to_decision,
)
from decisioning.portfolio_analytics import risk_distribution
from decisioning.scorecard import score_to_points, woe_binning

# ═══════════════════════════════════════════════════════════════════════════
# Decision engine
# ═══════════════════════════════════════════════════════════════════════════


class TestDecisionEngine:
    """Tests for the core decisioning logic."""

    @pytest.fixture()
    def thresholds(self) -> Thresholds:
        return Thresholds(approve_upper=0.15, decline_lower=0.40)

    def test_score_to_decision_approve(self, thresholds: Thresholds) -> None:
        """Score below approve_upper should map to APPROVE."""
        assert score_to_decision(0.05, thresholds) == DecisionBand.APPROVE
        assert score_to_decision(0.14, thresholds) == DecisionBand.APPROVE

    def test_score_to_decision_review(self, thresholds: Thresholds) -> None:
        """Score in [approve_upper, decline_lower) should map to REVIEW."""
        assert score_to_decision(0.15, thresholds) == DecisionBand.REVIEW
        assert score_to_decision(0.30, thresholds) == DecisionBand.REVIEW
        assert score_to_decision(0.39, thresholds) == DecisionBand.REVIEW

    def test_score_to_decision_decline(self, thresholds: Thresholds) -> None:
        """Score at or above decline_lower should map to DECLINE."""
        assert score_to_decision(0.40, thresholds) == DecisionBand.DECLINE
        assert score_to_decision(0.90, thresholds) == DecisionBand.DECLINE

    def test_batch_decide(self, thresholds: Thresholds) -> None:
        """batch_decide should return correct decisions for an array of scores."""
        scores = np.array([0.05, 0.20, 0.50, 0.14, 0.40])
        decisions = batch_decide(scores, thresholds)
        assert len(decisions) == 5
        assert decisions[0] == DecisionBand.APPROVE
        assert decisions[1] == DecisionBand.REVIEW
        assert decisions[2] == DecisionBand.DECLINE
        assert decisions[3] == DecisionBand.APPROVE
        assert decisions[4] == DecisionBand.DECLINE

    def test_batch_decide_empty(self, thresholds: Thresholds) -> None:
        """batch_decide with empty input should return empty list."""
        decisions = batch_decide(np.array([]), thresholds)
        assert decisions == []


# ═══════════════════════════════════════════════════════════════════════════
# explain_decision
# ═══════════════════════════════════════════════════════════════════════════


class TestExplainDecision:
    """Tests for the SHAP-based decision explanation function."""

    @pytest.fixture()
    def thresholds(self) -> Thresholds:
        return Thresholds(approve_upper=0.15, decline_lower=0.40)

    @pytest.fixture()
    def feature_names(self) -> list[str]:
        return [f"feat_{i}" for i in range(6)]

    def test_top_factors_sorted_by_abs_shap(self, thresholds, feature_names) -> None:
        # feat_5 has the largest |SHAP| (0.9), should appear first
        shap_vals = np.array([0.1, -0.2, 0.05, -0.3, 0.15, 0.9])
        result = explain_decision(0.05, shap_vals, feature_names, thresholds=thresholds)
        increases = result["top_factors"]["increases_risk"]
        decreases = result["top_factors"]["decreases_risk"]
        assert increases[0]["feature"] == "feat_5"
        assert all(e["shap_value"] > 0 for e in increases)
        assert all(e["shap_value"] <= 0 for e in decreases)
        # verify descending |shap| order within each list
        inc_abs = [abs(e["shap_value"]) for e in increases]
        dec_abs = [abs(e["shap_value"]) for e in decreases]
        assert inc_abs == sorted(inc_abs, reverse=True)
        assert dec_abs == sorted(dec_abs, reverse=True)

    def test_confidence_note_near_boundary(self, thresholds, feature_names) -> None:
        # score 0.16 is 0.01 from approve_upper=0.15 -> triggers near-boundary note
        shap_vals = np.zeros(6)
        result = explain_decision(0.16, shap_vals, feature_names, thresholds=thresholds)
        assert "boundary" in result["confidence_note"].lower()
        assert result["confidence_note"] != ""

    def test_shap_length_mismatch_raises(self, thresholds, feature_names) -> None:
        shap_vals = np.array([0.1, 0.2])  # length 2, feature_names has 6
        with pytest.raises(ValueError, match="does not match"):
            explain_decision(0.5, shap_vals, feature_names, thresholds=thresholds)


# ═══════════════════════════════════════════════════════════════════════════
# Business simulation
# ═══════════════════════════════════════════════════════════════════════════


class TestBusinessSimulation:
    """Tests for the business impact simulator."""

    @patch("decisioning.business_simulation._load_cost_matrix")
    def test_business_simulation_results(self, mock_cost: ...) -> None:
        """simulate should return valid metrics for synthetic data."""
        mock_cost.return_value = {"fp_cost": 50, "fn_cost": 500}
        from decisioning.business_simulation import simulate

        rng = np.random.default_rng(42)
        n = 500
        y_true = rng.binomial(1, 0.08, size=n)
        y_score = np.clip(rng.beta(2, 10, size=n), 0.001, 0.999)

        result = simulate(y_true, y_score, approve_upper=0.15, decline_lower=0.40)
        assert result.n_total == n
        assert result.n_approved + result.n_declined + result.n_review == n
        assert 0.0 <= result.approval_rate <= 1.0
        assert result.total_expected_cost >= 0.0

    @patch("decisioning.business_simulation._load_cost_matrix")
    def test_sweep_thresholds(self, mock_cost: ...) -> None:
        """Threshold sweep should return a DataFrame with multiple rows."""
        mock_cost.return_value = {"fp_cost": 50, "fn_cost": 500}
        from decisioning.business_simulation import sweep_thresholds

        rng = np.random.default_rng(42)
        n = 200
        y_true = rng.binomial(1, 0.10, size=n)
        y_score = np.clip(rng.beta(2, 8, size=n), 0.001, 0.999)

        sweep_df = sweep_thresholds(y_true, y_score, n_points=5)
        assert isinstance(sweep_df, pd.DataFrame)
        assert len(sweep_df) > 0
        assert "approval_rate" in sweep_df.columns


# ═══════════════════════════════════════════════════════════════════════════
# Scorecard
# ═══════════════════════════════════════════════════════════════════════════


class TestScorecard:
    """Tests for WoE binning and scorecard conversion."""

    def test_woe_binning_produces_iv(self) -> None:
        """WoE binning should compute a positive Information Value for a predictive feature."""
        rng = np.random.default_rng(42)
        n = 1000
        target = rng.binomial(1, 0.10, size=n)
        # Feature with signal: higher values = higher default risk
        feature = rng.standard_normal(n) + target * 1.0
        result = woe_binning(feature, target, n_bins=5, feature_name="test_feature")
        assert result.information_value > 0.0
        assert len(result.bins) > 0

    def test_scorecard_produces_scores(self) -> None:
        """score_to_points should return a finite numeric value."""
        pts = score_to_points(1.0, pdo=20.0, base_score=600.0, base_odds=50.0)
        assert isinstance(pts, float)
        assert np.isfinite(pts)

    def test_scorecard_monotonic_with_woe(self) -> None:
        """Higher WoE (riskier) should yield lower score points."""
        pts_low = score_to_points(0.5)
        pts_high = score_to_points(2.0)
        assert pts_high < pts_low


# ═══════════════════════════════════════════════════════════════════════════
# Portfolio analytics
# ═══════════════════════════════════════════════════════════════════════════


class TestPortfolioAnalytics:
    """Tests for portfolio risk distribution."""

    def test_portfolio_risk_distribution(self) -> None:
        """risk_distribution should compute valid summary statistics."""
        rng = np.random.default_rng(42)
        scores = rng.beta(2, 10, size=1000)
        dist = risk_distribution(scores)
        assert 0.0 <= dist.mean_score <= 1.0
        assert 0.0 <= dist.median_score <= 1.0
        assert dist.std_score >= 0.0
        assert len(dist.bin_counts) > 0
        assert sum(dist.bin_counts) == 1000
