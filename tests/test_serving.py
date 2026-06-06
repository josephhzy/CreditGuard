"""Tests for the FastAPI serving layer.

Uses httpx.AsyncClient with FastAPI's built-in ASGI transport
(no actual network requests).
"""

from __future__ import annotations

import copy
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

if TYPE_CHECKING:
    from fastapi import FastAPI

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_state():
    """Save and restore the module-level _state dict around each test."""
    from serving.app import _state

    snapshot = copy.deepcopy(_state)
    yield
    _state.clear()
    _state.update(snapshot)


# ---------------------------------------------------------------------------
# Helpers: build a test app with a mock model injected
# ---------------------------------------------------------------------------


def _create_test_app(
    model: Any = None,
    feature_names: list[str] | None = None,
    calibrator: Any = None,
) -> FastAPI:
    """Create a CreditGuard app with state overrides for testing.

    The serving stack expects the canonical bundle shape produced by
    ``serving.app._wrap_as_bundle``. We populate it directly here so the
    tests exercise the same code path the real lifespan does.
    """
    from serving.app import _state, create_app

    feats = feature_names if feature_names is not None else []

    bundle = {
        "model": model,
        "scaler": None,
        "calibrator": calibrator,
        "feature_names": list(feats),
        "model_type": "test",
        "version": "test-v0",
        "training_date": "2025-01-01",
    }

    _state["bundle"] = bundle if model is not None else None
    _state["model"] = model
    _state["feature_lookup"] = None
    _state["model_version"] = "test-v0"
    _state["feature_names"] = list(feats)
    # Mirror the production lifespan: build the SHAP explainer if shap can
    # bind to the model. ``_build_explainer`` returns None when binding
    # fails (e.g. MagicMock), which is what we want for negative-path tests.
    from serving.app import _build_explainer

    _state["explainer"] = _build_explainer(model)
    _state["config"] = {
        "decisioning": {
            "bands": {
                "approve": [0.0, 0.15],
                "review": [0.15, 0.40],
                "decline": [0.40, 1.0],
            },
        },
        "evaluation": {
            "discrimination": {"metrics": ["roc_auc"]},
            "threshold": {"cost_matrix": {"fp_cost": 50, "fn_cost": 500}, "target_recall": 0.70},
            "fairness": {"sensitive_features": ["CODE_GENDER"], "tolerance": 0.80},
        },
        "monitoring": {
            "psi": {"threshold_green": 0.10, "threshold_amber": 0.20},
            "drift": {"methods": ["psi"], "alert_threshold": 0.15},
        },
    }
    _state["config_hash"] = "abc123"
    _state["training_date"] = "2025-01-01"
    _state["start_time"] = 1_000_000.0

    app = create_app()
    # Remove lifespan so we don't trigger file loading
    app.router.lifespan_context = _noop_lifespan
    return app


from contextlib import asynccontextmanager

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@asynccontextmanager
async def _noop_lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield


def _make_mock_model() -> MagicMock:
    """Return a mock model that behaves like a sklearn classifier.

    ``predict_proba`` returns a fixed score so tests assert API shape, not
    model behaviour. The real-artifact integration test below covers the
    "different inputs produce different scores" property; no individual
    unit test should rely on it.
    """
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.88, 0.12]])
    model.feature_importances_ = None
    return model


# ---------------------------------------------------------------------------
# Async client fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def client_no_model():
    """Synchronous-friendly client generator with no model loaded."""
    app = _create_test_app(model=None)
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture()
def client_with_model():
    """Synchronous-friendly client generator with a mock model."""
    model = _make_mock_model()
    app = _create_test_app(model=model)
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


# ═══════════════════════════════════════════════════════════════════════════
# Health & metadata
# ═══════════════════════════════════════════════════════════════════════════


class TestHealthEndpoint:
    """Tests for /health and /metadata."""

    @pytest.mark.anyio
    async def test_health_endpoint_returns_503_when_degraded(
        self, client_no_model: AsyncClient
    ) -> None:
        """No-model state must surface as 503 so orchestrators stop routing."""
        async with client_no_model as client:
            resp = await client.get("/health")
        assert resp.status_code == 503
        body = resp.json()
        assert body["model_loaded"] is False
        assert body["status"] == "degraded"

    @pytest.mark.anyio
    async def test_health_endpoint_returns_200_when_ready(
        self, client_with_model: AsyncClient
    ) -> None:
        """Loaded-model state must return 200 so orchestrators route traffic."""
        async with client_with_model as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_loaded"] is True
        assert body["status"] == "ok"

    @pytest.mark.anyio
    async def test_metadata_endpoint(self, client_no_model: AsyncClient) -> None:
        """Metadata endpoint should return model version and feature count.

        With no model loaded the feature count is zero — there is no schema
        to report on. The previous test hardcoded ``feature_count == 3`` by
        seeding ``_state["feature_names"]`` independently of the bundle,
        which masked the model-not-loaded path.
        """
        async with client_no_model as client:
            resp = await client.get("/metadata")
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_version"] == "test-v0"
        assert body["feature_count"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# Prediction
# ═══════════════════════════════════════════════════════════════════════════


class TestPredictEndpoint:
    """Tests for /predict and /predict-batch."""

    @pytest.mark.anyio
    async def test_predict_endpoint(self, client_with_model: AsyncClient) -> None:
        """Single prediction should return risk_score, decision, and risk_band."""
        payload = {
            "SK_ID_CURR": 100001,
            "AMT_INCOME_TOTAL": 200_000.0,
            "AMT_CREDIT": 500_000.0,
            "DAYS_BIRTH": -12000,
        }
        async with client_with_model as client:
            resp = await client.post("/predict", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert "risk_score" in body
        assert "decision" in body
        assert body["sk_id_curr"] == 100001

    @pytest.mark.anyio
    async def test_predict_batch_endpoint(self, client_with_model: AsyncClient) -> None:
        """Batch prediction should return one result per applicant."""
        applicants = [
            {
                "SK_ID_CURR": 1,
                "AMT_INCOME_TOTAL": 100_000,
                "AMT_CREDIT": 300_000,
                "DAYS_BIRTH": -10000,
            },
            {
                "SK_ID_CURR": 2,
                "AMT_INCOME_TOTAL": 200_000,
                "AMT_CREDIT": 600_000,
                "DAYS_BIRTH": -15000,
            },
        ]
        async with client_with_model as client:
            resp = await client.post("/predict-batch", json={"applicants": applicants})
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 2
        assert len(body["predictions"]) == 2

    @pytest.mark.anyio
    async def test_predict_returns_decision_band(self, client_with_model: AsyncClient) -> None:
        """Response should include a valid risk_band string."""
        payload = {
            "SK_ID_CURR": 999,
            "AMT_INCOME_TOTAL": 150_000,
            "AMT_CREDIT": 400_000,
            "DAYS_BIRTH": -11000,
        }
        async with client_with_model as client:
            resp = await client.post("/predict", json=payload)
        body = resp.json()
        assert body["risk_band"] in ("low", "medium", "high")

    @pytest.mark.anyio
    async def test_predict_returns_422_for_unknown_id_without_features(self) -> None:
        """Unknown SK_ID_CURR with no engineered_features must return 422.

        The 422 branch in _build_feature_vector is only reachable when
        feature_names is non-empty (otherwise the empty-schema shortcut fires).
        We wire a client with a real feature list and no feature_lookup so the
        endpoint has no lookup row to fall back on.
        """
        model = _make_mock_model()
        app = _create_test_app(model=model, feature_names=["AMT_INCOME_TOTAL", "AMT_CREDIT"])
        from serving.app import _state

        _state["feature_lookup"] = None

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/predict", json={"SK_ID_CURR": 999999})
        assert resp.status_code == 422
        assert "999999" in resp.json()["detail"]


# ═══════════════════════════════════════════════════════════════════════════
# Explain
# ═══════════════════════════════════════════════════════════════════════════


class TestExplainEndpoint:
    """Tests for /explain (SHAP-based)."""

    @pytest.mark.anyio
    async def test_explain_endpoint_with_mock_model_returns_503(
        self, client_with_model: AsyncClient
    ) -> None:
        """SHAP TreeExplainer cannot operate on a MagicMock, so the cached
        explainer is None at startup and the endpoint returns 503 — i.e.
        "this server can't explain", not "this request errored". A 200
        with garbage values would be a regression; the real-artifact
        integration test exercises the happy 200 path with a real model.
        """
        payload = {
            "SK_ID_CURR": 100001,
            "AMT_INCOME_TOTAL": 200_000.0,
            "AMT_CREDIT": 500_000.0,
            "DAYS_BIRTH": -12000,
        }
        async with client_with_model as client:
            resp = await client.post("/explain", json=payload)
        assert resp.status_code == 503
        assert "detail" in resp.json()

    @pytest.mark.anyio
    async def test_explain_endpoint_serializes_numpy_feature_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Feature rows from parquet carry numpy scalars; responses must be JSON-safe."""

        class FakeExplanation:
            base_values = np.array([0.1])
            values = np.array([[0.2, -0.1]])

        class FakeExplainer:
            def __init__(self, model: Any) -> None:
                self.model = model

            def __call__(self, df: Any) -> FakeExplanation:
                return FakeExplanation()

        monkeypatch.setitem(sys.modules, "shap", SimpleNamespace(TreeExplainer=FakeExplainer))

        import pandas as pd

        from serving.app import _state

        app = _create_test_app(model=_make_mock_model(), feature_names=["f1", "f2"])
        _state["feature_lookup"] = pd.DataFrame(
            {
                "SK_ID_CURR": np.array([100001], dtype=np.int64),
                "f1": np.array([7], dtype=np.int64),
                "f2": np.array([0.25], dtype=np.float64),
            }
        ).set_index("SK_ID_CURR", drop=True)
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/explain", json={"SK_ID_CURR": 100001})

        assert resp.status_code == 200
        body = resp.json()
        assert body["feature_contributions"][0]["value"] == 7
        assert isinstance(body["feature_contributions"][0]["value"], int)


# ═══════════════════════════════════════════════════════════════════════════
# Model card
# ═══════════════════════════════════════════════════════════════════════════


class TestModelCardEndpoint:
    """Tests for /model-card governance endpoint."""

    @pytest.mark.anyio
    async def test_model_card_endpoint(self, client_no_model: AsyncClient) -> None:
        """Model card endpoint should return governance metadata."""
        async with client_no_model as client:
            resp = await client.get("/model-card")
        assert resp.status_code == 200
        body = resp.json()
        assert "model_name" in body
        assert "intended_use" in body
        assert "limitations" in body


# ═══════════════════════════════════════════════════════════════════════════
# Threshold
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# Real-artifact integration — the test the bug review demanded
# ═══════════════════════════════════════════════════════════════════════════


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_BUNDLE = PROJECT_ROOT / "artifacts" / "lightgbm_model.pkl"
REAL_FEATURES = PROJECT_ROOT / "data" / "processed" / "features.parquet"


@pytest.mark.skipif(
    not (REAL_BUNDLE.exists() and REAL_FEATURES.exists()),
    reason="Real artifact + features parquet must be present (training pipeline output)",
)
class TestRealBundleIntegration:
    """End-to-end smoke test against the real LightGBM bundle.

    This is the test that would have caught the artifact-as-dict bug, the
    NaN-fill demo-killer, and the missing-calibrator regression. It loads
    the actual bundle, scores two real applicants whose engineered feature
    vectors are known to differ, and asserts:
      1. Both requests return 200 (the bundle is wired correctly).
      2. Their risk_scores differ (raw fields are not being silently
         NaN-filled).
      3. risk_scores fall in the calibrated range (mean default rate ≈ 8%
         on Home Credit; calibrator applied means most scores live well
         below 0.5, not at the raw classifier mean of ~0.38).
      4. The decision is taken on the raw model score, not the calibrated
         risk_score (the threshold score-scale bug).
    """

    @pytest.mark.anyio
    async def test_two_distinct_applicants_get_distinct_scores(self) -> None:
        from serving.app import _state, _wrap_as_bundle, create_app

        with open(REAL_BUNDLE, "rb") as f:
            loaded = pickle.load(f)
        bundle = _wrap_as_bundle(loaded)

        import pandas as pd

        feature_lookup = pd.read_parquet(REAL_FEATURES).set_index("SK_ID_CURR", drop=True)

        # Pick two applicants whose engineered features are known to differ.
        ids = [100002, 100003]
        for sk in ids:
            assert sk in feature_lookup.index, f"applicant {sk} missing from lookup"

        _state["bundle"] = bundle
        _state["model"] = bundle["model"]
        _state["feature_lookup"] = feature_lookup
        _state["model_version"] = bundle["version"]
        _state["feature_names"] = bundle["feature_names"]
        _state["config"] = {
            "decisioning": {
                "bands": {
                    "approve": [0.0, 0.15],
                    "review": [0.15, 0.40],
                    "decline": [0.40, 1.0],
                },
            },
        }
        _state["config_hash"] = "real"
        _state["training_date"] = bundle["training_date"]
        _state["start_time"] = 1_000_000.0

        app = create_app()
        app.router.lifespan_context = _noop_lifespan
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            scores: list[float] = []
            decisions: list[str] = []
            for sk in ids:
                resp = await client.post("/predict", json={"SK_ID_CURR": sk})
                assert resp.status_code == 200, resp.text
                body = resp.json()
                assert body["sk_id_curr"] == sk
                assert 0.0 <= body["risk_score"] <= 1.0
                scores.append(body["risk_score"])
                decisions.append(body["decision"])

        assert scores[0] != scores[1], (
            "Both applicants returned the same risk_score — the demo-killer "
            "bug from the review (NaN-filling all engineered features) is back."
        )

        # Calibrator sanity: the bundle has a Platt calibrator on OOF preds.
        # The OOF default rate is ~8%, so a sample of two applicants from the
        # training set should not both land above the decline gate. The raw
        # uncalibrated scores would, since the raw mean is ~0.38.
        assert min(scores) < 0.40, (
            f"Both calibrated scores ≥ 0.40 (got {scores}); calibrator may "
            "not be applied — raw classifier mean is ~0.38 so this is the "
            "regression to watch."
        )

        # Regression guard for the threshold score-scale bug: the decision must
        # be taken on the RAW model score, not the calibrated risk_score.
        # Applicant 100002 is a known high-risk case — its raw score (~0.86) is
        # well above the 0.40 decline gate, while its calibrated risk_score
        # (~0.35) sits below it. If the decision were (wrongly) computed from
        # the calibrated PD, 100002 would come back REVIEW.
        assert decisions[0] == "DECLINE", (
            f"Applicant {ids[0]} returned decision={decisions[0]} with "
            f"risk_score={scores[0]}. The decision must be computed from the "
            "raw model score; a non-DECLINE here means serving regressed to "
            "banding the calibrated PD with raw-derived thresholds."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Threshold
# ═══════════════════════════════════════════════════════════════════════════


class TestThresholdEndpoint:
    """Tests for /threshold/* endpoints."""

    @pytest.mark.anyio
    async def test_threshold_endpoint(self, client_no_model: AsyncClient) -> None:
        """Threshold recommendation endpoint should return current and recommended bands."""
        async with client_no_model as client:
            resp = await client.get("/threshold/recommendation")
        assert resp.status_code == 200
        body = resp.json()
        assert "current_thresholds" in body
        assert "recommended_thresholds" in body
        assert len(body["current_thresholds"]) == 3

    @pytest.mark.anyio
    async def test_threshold_simulate_valid(self, client_no_model: AsyncClient) -> None:
        """Valid simulate request returns proposed_thresholds and impact."""
        async with client_no_model as client:
            resp = await client.post(
                "/threshold/simulate",
                json={"approve_upper": 0.15, "decline_lower": 0.40},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert "proposed_thresholds" in body
        assert "impact" in body
        assert len(body["proposed_thresholds"]) == 3

    @pytest.mark.anyio
    async def test_threshold_simulate_invalid_thresholds(
        self, client_no_model: AsyncClient
    ) -> None:
        """approve_upper >= decline_lower must return 422."""
        async with client_no_model as client:
            resp = await client.post(
                "/threshold/simulate",
                json={"approve_upper": 0.50, "decline_lower": 0.30},
            )
        assert resp.status_code == 422
