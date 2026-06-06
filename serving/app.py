"""FastAPI application factory for CreditGuard.

Creates the application, loads the champion model on startup, registers
all route routers, and wires up middleware for structured logging, CORS,
and exception handling.

Start locally with::

    uvicorn serving.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import hashlib
import os
import pickle
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
import yaml
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from serving.routes import drift, explain, health, model_card, predict, threshold

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Shared application state (populated at startup)
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {
    "model": None,
    "bundle": None,
    "feature_lookup": None,
    "model_version": "unknown",
    "feature_names": [],
    "config": {},
    "config_hash": "",
    "training_date": "unknown",
    "start_time": 0.0,
    # SHAP TreeExplainer is built once at startup (~hundreds of ms for the
    # shipped LightGBM bundle) and reused by /predict and /explain. Building
    # per request was the dominant /explain latency before this cache.
    "explainer": None,
}


def _wrap_as_bundle(loaded: Any) -> dict[str, Any]:
    """Normalise a loaded artifact into the canonical bundle shape.

    The training pipeline saves a dict ``{"model", "scaler", "feature_names",
    "model_type", "params", ...}`` (and optionally ``calibrator``/``version``).
    Tests inject a naked ``MagicMock`` directly as the model. Both shapes are
    accepted here so the serving stack never has to care which one it got.
    """
    if isinstance(loaded, dict) and "model" in loaded:
        return {
            "model": loaded["model"],
            "scaler": loaded.get("scaler"),
            "calibrator": loaded.get("calibrator"),
            "feature_names": list(loaded.get("feature_names", [])),
            "model_type": loaded.get("model_type", "unknown"),
            "version": loaded.get("version", "0.1.0"),
            "training_date": loaded.get("training_date", "unknown"),
        }
    return {
        "model": loaded,
        "scaler": None,
        "calibrator": None,
        "feature_names": list(getattr(loaded, "feature_names_in_", []) or []),
        "model_type": "unknown",
        "version": getattr(loaded, "version_", "0.1.0"),
        "training_date": getattr(loaded, "training_date_", "unknown"),
    }


def get_state() -> dict[str, Any]:
    """Return the shared application state dict.

    Route modules import this to access the loaded model and metadata
    without circular dependencies.
    """
    return _state


# ---------------------------------------------------------------------------
# Config / model loading helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_config() -> dict:
    """Load ``config/train.yaml`` and return the parsed dict."""
    config_path = _PROJECT_ROOT / "config" / "train.yaml"
    if not config_path.exists():
        logger.warning("config_not_found", path=str(config_path))
        return {}
    with open(config_path) as f:
        cfg: dict = yaml.safe_load(f) or {}
    # Compute a short hash for change-detection
    raw = config_path.read_bytes()
    _state["config_hash"] = hashlib.sha256(raw).hexdigest()[:12]
    return cfg


def _load_model(model_path: str | None) -> Any:
    """Load a pickled model from *model_path*.

    Returns ``None`` (with a warning) if the file does not exist so the
    API can still start in a degraded state for health-check purposes.
    """
    if model_path is None:
        model_path = os.environ.get("MODEL_PATH", "artifacts/lightgbm_model.pkl")

    path = Path(model_path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path

    if not path.exists():
        logger.warning("model_file_not_found", path=str(path))
        return None

    logger.info("loading_model", path=str(path))

    # Verify model integrity via SHA-256 sidecar if available
    model_bytes = path.read_bytes()
    model_hash = hashlib.sha256(model_bytes).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.exists():
        expected = sidecar.read_text().strip().split()[0]
        if model_hash != expected:
            logger.error("model_hash_mismatch", expected=expected, actual=model_hash)
            raise ValueError(f"Model integrity check failed for {path}")
        logger.info("model_hash_verified", hash=model_hash[:12])
    else:
        logger.info(
            "model_hash_computed",
            hash=model_hash[:12],
            detail="No .sha256 sidecar found \u2014 skipping verification",
        )

    import io

    loaded = pickle.load(io.BytesIO(model_bytes))  # noqa: S301 \u2014 trusted internal artifact
    logger.info(
        "model_loaded",
        path=str(path),
        is_bundle=isinstance(loaded, dict),
    )
    return loaded


def _load_feature_schema() -> list[str]:
    """Load expected feature names from the feature schema file."""
    schema_path_env = os.environ.get("FEATURE_SCHEMA_PATH", "config/feature_schema.yaml")
    schema_path = Path(schema_path_env)
    if not schema_path.is_absolute():
        schema_path = _PROJECT_ROOT / schema_path

    if not schema_path.exists():
        logger.warning("feature_schema_not_found", path=str(schema_path))
        return []

    with open(schema_path) as f:
        schema = yaml.safe_load(f)

    features: list[str] = schema.get("features", [])
    logger.info("feature_schema_loaded", n_features=len(features))
    return features


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


def _load_feature_lookup() -> Any:
    """Load the precomputed engineered-features parquet for SK_ID_CURR lookup.

    Used by ``/predict`` when a caller sends raw applicant fields instead of
    the full engineered feature vector. Returns ``None`` if the file is not
    present (the lookup path becomes unavailable, but serving still works
    when callers supply ``engineered_features`` directly).
    """
    lookup_path_env = os.environ.get("FEATURE_LOOKUP_PATH", "data/processed/features.parquet")
    lookup_path = Path(lookup_path_env)
    if not lookup_path.is_absolute():
        lookup_path = _PROJECT_ROOT / lookup_path

    if not lookup_path.exists():
        logger.warning("feature_lookup_not_found", path=str(lookup_path))
        return None

    try:
        import pandas as pd

        df = pd.read_parquet(lookup_path)
        if "SK_ID_CURR" not in df.columns:
            logger.warning("feature_lookup_missing_id", path=str(lookup_path))
            return None
        df = df.set_index("SK_ID_CURR", drop=True)
        logger.info("feature_lookup_loaded", n_rows=len(df), n_cols=df.shape[1])
        return df
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("feature_lookup_load_failed", error=str(exc))
        return None


def _build_explainer(model: Any) -> Any:
    """Construct a SHAP TreeExplainer once, at startup.

    Returns ``None`` if SHAP cannot bind to the model (e.g. an unsupported
    estimator type or a stub injected by tests). Routes that need SHAP fall
    back to model feature_importances_ when this is ``None``.
    """
    if model is None:
        return None
    try:
        import shap

        return shap.TreeExplainer(model)
    except Exception as exc:
        logger.warning("explainer_build_failed", error=str(exc), type=type(exc).__name__)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: load model and config on startup."""
    _state["start_time"] = time.time()
    _state["config"] = _load_config()

    loaded = _load_model(os.environ.get("MODEL_PATH"))
    if loaded is None:
        _state["bundle"] = None
        _state["model"] = None
        _state["model_version"] = "no-model"
        _state["training_date"] = "unknown"
        _state["feature_names"] = _load_feature_schema()
    else:
        bundle = _wrap_as_bundle(loaded)
        _state["bundle"] = bundle
        _state["model"] = bundle["model"]
        _state["model_version"] = bundle["version"]
        _state["training_date"] = bundle["training_date"]
        # Prefer feature names from the bundle; fall back to the schema file.
        _state["feature_names"] = bundle["feature_names"] or _load_feature_schema()

    _state["feature_lookup"] = _load_feature_lookup()
    _state["explainer"] = _build_explainer(_state["model"])

    logger.info(
        "startup_complete",
        model_loaded=_state["model"] is not None,
        has_calibrator=bool(_state["bundle"] and _state["bundle"].get("calibrator")),
        explainer_ready=_state["explainer"] is not None,
        model_version=_state["model_version"],
        feature_count=len(_state["feature_names"]),
        feature_lookup_available=_state["feature_lookup"] is not None,
    )
    yield
    logger.info("shutdown")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    application = FastAPI(
        title="CreditGuard API",
        description="Credit Default Early Warning and Decisioning System",
        version="0.1.0",
        lifespan=lifespan,
    )

    # -- CORS --
    application.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- Structured logging middleware --
    @application.middleware("http")
    async def logging_middleware(request: Request, call_next: Any) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start = time.perf_counter()

        # Attach request_id for downstream log entries
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        response: Response = await call_next(request)

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=duration_ms,
        )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Duration-Ms"] = str(duration_ms)
        return response

    # -- Exception handlers --
    @application.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        logger.error("validation_error", detail=str(exc))
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @application.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled_exception", error=str(exc), type=type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. Check server logs for details."},
        )

    # -- Routers --
    application.include_router(health.router)
    application.include_router(predict.router)
    application.include_router(explain.router)
    application.include_router(drift.router)
    application.include_router(model_card.router)
    application.include_router(threshold.router)

    return application


# Module-level instance used by ``uvicorn serving.app:app``
app = create_app()
