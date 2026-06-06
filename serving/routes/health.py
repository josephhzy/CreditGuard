"""Health and metadata endpoints.

GET /health   -- liveness check (200 = ready, 503 = degraded)
GET /metadata -- model version, feature count, training date, config hash
"""

from __future__ import annotations

import time

import structlog
from fastapi import APIRouter, Response

from serving.schemas import HealthResponse, MetadataResponse

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(response: Response) -> HealthResponse:
    """Readiness check returning model status and uptime.

    Returns HTTP 503 when no model is loaded so orchestrators (Docker
    healthcheck, k8s readinessProbe, load balancers) treat the instance as
    not-ready and stop routing traffic to it. Previously this returned 200
    with ``status=degraded`` — the response body said "broken" but the
    status code said "fine", which is exactly the wrong signal for a
    readiness probe.
    """
    from serving.app import get_state

    state = get_state()
    model_loaded = state["model"] is not None
    uptime = round(time.time() - state["start_time"], 2)

    if not model_loaded:
        response.status_code = 503

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_version=state["model_version"],
        uptime_seconds=uptime,
    )


@router.get("/metadata", response_model=MetadataResponse)
async def metadata() -> MetadataResponse:
    """Return model and configuration metadata."""
    from serving.app import get_state

    state = get_state()
    return MetadataResponse(
        model_version=state["model_version"],
        feature_count=len(state["feature_names"]),
        training_date=state["training_date"],
        config_hash=state["config_hash"],
    )
