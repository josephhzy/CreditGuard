"""Drift monitoring endpoint.

GET /drift/report -- latest feature drift report with per-feature
                     statistics, overall severity, and recommended actions.
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog
from fastapi import APIRouter, HTTPException

from serving.schemas import (
    DriftAlert,
    DriftReportResponse,
    DriftSeverity,
    FeatureDrift,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/drift", tags=["monitoring"])

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default path where the monitoring pipeline writes the latest drift report
_DEFAULT_REPORT_PATH = _PROJECT_ROOT / "monitoring" / "reports" / "latest_drift.json"


def _load_drift_report(report_path: Path | None = None) -> dict:
    """Load the most recent drift report from disk.

    The monitoring module is expected to write a JSON file with
    per-feature drift statistics after each scheduled run.
    """
    path = report_path or _DEFAULT_REPORT_PATH
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No drift report found at {path}. Run the monitoring pipeline first.",
        )
    with open(path) as f:
        data: dict = json.load(f)
    return data


def _parse_severity(value: str | float) -> DriftSeverity:
    """Convert a string or numeric severity to the enum."""
    if isinstance(value, str):
        return DriftSeverity(value.upper())
    # Numeric: interpret as PSI-style threshold
    if value < 0.10:
        return DriftSeverity.GREEN
    elif value < 0.20:
        return DriftSeverity.AMBER
    return DriftSeverity.RED


def _build_report(raw: dict) -> DriftReportResponse:
    """Transform the raw JSON report into the API response schema."""
    from serving.app import get_state

    state = get_state()
    config = state.get("config", {})
    monitoring_cfg = config.get("monitoring", {})
    drift_cfg = monitoring_cfg.get("drift", {})
    alert_threshold = drift_cfg.get("alert_threshold", 0.15)

    feature_drifts: list[FeatureDrift] = []
    alerts: list[DriftAlert] = []

    for entry in raw.get("features", []):
        severity = _parse_severity(entry.get("severity", entry.get("statistic", 0.0)))
        fd = FeatureDrift(
            feature=entry["feature"],
            method=entry.get("method", "psi"),
            statistic=round(entry.get("statistic", 0.0), 6),
            threshold=entry.get("threshold", alert_threshold),
            drifted=entry.get("drifted", entry.get("statistic", 0.0) > alert_threshold),
            severity=severity,
        )
        feature_drifts.append(fd)

        if fd.drifted:
            alerts.append(
                DriftAlert(
                    feature=fd.feature,
                    message=f"{fd.feature} shows significant drift ({fd.method}={fd.statistic:.4f})",
                    recommended_action=(
                        "Investigate distribution shift and consider retraining"
                        if severity == DriftSeverity.RED
                        else "Monitor closely in the next reporting period"
                    ),
                )
            )

    # Derive overall severity
    if any(fd.severity == DriftSeverity.RED for fd in feature_drifts):
        overall = DriftSeverity.RED
    elif any(fd.severity == DriftSeverity.AMBER for fd in feature_drifts):
        overall = DriftSeverity.AMBER
    else:
        overall = DriftSeverity.GREEN

    recommended_actions: list[str] = []
    if overall == DriftSeverity.RED:
        recommended_actions.append("Trigger model retraining pipeline.")
        recommended_actions.append("Review feature engineering for shifted features.")
    elif overall == DriftSeverity.AMBER:
        recommended_actions.append("Schedule review of drifting features within one week.")

    return DriftReportResponse(
        report_date=raw.get("report_date", "unknown"),
        overall_severity=overall,
        feature_drift=feature_drifts,
        alerts=alerts,
        recommended_actions=recommended_actions,
    )


@router.get("/report", response_model=DriftReportResponse)
async def drift_report() -> DriftReportResponse:
    """Return the latest drift monitoring report."""
    logger.info("drift_report_requested")
    raw = _load_drift_report()
    return _build_report(raw)
