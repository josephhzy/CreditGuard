"""Monitoring alert manager for credit model operations.

Aggregates signals from drift detection, PSI monitoring, and calibration
tracking into a unified alert system with traffic-light severities.
Produces formatted summaries for operations dashboards and determines
whether accumulated alerts should trigger a model retrain.

Typical usage::

    from monitoring.alert_manager import generate_alerts, format_alert_summary

    alerts = generate_alerts(drift_report, psi_report, calibration_report)
    summary = format_alert_summary(alerts)
    should_retrain = retrain_trigger(alerts)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from monitoring.calibration_drift import CalibrationDriftResult
from monitoring.drift_engine import DriftReport

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Alert data structures
# ---------------------------------------------------------------------------


class AlertSeverity(str, enum.Enum):
    """Alert severity levels."""

    GREEN = "green"
    AMBER = "amber"
    RED = "red"


@dataclass
class Alert:
    """A single monitoring alert."""

    feature: str
    metric: str
    value: float
    severity: AlertSeverity
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a flat dict."""
        return {
            "feature": self.feature,
            "metric": self.metric,
            "value": round(self.value, 6),
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Alert thresholds (defaults; overridable)
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLDS = {
    "psi_amber": 0.10,
    "psi_red": 0.20,
    "ks_amber": 0.10,
    "ks_red": 0.20,
    "js_amber": 0.05,
    "js_red": 0.10,
    "calibration_brier_amber": 0.02,
    "calibration_brier_red": 0.05,
}


# ---------------------------------------------------------------------------
# Alert generation
# ---------------------------------------------------------------------------


def generate_alerts(
    drift_report: DriftReport | None = None,
    psi_values: dict[str, float] | None = None,
    calibration_result: CalibrationDriftResult | None = None,
    *,
    thresholds: dict[str, float] | None = None,
) -> list[Alert]:
    """Generate alerts from monitoring reports.

    Parameters
    ----------
    drift_report:
        Output from ``drift_engine.detect_drift()``.
    psi_values:
        Per-feature PSI values from ``psi.psi_per_feature()``.
    calibration_result:
        Output from ``calibration_drift.detect_calibration_drift()``.
    thresholds:
        Override default alert thresholds.

    Returns
    -------
    list[Alert]
        All generated alerts, sorted by severity (red first).
    """
    t = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}
    alerts: list[Alert] = []

    # --- Drift report alerts ---
    if drift_report is not None:
        for fd in drift_report.per_feature:
            # PSI from drift report
            if fd.psi >= t["psi_red"]:
                alerts.append(
                    Alert(
                        feature=fd.feature,
                        metric="psi",
                        value=fd.psi,
                        severity=AlertSeverity.RED,
                        message=f"PSI={fd.psi:.4f} (RED) for {fd.feature}. Significant distribution shift detected.",
                    )
                )
            elif fd.psi >= t["psi_amber"]:
                alerts.append(
                    Alert(
                        feature=fd.feature,
                        metric="psi",
                        value=fd.psi,
                        severity=AlertSeverity.AMBER,
                        message=f"PSI={fd.psi:.4f} (AMBER) for {fd.feature}. Moderate shift detected.",
                    )
                )

            # KS test
            if fd.ks_statistic >= t["ks_red"]:
                alerts.append(
                    Alert(
                        feature=fd.feature,
                        metric="ks_statistic",
                        value=fd.ks_statistic,
                        severity=AlertSeverity.RED,
                        message=f"KS={fd.ks_statistic:.4f} (RED) for {fd.feature}. Distribution significantly different.",
                    )
                )
            elif fd.ks_statistic >= t["ks_amber"]:
                alerts.append(
                    Alert(
                        feature=fd.feature,
                        metric="ks_statistic",
                        value=fd.ks_statistic,
                        severity=AlertSeverity.AMBER,
                        message=f"KS={fd.ks_statistic:.4f} (AMBER) for {fd.feature}.",
                    )
                )

            # JS divergence
            if fd.js_divergence >= t["js_red"]:
                alerts.append(
                    Alert(
                        feature=fd.feature,
                        metric="js_divergence",
                        value=fd.js_divergence,
                        severity=AlertSeverity.RED,
                        message=f"JS={fd.js_divergence:.4f} (RED) for {fd.feature}.",
                    )
                )
            elif fd.js_divergence >= t["js_amber"]:
                alerts.append(
                    Alert(
                        feature=fd.feature,
                        metric="js_divergence",
                        value=fd.js_divergence,
                        severity=AlertSeverity.AMBER,
                        message=f"JS={fd.js_divergence:.4f} (AMBER) for {fd.feature}.",
                    )
                )

    # --- Standalone PSI alerts (if provided separately) ---
    if psi_values is not None:
        for feat, psi_val in psi_values.items():
            # Avoid duplicating alerts already generated from drift_report
            if drift_report is not None:
                existing = [a for a in alerts if a.feature == feat and a.metric == "psi"]
                if existing:
                    continue

            if psi_val >= t["psi_red"]:
                alerts.append(
                    Alert(
                        feature=feat,
                        metric="psi",
                        value=psi_val,
                        severity=AlertSeverity.RED,
                        message=f"PSI={psi_val:.4f} (RED) for {feat}.",
                    )
                )
            elif psi_val >= t["psi_amber"]:
                alerts.append(
                    Alert(
                        feature=feat,
                        metric="psi",
                        value=psi_val,
                        severity=AlertSeverity.AMBER,
                        message=f"PSI={psi_val:.4f} (AMBER) for {feat}.",
                    )
                )

    # --- Calibration alerts ---
    if calibration_result is not None:
        delta = calibration_result.brier_delta
        if delta >= t["calibration_brier_red"]:
            alerts.append(
                Alert(
                    feature="_model_",
                    metric="calibration_brier_delta",
                    value=delta,
                    severity=AlertSeverity.RED,
                    message=f"Calibration Brier delta={delta:.4f} (RED). {calibration_result.message}",
                )
            )
        elif delta >= t["calibration_brier_amber"]:
            alerts.append(
                Alert(
                    feature="_model_",
                    metric="calibration_brier_delta",
                    value=delta,
                    severity=AlertSeverity.AMBER,
                    message=f"Calibration Brier delta={delta:.4f} (AMBER). {calibration_result.message}",
                )
            )

    # Sort by severity: RED first, then AMBER, then GREEN
    severity_order = {AlertSeverity.RED: 0, AlertSeverity.AMBER: 1, AlertSeverity.GREEN: 2}
    alerts.sort(key=lambda a: (severity_order[a.severity], -a.value))

    n_red = sum(1 for a in alerts if a.severity == AlertSeverity.RED)
    n_amber = sum(1 for a in alerts if a.severity == AlertSeverity.AMBER)

    logger.info(
        "alerts_generated",
        total=len(alerts),
        n_red=n_red,
        n_amber=n_amber,
    )
    return alerts


# ---------------------------------------------------------------------------
# Alert formatting
# ---------------------------------------------------------------------------


def format_alert_summary(alerts: list[Alert]) -> str:
    """Format alerts as a markdown table for dashboard display.

    Parameters
    ----------
    alerts:
        List of Alert objects to format.

    Returns
    -------
    str
        Markdown-formatted summary table.
    """
    if not alerts:
        return "## Monitoring Alert Summary\n\nNo alerts. All metrics within acceptable ranges."

    lines: list[str] = [
        "## Monitoring Alert Summary",
        "",
        f"**Total alerts:** {len(alerts)}",
        f"**RED:** {sum(1 for a in alerts if a.severity == AlertSeverity.RED)}  |  "
        f"**AMBER:** {sum(1 for a in alerts if a.severity == AlertSeverity.AMBER)}",
        "",
        "| Severity | Feature | Metric | Value | Message |",
        "|----------|---------|--------|-------|---------|",
    ]

    for alert in alerts:
        sev_icon = {
            AlertSeverity.RED: "RED",
            AlertSeverity.AMBER: "AMBER",
            AlertSeverity.GREEN: "GREEN",
        }[alert.severity]

        lines.append(
            f"| {sev_icon} | {alert.feature} | {alert.metric} | "
            f"{alert.value:.4f} | {alert.message} |"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Retrain trigger
# ---------------------------------------------------------------------------


def retrain_trigger(
    alerts: list[Alert],
    *,
    red_threshold: int = 1,
    amber_threshold: int = 3,
) -> bool:
    """Determine whether accumulated alerts should trigger a model retrain.

    Triggers retrain if:
    - Any RED alert on calibration
    - ``red_threshold`` or more RED feature drift alerts
    - ``amber_threshold`` or more AMBER alerts across all metrics

    Parameters
    ----------
    alerts:
        List of Alert objects from ``generate_alerts()``.
    red_threshold:
        Number of RED alerts that triggers retrain.
    amber_threshold:
        Number of AMBER alerts that triggers retrain.

    Returns
    -------
    bool
        ``True`` if retrain is recommended.

    Note
    ----
    This function evaluates the **current** alert list only.  It contains no
    time-window or sustained-duration logic.  The 7-day sustain window for
    RED drift alerts (and the 30-day window for performance alerts) described
    in ``monitoring/RETRAIN_POLICY.md`` is enforced by the **caller**, not
    here.  A ``True`` return from this function means "the threshold has been
    crossed right now"; whether that warrants an actual retrain depends on
    whether the condition has persisted across the required number of
    consecutive daily runs.
    """
    n_red = len(
        {a.feature for a in alerts if a.severity == AlertSeverity.RED and a.feature != "_model_"}
    )
    n_amber = sum(1 for a in alerts if a.severity == AlertSeverity.AMBER)
    has_calibration_red = any(
        a.severity == AlertSeverity.RED and "calibration" in a.metric for a in alerts
    )

    should_retrain = has_calibration_red or n_red >= red_threshold or n_amber >= amber_threshold

    logger.info(
        "retrain_trigger_evaluated",
        should_retrain=should_retrain,
        n_red=n_red,
        n_amber=n_amber,
        has_calibration_red=has_calibration_red,
    )
    return should_retrain


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build synthetic alerts for demonstration
    from monitoring.calibration_drift import CalibrationDriftResult
    from monitoring.drift_engine import DriftReport, FeatureDrift

    # Simulate a drift report with some problematic features
    features_drift = [
        FeatureDrift(
            feature="income",
            psi=0.25,
            psi_severity="red",
            ks_statistic=0.18,
            ks_pvalue=0.001,
            js_divergence=0.12,
            mean_shift=0.5,
            direction="shift_up",
        ),
        FeatureDrift(
            feature="age",
            psi=0.05,
            psi_severity="green",
            ks_statistic=0.03,
            ks_pvalue=0.45,
            js_divergence=0.01,
            mean_shift=-0.1,
            direction="stable",
        ),
        FeatureDrift(
            feature="debt_ratio",
            psi=0.14,
            psi_severity="amber",
            ks_statistic=0.11,
            ks_pvalue=0.02,
            js_divergence=0.06,
            mean_shift=0.2,
            direction="shift_up",
        ),
    ]

    drift_rpt = DriftReport(
        per_feature=features_drift,
        top_drifting_features=["income", "debt_ratio"],
        overall_severity="red",
        n_features_green=1,
        n_features_amber=1,
        n_features_red=1,
        root_cause_notes=["income: PSI=0.2500 [red], mean shifted shift_up by +0.5000"],
    )

    cal_result = CalibrationDriftResult(
        reference_brier=0.065,
        current_brier=0.092,
        brier_delta=0.027,
        has_drifted=True,
        severity="amber",
        message="Calibration moderately degraded.",
    )

    print("=== Generate Alerts ===")
    alerts = generate_alerts(drift_report=drift_rpt, calibration_result=cal_result)
    for a in alerts:
        print(f"  [{a.severity.value.upper()}] {a.feature} / {a.metric}: {a.message}")

    print("\n=== Formatted Summary ===")
    summary = format_alert_summary(alerts)
    print(summary)

    print(f"\n=== Retrain Trigger: {retrain_trigger(alerts)} ===")
