"""Pydantic request/response models for the CreditGuard API.

All models use Pydantic v2 conventions. Field descriptions serve as
automatic OpenAPI documentation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Decision(str, Enum):
    """Lending decision mapped from the model risk score."""

    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    DECLINE = "DECLINE"


class DriftSeverity(str, Enum):
    """Overall severity level for a drift monitoring report."""

    GREEN = "GREEN"
    AMBER = "AMBER"
    RED = "RED"


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Single-applicant prediction request.

    Contains the key application-level fields plus any derived features
    that the model expects.  Unknown/optional fields are accepted via
    ``model_config`` so callers can pass the full row without filtering.
    """

    model_config = {"extra": "allow"}

    SK_ID_CURR: int = Field(..., description="Unique applicant identifier")
    AMT_INCOME_TOTAL: float | None = Field(None, description="Total income")
    AMT_CREDIT: float | None = Field(None, description="Credit amount of the loan")
    AMT_ANNUITY: float | None = Field(None, description="Loan annuity")
    AMT_GOODS_PRICE: float | None = Field(
        None, description="Price of the goods for which the loan is given"
    )
    DAYS_BIRTH: float | None = Field(
        None, description="Days before application the client was born (negative)"
    )
    DAYS_EMPLOYED: float | None = Field(
        None, description="Days before application the person started employment"
    )
    NAME_EDUCATION_TYPE: str | None = Field(None, description="Education level")
    NAME_HOUSING_TYPE: str | None = Field(None, description="Housing situation")
    CODE_GENDER: str | None = Field(None, description="Gender")
    CNT_CHILDREN: int | None = Field(None, description="Number of children")
    CNT_FAM_MEMBERS: float | None = Field(None, description="Number of family members")

    engineered_features: dict[str, float] | None = Field(
        default=None,
        description=(
            "Pre-computed engineered feature vector (keys must match the model's "
            "training feature schema, e.g. 'feat_*'). Production callers should "
            "supply this directly from a feature store. When omitted, the API "
            "looks up the engineered features by SK_ID_CURR from the demo "
            "lookup table; if neither is available, the request is rejected."
        ),
    )


class TopFactor(BaseModel):
    """A single influential feature in the prediction."""

    feature: str
    value: Any
    contribution: float = Field(description="SHAP value or similar contribution measure")


class PredictResponse(BaseModel):
    """Response for a single-applicant prediction."""

    sk_id_curr: int
    risk_score: float = Field(ge=0.0, le=1.0, description="Predicted probability of default")
    risk_band: str = Field(description="Human-readable risk band label")
    decision: Decision
    top_factors: list[TopFactor] = Field(
        default_factory=list, description="Top contributing features"
    )
    model_version: str
    confidence_note: str = Field(
        default="",
        description="Caveats about prediction confidence, e.g. extrapolation warnings",
    )


class BatchPredictRequest(BaseModel):
    """Batch prediction request containing multiple applicants."""

    applicants: list[PredictRequest] = Field(..., min_length=1, max_length=1000)


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""

    predictions: list[PredictResponse]
    count: int


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------


class FeatureContribution(BaseModel):
    """One feature's contribution to the SHAP explanation."""

    feature: str
    value: Any
    shap_value: float


class ExplainResponse(BaseModel):
    """Detailed SHAP explanation for a single applicant."""

    sk_id_curr: int
    base_value: float = Field(description="Expected model output (base rate)")
    feature_contributions: list[FeatureContribution] = Field(
        description="Feature contributions sorted by |shap_value| descending"
    )
    summary: str = Field(description="Natural-language explanation summary")


# ---------------------------------------------------------------------------
# Drift monitoring
# ---------------------------------------------------------------------------


class FeatureDrift(BaseModel):
    """Drift statistics for a single feature."""

    feature: str
    method: str = Field(description="Drift detection method (psi, ks, js_divergence)")
    statistic: float
    threshold: float
    drifted: bool
    severity: DriftSeverity


class DriftAlert(BaseModel):
    """An actionable alert raised by drift monitoring."""

    feature: str
    message: str
    recommended_action: str


class DriftReportResponse(BaseModel):
    """Aggregated drift monitoring report."""

    report_date: str
    overall_severity: DriftSeverity
    feature_drift: list[FeatureDrift]
    alerts: list[DriftAlert]
    recommended_actions: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Model card / governance
# ---------------------------------------------------------------------------


class PerformanceMetric(BaseModel):
    """A single performance metric from training evaluation."""

    metric: str
    value: float
    dataset: str = Field(default="validation", description="Dataset the metric was computed on")


class ModelCardResponse(BaseModel):
    """Model governance metadata (model card)."""

    model_name: str
    version: str
    description: str
    training_date: str
    intended_use: str
    non_use_policy: str
    limitations: list[str]
    performance_metrics: list[PerformanceMetric]
    sensitive_features: list[str] = Field(default_factory=list)
    fairness_notes: str = ""


# ---------------------------------------------------------------------------
# Health / metadata
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Liveness and readiness check response."""

    status: str = Field(description="'ok' or 'degraded'")
    model_loaded: bool
    model_version: str
    uptime_seconds: float


class MetadataResponse(BaseModel):
    """Model and configuration metadata."""

    model_version: str
    feature_count: int
    training_date: str
    config_hash: str


# ---------------------------------------------------------------------------
# Threshold management
# ---------------------------------------------------------------------------


class ThresholdBand(BaseModel):
    """A single decision threshold band."""

    band: str
    lower: float
    upper: float


class ImpactSummary(BaseModel):
    """Simulated impact of threshold changes."""

    approve_rate: float
    review_rate: float
    decline_rate: float
    estimated_default_rate: float
    estimated_loss_reduction: float


class ThresholdResponse(BaseModel):
    """Current and recommended decision thresholds."""

    current_thresholds: list[ThresholdBand]
    recommended_thresholds: list[ThresholdBand]
    impact_summary: ImpactSummary


class ThresholdSimulateRequest(BaseModel):
    """Request to simulate new threshold settings."""

    approve_upper: float = Field(ge=0.0, le=1.0, description="Upper bound for APPROVE band")
    decline_lower: float = Field(ge=0.0, le=1.0, description="Lower bound for DECLINE band")


class ThresholdSimulateResponse(BaseModel):
    """Result of a threshold simulation."""

    proposed_thresholds: list[ThresholdBand]
    impact: ImpactSummary
    warnings: list[str] = Field(default_factory=list)
