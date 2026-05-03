"""Pydantic request / response schemas for the serving API.

The categorical values are hardcoded as ``Literal`` types here, intentionally
duplicated from the Pandera ``RAW_SCHEMA`` enums. The data layer's allowed
values describe the *training distribution*; the API's allowed values describe
the *public contract*. They happen to coincide today, but coupling them via a
shared list would let an upstream data change silently break the OpenAPI
schema clients depend on. The duplication is the contract boundary.

``TotalCharges`` is nullable because tenure-0 customers have no billing history
yet — the loader replaces null with 0.0 before passing to the feature pipeline,
matching the training-time preprocessing.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """One Telco customer record. Mirrors the columns the feature pipeline expects."""

    # Pydantic v2 reserves ``model_*`` for its API; we never use a ``model_``
    # field name on this type, but keep the override consistent across the codebase.
    model_config = ConfigDict(protected_namespaces=())

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=100)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(gt=0.0, le=200.0)
    TotalCharges: float | None = Field(default=None, ge=0.0)


class PredictResponse(BaseModel):
    """Prediction for a single customer."""

    model_config = ConfigDict(protected_namespaces=())

    churn_probability: float = Field(ge=0.0, le=1.0)
    prediction: Literal[0, 1]
    threshold: float = Field(ge=0.0, le=1.0)


class BatchPredictRequest(BaseModel):
    """Batch input: list of customer records (max 1000 to bound request size)."""

    records: list[CustomerFeatures] = Field(min_length=1, max_length=1000)


class BatchPredictResponse(BaseModel):
    """Batch output: aligned with the input order."""

    predictions: list[PredictResponse]


class HealthResponse(BaseModel):
    """Liveness + model-loaded indicator. Used by container orchestrators."""

    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_type: str | None = None
    model_version: str | None = None


class ModelInfoResponse(BaseModel):
    """Metadata about the version currently being served."""

    model_config = ConfigDict(protected_namespaces=())

    registered_model_name: str
    model_type: str
    model_version: str
    stage: str
    run_id: str
    feature_pipeline_version: str | None = None
