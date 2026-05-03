from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from churn.serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    CustomerFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictResponse,
)


def _valid_customer_payload() -> dict[str, Any]:
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 358.20,
    }


# --- CustomerFeatures ----------------------------------------------------


def test_valid_payload_parses():
    cf = CustomerFeatures(**_valid_customer_payload())
    assert cf.gender == "Female"
    assert cf.tenure == 12


def test_total_charges_can_be_null():
    """Tenure-0 customers have no billing history; the API must accept that."""
    payload = _valid_customer_payload()
    payload["tenure"] = 0
    payload["TotalCharges"] = None
    cf = CustomerFeatures(**payload)
    assert cf.TotalCharges is None


def test_unknown_gender_rejected():
    payload = _valid_customer_payload()
    payload["gender"] = "Other"
    with pytest.raises(ValidationError):
        CustomerFeatures(**payload)


def test_unknown_payment_method_rejected():
    payload = _valid_customer_payload()
    payload["PaymentMethod"] = "Bitcoin"
    with pytest.raises(ValidationError):
        CustomerFeatures(**payload)


def test_negative_tenure_rejected():
    payload = _valid_customer_payload()
    payload["tenure"] = -1
    with pytest.raises(ValidationError):
        CustomerFeatures(**payload)


def test_zero_monthly_charges_rejected():
    payload = _valid_customer_payload()
    payload["MonthlyCharges"] = 0.0
    with pytest.raises(ValidationError):
        CustomerFeatures(**payload)


def test_missing_required_field_rejected():
    payload = _valid_customer_payload()
    del payload["Contract"]
    with pytest.raises(ValidationError):
        CustomerFeatures(**payload)


def test_extra_field_silently_ignored_by_default():
    """Pydantic default is to ignore extras. Document that explicitly."""
    payload = _valid_customer_payload()
    payload["unknown_extra"] = "junk"
    cf = CustomerFeatures(**payload)  # should NOT raise
    assert not hasattr(cf, "unknown_extra")


# --- PredictResponse -----------------------------------------------------


def test_predict_response_validates_probability_range():
    with pytest.raises(ValidationError):
        PredictResponse(churn_probability=1.5, prediction=1, threshold=0.5)


def test_predict_response_prediction_must_be_binary():
    with pytest.raises(ValidationError):
        PredictResponse(churn_probability=0.5, prediction=2, threshold=0.5)  # type: ignore[arg-type]


# --- Batch ---------------------------------------------------------------


def test_batch_request_min_length_one():
    with pytest.raises(ValidationError):
        BatchPredictRequest(records=[])


def test_batch_request_max_length_1000():
    payload = _valid_customer_payload()
    too_many = [CustomerFeatures(**payload) for _ in range(1001)]
    with pytest.raises(ValidationError):
        BatchPredictRequest(records=too_many)


def test_batch_response_aligns_with_request():
    payload = _valid_customer_payload()
    req = BatchPredictRequest(records=[CustomerFeatures(**payload)] * 3)
    resp = BatchPredictResponse(
        predictions=[
            PredictResponse(churn_probability=0.1, prediction=0, threshold=0.5),
            PredictResponse(churn_probability=0.6, prediction=1, threshold=0.5),
            PredictResponse(churn_probability=0.4, prediction=0, threshold=0.5),
        ]
    )
    assert len(resp.predictions) == len(req.records)


# --- Health / model info -------------------------------------------------


def test_health_status_is_constrained():
    assert HealthResponse(status="ok", model_loaded=True).status == "ok"
    with pytest.raises(ValidationError):
        HealthResponse(status="explody", model_loaded=False)  # type: ignore[arg-type]


def test_model_info_serializes_to_json_with_all_required_fields():
    info = ModelInfoResponse(
        registered_model_name="churn_classifier",
        model_type="xgboost",
        model_version="3",
        stage="Production",
        run_id="abc123",
        feature_pipeline_version="v1",
    )
    payload = info.model_dump()
    assert payload["registered_model_name"] == "churn_classifier"
    assert payload["feature_pipeline_version"] == "v1"
