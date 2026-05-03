"""End-to-end serving tests via FastAPI's TestClient.

Most tests reuse the session-scoped Production model from conftest. Tests that
need a *missing* model (degraded health) build their own per-test fixture
since they need an empty registry.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import mlflow
import pytest
from fastapi.testclient import TestClient

from churn.serving.app import create_app

VALID_PAYLOAD = {
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


@pytest.fixture
def client_no_production_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """Empty MLflow registry → API starts but reports degraded."""
    db = tmp_path / "mlruns.db"
    tracking_uri = f"sqlite:///{db}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test_no_model")
    monkeypatch.setenv("MODEL_NAME", "empty_registry_classifier")
    mlflow.set_tracking_uri(tracking_uri)
    app = create_app()
    with TestClient(app) as client:
        yield client


# --- /health -------------------------------------------------------------


def test_health_ok_when_model_loaded(production_test_client: TestClient):
    r = production_test_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_type"] == "logreg"
    assert body["model_version"] is not None


def test_health_degraded_when_no_model(client_no_production_model: TestClient):
    r = client_no_production_model.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False


# --- /model_info ---------------------------------------------------------


def test_model_info_returns_metadata(production_test_client: TestClient, production_env):
    r = production_test_client.get("/model_info")
    assert r.status_code == 200
    body = r.json()
    assert body["registered_model_name"] == production_env.registry_name
    assert body["model_type"] == "logreg"
    assert body["stage"] == "Production"
    assert body["feature_pipeline_version"] == "v1"
    assert body["run_id"]


def test_model_info_503_when_no_model(client_no_production_model: TestClient):
    r = client_no_production_model.get("/model_info")
    assert r.status_code == 503


# --- /predict ------------------------------------------------------------


def test_predict_happy_path(production_test_client: TestClient):
    r = production_test_client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert body["threshold"] == 0.5


def test_predict_handles_null_total_charges(production_test_client: TestClient):
    payload = {**VALID_PAYLOAD, "tenure": 0, "TotalCharges": None}
    r = production_test_client.post("/predict", json=payload)
    assert r.status_code == 200


def test_predict_rejects_bad_payment_method(production_test_client: TestClient):
    payload = {**VALID_PAYLOAD, "PaymentMethod": "Bitcoin"}
    r = production_test_client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_rejects_missing_field(production_test_client: TestClient):
    payload = {**VALID_PAYLOAD}
    del payload["Contract"]
    r = production_test_client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_503_when_no_model(client_no_production_model: TestClient):
    r = client_no_production_model.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 503


# --- /predict_batch ------------------------------------------------------


def test_predict_batch_returns_aligned_predictions(
    production_test_client: TestClient,
):
    r = production_test_client.post(
        "/predict_batch",
        json={"records": [VALID_PAYLOAD, VALID_PAYLOAD, VALID_PAYLOAD]},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 3
    for p in body["predictions"]:
        assert 0.0 <= p["churn_probability"] <= 1.0


def test_predict_batch_rejects_empty_list(production_test_client: TestClient):
    r = production_test_client.post("/predict_batch", json={"records": []})
    assert r.status_code == 422


def test_predict_batch_consistent_with_individual_predict(
    production_test_client: TestClient,
):
    """Batch prediction must equal single-row prediction for the same input."""
    single = production_test_client.post("/predict", json=VALID_PAYLOAD).json()
    batch = production_test_client.post("/predict_batch", json={"records": [VALID_PAYLOAD]}).json()
    assert single["churn_probability"] == pytest.approx(
        batch["predictions"][0]["churn_probability"], abs=1e-9
    )


# --- OpenAPI sanity -----------------------------------------------------


def test_openapi_spec_lists_all_endpoints(production_test_client: TestClient):
    r = production_test_client.get("/openapi.json")
    assert r.status_code == 200
    paths = set(r.json()["paths"].keys())
    assert {"/health", "/model_info", "/predict", "/predict_batch", "/drift-report"}.issubset(paths)
