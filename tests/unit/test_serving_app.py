from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import mlflow
import pytest
from fastapi.testclient import TestClient

from churn.data.ingest import load_raw, preprocess
from churn.data.splits import make_splits
from churn.features.pipeline import build_feature_pipeline
from churn.models.logreg import LogRegModel
from churn.serving.app import create_app
from churn.training.promote import promote_version
from churn.training.train import train_all_models

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "telco_sample.csv"
REGISTRY_NAME = "test_serving_app_classifier"


@pytest.fixture
def isolated_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    db = tmp_path / "mlruns.db"
    tracking_uri = f"sqlite:///{db}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test_serving_app")
    monkeypatch.setenv("MODEL_NAME", REGISTRY_NAME)
    mlflow.set_tracking_uri(tracking_uri)
    yield tracking_uri


@pytest.fixture
def client_with_production_model(isolated_mlflow: str):
    """Train + register + promote a tiny logreg, then yield a TestClient."""
    df = preprocess(load_raw(FIXTURE_PATH))
    splits = make_splits(df, random_state=42)
    pipeline = build_feature_pipeline().fit(splits.X_train)

    with patch("churn.training.train.MODEL_REGISTRY", {"logreg": LogRegModel}):
        train_all_models(
            splits=splits,
            feature_pipeline=pipeline,
            dataset_md5="test-md5",
            register=True,
        )
    mlflow_client = mlflow.tracking.MlflowClient()
    versions = mlflow_client.search_model_versions(f"name = '{REGISTRY_NAME}'")
    version = str(max(versions, key=lambda v: int(v.version)).version)
    promote_version(version=version, model_name=REGISTRY_NAME)

    app = create_app()
    with TestClient(app) as client:
        yield client


@pytest.fixture
def client_no_production_model(isolated_mlflow: str):
    """No model registered → API starts but reports degraded."""
    app = create_app()
    with TestClient(app) as client:
        yield client


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


# --- /health -------------------------------------------------------------


def test_health_ok_when_model_loaded(client_with_production_model):
    r = client_with_production_model.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_type"] == "logreg"
    assert body["model_version"] is not None


def test_health_degraded_when_no_model(client_no_production_model):
    r = client_no_production_model.get("/health")
    assert r.status_code == 200  # health endpoint itself returns 200 even when degraded
    body = r.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False


# --- /model_info ---------------------------------------------------------


def test_model_info_returns_metadata(client_with_production_model):
    r = client_with_production_model.get("/model_info")
    assert r.status_code == 200
    body = r.json()
    assert body["registered_model_name"] == REGISTRY_NAME
    assert body["model_type"] == "logreg"
    assert body["stage"] == "Production"
    assert body["feature_pipeline_version"] == "v1"
    assert body["run_id"]


def test_model_info_503_when_no_model(client_no_production_model):
    r = client_no_production_model.get("/model_info")
    assert r.status_code == 503


# --- /predict ------------------------------------------------------------


def test_predict_happy_path(client_with_production_model):
    r = client_with_production_model.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert body["threshold"] == 0.5


def test_predict_handles_null_total_charges(client_with_production_model):
    payload = {**VALID_PAYLOAD, "tenure": 0, "TotalCharges": None}
    r = client_with_production_model.post("/predict", json=payload)
    assert r.status_code == 200


def test_predict_rejects_bad_payment_method(client_with_production_model):
    payload = {**VALID_PAYLOAD, "PaymentMethod": "Bitcoin"}
    r = client_with_production_model.post("/predict", json=payload)
    assert r.status_code == 422  # pydantic validation error


def test_predict_rejects_missing_field(client_with_production_model):
    payload = {**VALID_PAYLOAD}
    del payload["Contract"]
    r = client_with_production_model.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_503_when_no_model(client_no_production_model):
    r = client_no_production_model.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 503


# --- /predict_batch ------------------------------------------------------


def test_predict_batch_returns_aligned_predictions(client_with_production_model):
    r = client_with_production_model.post(
        "/predict_batch",
        json={"records": [VALID_PAYLOAD, VALID_PAYLOAD, VALID_PAYLOAD]},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 3
    for p in body["predictions"]:
        assert 0.0 <= p["churn_probability"] <= 1.0


def test_predict_batch_rejects_empty_list(client_with_production_model):
    r = client_with_production_model.post("/predict_batch", json={"records": []})
    assert r.status_code == 422


def test_predict_batch_consistent_with_individual_predict(
    client_with_production_model,
):
    """Batch prediction must equal single-row prediction for the same input."""
    single = client_with_production_model.post("/predict", json=VALID_PAYLOAD).json()
    batch = client_with_production_model.post(
        "/predict_batch", json={"records": [VALID_PAYLOAD]}
    ).json()
    assert single["churn_probability"] == pytest.approx(
        batch["predictions"][0]["churn_probability"], abs=1e-9
    )


# --- OpenAPI sanity -----------------------------------------------------


def test_openapi_spec_lists_all_endpoints(client_with_production_model):
    r = client_with_production_model.get("/openapi.json")
    assert r.status_code == 200
    paths = set(r.json()["paths"].keys())
    assert {"/health", "/model_info", "/predict", "/predict_batch"}.issubset(paths)
