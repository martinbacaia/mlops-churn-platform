"""Drift-endpoint tests using the session-scoped Production fixture."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import mlflow
import pytest
from fastapi.testclient import TestClient

from churn.data.ingest import load_raw
from churn.serving.app import create_app

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "telco_sample.csv"


def _payload_records(n: int = 5):
    df = load_raw(FIXTURE_PATH).iloc[:n]
    df = df.drop(columns=["customerID", "Churn"])
    return df.to_dict(orient="records")


def test_drift_report_endpoint_returns_canonical_schema(
    production_test_client: TestClient,
):
    r = production_test_client.post("/drift-report", json={"records": _payload_records(5)})
    assert r.status_code == 200
    body = r.json()
    assert "feature_drift" in body
    assert "summary" in body
    assert "prediction_drift" in body
    assert body["psi_threshold"] > 0


def test_drift_report_response_is_json_serializable(
    production_test_client: TestClient,
):
    """Categorical features have NaN KS statistics; make sure these serialize cleanly."""
    r = production_test_client.post("/drift-report", json={"records": _payload_records(10)})
    body = r.json()
    cats = [f for f in body["feature_drift"] if f["type"] == "categorical"]
    assert cats, "Expected at least one categorical feature in the report"
    assert cats[0]["ks_statistic"] is None
    assert cats[0]["ks_pvalue"] is None


def test_drift_report_rejects_empty_records(production_test_client: TestClient):
    r = production_test_client.post("/drift-report", json={"records": []})
    assert r.status_code == 422


@pytest.fixture
def client_without_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, production_env
) -> Iterator[TestClient]:
    """Production model exists but baseline CSV is missing → /drift-report returns 503."""
    # Override DATA_DIR to a fresh tmp path so the baseline CSV is absent.
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    mlflow.set_tracking_uri(production_env.tracking_uri)
    app = create_app()
    with TestClient(app) as cl:
        yield cl


def test_drift_report_503_when_baseline_csv_missing(
    client_without_baseline: TestClient,
):
    r = client_without_baseline.post("/drift-report", json={"records": _payload_records(3)})
    assert r.status_code == 503
    assert "Baseline dataset not found" in r.json()["detail"]
