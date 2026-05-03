"""Shared pytest fixtures.

Two layers of fixtures:

* **Function-scoped** (default): the autouse ``_isolate_settings_env`` clears
  env vars that would override Settings defaults during any test. Tests that
  need a custom environment opt in via ``monkeypatch.setenv`` themselves.
* **Session-scoped**: ``session_production_state`` runs the full
  train → register → promote pipeline **once per pytest session** for tests
  that need a real Production model. Per-test setup time drops from ~7 s to
  the cost of ``create_app`` + ``mlflow.sklearn.load_model`` (~2 s).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import mlflow
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from churn.data.ingest import load_raw, preprocess
from churn.data.splits import Splits, make_splits
from churn.features.pipeline import build_feature_pipeline
from churn.models.logreg import LogRegModel
from churn.serving.app import create_app
from churn.training.promote import promote_version
from churn.training.train import train_all_models

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TELCO_SAMPLE_CSV = FIXTURES_DIR / "telco_sample.csv"
SESSION_REGISTRY_NAME = "test_session_classifier"

_SESSION_ENV_VARS = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME",
    "MODEL_NAME",
    "DATA_DIR",
)


@pytest.fixture
def telco_sample_df() -> pd.DataFrame:
    """A 200-row stratified slice of the real Telco CSV. Cheap to load, real schema."""
    return pd.read_csv(TELCO_SAMPLE_CSV)


@pytest.fixture(autouse=True)
def _isolate_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip env vars that would override Settings defaults during a test.

    Without this, a developer running tests with a populated .env or shell
    environment would silently change defaults under the test (e.g. RANDOM_STATE=7).
    Tests that *want* an override use monkeypatch explicitly.
    """
    for var in (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT_NAME",
        "MODEL_NAME",
        "MODEL_STAGE",
        "DATA_DIR",
        "LOG_LEVEL",
        "LOG_FORMAT",
        "RANDOM_STATE",
    ):
        monkeypatch.delenv(var, raising=False)


# --- Session-scoped Production setup --------------------------------------


@dataclass(frozen=True)
class SessionProductionState:
    """State produced once per session by :func:`session_production_state`."""

    tracking_uri: str
    experiment_name: str
    registry_name: str
    data_dir: Path
    splits: Splits
    pipeline: object  # ColumnTransformer; avoid the import cost in conftest


@pytest.fixture(scope="session")
def session_production_state(
    tmp_path_factory: pytest.TempPathFactory,
) -> SessionProductionState:
    """Train + register + promote one LogReg, once for the whole test session.

    Tests that need a Production model pull the function-scoped
    :func:`production_env` (which re-applies the env vars / tracking URI) or
    :func:`production_test_client` (which also yields a configured TestClient).
    The trained model + SQLite registry persist across tests.
    """
    tmp = tmp_path_factory.mktemp("session_mlflow")
    db = tmp / "mlruns.db"
    tracking_uri = f"sqlite:///{db}"
    experiment_name = "test_session"

    # Set env vars only for the duration of the setup. The ``_isolate_settings_env``
    # autouse fixture deletes them between tests, so per-test usage re-sets them
    # via the ``production_env`` fixture below.
    import os

    saved = {k: os.environ.get(k) for k in _SESSION_ENV_VARS}
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
    os.environ["MODEL_NAME"] = SESSION_REGISTRY_NAME
    os.environ["DATA_DIR"] = str(tmp)
    mlflow.set_tracking_uri(tracking_uri)

    try:
        df = preprocess(load_raw(TELCO_SAMPLE_CSV))
        splits = make_splits(df, random_state=42)
        pipeline = build_feature_pipeline().fit(splits.X_train)

        with patch("churn.training.train.MODEL_REGISTRY", {"logreg": LogRegModel}):
            train_all_models(
                splits=splits,
                feature_pipeline=pipeline,
                dataset_md5="session-md5",
                register=True,
            )

        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name = '{SESSION_REGISTRY_NAME}'")
        version = str(max(versions, key=lambda v: int(v.version)).version)
        promote_version(version=version, model_name=SESSION_REGISTRY_NAME)

        # Place a baseline CSV where the /drift-report endpoint expects it.
        baseline_dir = tmp / "raw"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        load_raw(TELCO_SAMPLE_CSV).to_csv(baseline_dir / "telco.csv", index=False)
    finally:
        # Don't leak env to tests that didn't request this fixture.
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    return SessionProductionState(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        registry_name=SESSION_REGISTRY_NAME,
        data_dir=tmp,
        splits=splits,
        pipeline=pipeline,
    )


@pytest.fixture
def production_env(
    session_production_state: SessionProductionState,
    monkeypatch: pytest.MonkeyPatch,
) -> SessionProductionState:
    """Re-apply the session's MLflow + DATA_DIR env vars for the current test."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", session_production_state.tracking_uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", session_production_state.experiment_name)
    monkeypatch.setenv("MODEL_NAME", session_production_state.registry_name)
    monkeypatch.setenv("DATA_DIR", str(session_production_state.data_dir))
    mlflow.set_tracking_uri(session_production_state.tracking_uri)
    return session_production_state


@pytest.fixture
def production_test_client(
    production_env: SessionProductionState,
) -> Iterator[TestClient]:
    """A TestClient backed by the session's Production model."""
    app = create_app()
    with TestClient(app) as client:
        yield client
