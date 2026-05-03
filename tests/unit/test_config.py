from __future__ import annotations

import pytest

from churn.config import Settings


def test_defaults_match_env_example():
    s = Settings()
    assert s.mlflow_tracking_uri == "sqlite:///mlruns/mlflow.db"
    assert s.mlflow_experiment_name == "churn"
    assert s.model_name == "churn_classifier"
    assert s.model_stage == "Production"
    assert s.log_level == "INFO"
    assert s.log_format == "json"
    assert s.random_state == 42


def test_env_overrides(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODEL_STAGE", "Staging")
    monkeypatch.setenv("RANDOM_STATE", "7")
    monkeypatch.setenv("LOG_FORMAT", "console")
    s = Settings()
    assert s.model_stage == "Staging"
    assert s.random_state == 7
    assert s.log_format == "console"


def test_invalid_literal_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODEL_STAGE", "Bogus")
    with pytest.raises(ValueError):
        Settings()
