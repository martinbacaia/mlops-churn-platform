"""Typed application settings, loaded from environment / .env file.

Settings are the single source of truth for *configurable* behavior. Code that
needs MLflow URIs, model names, or paths reads them from here — never from
ad-hoc os.environ lookups scattered across modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        # Pydantic v2 reserves the ``model_`` prefix for its own API. Disable that
        # guard here — ``model_name`` / ``model_stage`` are domain vocabulary
        # (registered ML model in MLflow), not framework hooks.
        protected_namespaces=(),
    )

    # MLflow
    mlflow_tracking_uri: str = "sqlite:///mlruns/mlflow.db"
    mlflow_experiment_name: str = "churn"

    # Model registry — serving loads `models:/{model_name}/{model_stage}`.
    model_name: str = "churn_classifier"
    model_stage: Literal["Production", "Staging", "Archived", "None"] = "Production"

    # Data
    data_dir: Path = Field(default=PROJECT_ROOT / "data")

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "console"] = "json"

    # Reproducibility — propagated to splits, CV, samplers, model RNGs.
    random_state: int = 42


def get_settings() -> Settings:
    """Construct a fresh Settings instance.

    Not memoized: tests rely on monkeypatching env vars between calls. Callers
    that want a stable reference should hold one themselves.
    """
    return Settings()
