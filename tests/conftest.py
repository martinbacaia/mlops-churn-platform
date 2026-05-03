"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TELCO_SAMPLE_CSV = FIXTURES_DIR / "telco_sample.csv"


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
