"""Shared pytest fixtures.

Module-1 fixtures only; expanded as data / features / models land.
"""

from __future__ import annotations

import pytest


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
