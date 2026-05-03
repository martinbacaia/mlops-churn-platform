from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# scripts/ is not a package; add it to sys.path so the test can import.
SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from inject_drift import inject_drift  # type: ignore[import-not-found]  # noqa: E402

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "telco_sample.csv"


def test_inject_drift_returns_dataframe_of_same_length():
    df = pd.read_csv(FIXTURE_PATH)
    out = inject_drift(df, seed=0)
    assert len(out) == len(df)
    assert set(out.columns) == set(df.columns)


def test_inject_drift_actually_changes_monthly_charges():
    df = pd.read_csv(FIXTURE_PATH)
    out = inject_drift(df, seed=0)
    # Distribution should shift up by ~25%.
    assert out["MonthlyCharges"].mean() > df["MonthlyCharges"].mean() * 1.20


def test_inject_drift_skews_contract_toward_month_to_month():
    df = pd.read_csv(FIXTURE_PATH)
    out = inject_drift(df, seed=0)
    mtm_share = (out["Contract"] == "Month-to-month").mean()
    assert mtm_share > 0.7  # designed to be ~0.85


def test_inject_drift_is_deterministic_for_same_seed():
    df = pd.read_csv(FIXTURE_PATH)
    a = inject_drift(df, seed=42)
    b = inject_drift(df, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_inject_drift_does_not_mutate_input():
    df = pd.read_csv(FIXTURE_PATH)
    original_mc_mean = df["MonthlyCharges"].mean()
    inject_drift(df, seed=0)
    assert df["MonthlyCharges"].mean() == original_mc_mean
