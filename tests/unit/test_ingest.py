from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pandera as pa
import pytest

from churn.data.ingest import ingest_to_splits, load_raw, preprocess
from churn.data.schema import TARGET_COLUMN


@pytest.fixture
def fixture_csv_path() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures" / "telco_sample.csv"


def test_load_raw_returns_validated_dataframe(fixture_csv_path: Path) -> None:
    df = load_raw(fixture_csv_path)
    assert "customerID" in df.columns
    assert "Churn" in df.columns
    assert df["TotalCharges"].dtype == float


def test_load_raw_handles_whitespace_total_charges(tmp_path: Path, fixture_csv_path: Path) -> None:
    """Inject a whitespace TotalCharges row mirroring the source data quirk."""
    df = pd.read_csv(fixture_csv_path, dtype={"TotalCharges": "string"})
    df.loc[0, "tenure"] = 0
    df.loc[0, "TotalCharges"] = " "
    out_path = tmp_path / "telco_with_quirk.csv"
    df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)

    loaded = load_raw(out_path)
    assert pd.isna(loaded.loc[0, "TotalCharges"])


def test_load_raw_rejects_corrupt_data(tmp_path: Path, fixture_csv_path: Path) -> None:
    df = pd.read_csv(fixture_csv_path)
    df.loc[0, "gender"] = "Other"
    out_path = tmp_path / "corrupt.csv"
    df.to_csv(out_path, index=False)
    # ``lazy=True`` collects every error and raises ``SchemaErrors`` (plural).
    with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
        load_raw(out_path)


def test_preprocess_drops_customer_id(fixture_csv_path: Path) -> None:
    raw = load_raw(fixture_csv_path)
    processed = preprocess(raw)
    assert "customerID" not in processed.columns


def test_preprocess_encodes_churn_as_binary_int64(fixture_csv_path: Path) -> None:
    raw = load_raw(fixture_csv_path)
    processed = preprocess(raw)
    assert processed[TARGET_COLUMN].dtype == np.int64
    assert set(processed[TARGET_COLUMN].unique()).issubset({0, 1})
    # Mapping is Yes -> 1; check by intersecting with raw.
    yes_mask = (raw["Churn"] == "Yes").to_numpy()
    np.testing.assert_array_equal(processed[TARGET_COLUMN].to_numpy(), yes_mask.astype("int64"))


def test_preprocess_fills_total_charges_nan_with_zero(
    tmp_path: Path, fixture_csv_path: Path
) -> None:
    df = pd.read_csv(fixture_csv_path, dtype={"TotalCharges": "string"})
    df.loc[0, "tenure"] = 0
    df.loc[0, "TotalCharges"] = " "
    out_path = tmp_path / "with_nan.csv"
    df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)

    raw = load_raw(out_path)
    processed = preprocess(raw)
    assert processed.loc[0, "TotalCharges"] == 0.0
    assert processed["TotalCharges"].isna().sum() == 0


def test_ingest_to_splits_uses_provided_path(fixture_csv_path: Path) -> None:
    splits = ingest_to_splits(raw_path=fixture_csv_path, random_state=42)
    total = splits.sizes["train"] + splits.sizes["val"] + splits.sizes["test"]
    assert total == 200
    assert TARGET_COLUMN not in splits.X_train.columns


def test_ingest_to_splits_is_deterministic(fixture_csv_path: Path) -> None:
    a = ingest_to_splits(raw_path=fixture_csv_path, random_state=42)
    b = ingest_to_splits(raw_path=fixture_csv_path, random_state=42)
    pd.testing.assert_frame_equal(a.X_train, b.X_train)
    pd.testing.assert_series_equal(a.y_test, b.y_test)
