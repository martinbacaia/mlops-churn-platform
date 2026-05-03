from __future__ import annotations

import numpy as np
import pandas as pd
import pandera as pa
import pytest

from churn.data.schema import PROCESSED_SCHEMA, RAW_SCHEMA, TARGET_COLUMN


def test_raw_schema_accepts_clean_fixture(telco_sample_df: pd.DataFrame) -> None:
    RAW_SCHEMA.validate(telco_sample_df)


def test_raw_schema_allows_nan_total_charges(telco_sample_df: pd.DataFrame) -> None:
    """Brand-new customers (tenure=0) have whitespace TotalCharges in the source CSV;
    the loader coerces those to NaN. The schema must accept that."""
    df = telco_sample_df.copy()
    df.loc[0, "TotalCharges"] = np.nan
    RAW_SCHEMA.validate(df)  # nullable=True for TotalCharges


def test_raw_schema_rejects_unknown_gender(telco_sample_df: pd.DataFrame) -> None:
    df = telco_sample_df.copy()
    df.loc[0, "gender"] = "Other"
    with pytest.raises(pa.errors.SchemaError):
        RAW_SCHEMA.validate(df)


def test_raw_schema_rejects_extra_column(telco_sample_df: pd.DataFrame) -> None:
    df = telco_sample_df.copy()
    df["unexpected"] = 1
    with pytest.raises(pa.errors.SchemaError):
        RAW_SCHEMA.validate(df)


def test_raw_schema_rejects_negative_tenure(telco_sample_df: pd.DataFrame) -> None:
    df = telco_sample_df.copy()
    df.loc[0, "tenure"] = -1
    with pytest.raises(pa.errors.SchemaError):
        RAW_SCHEMA.validate(df)


def test_raw_schema_rejects_duplicate_customer_ids(telco_sample_df: pd.DataFrame) -> None:
    df = telco_sample_df.copy()
    df.loc[1, "customerID"] = df.loc[0, "customerID"]
    with pytest.raises(pa.errors.SchemaError):
        RAW_SCHEMA.validate(df)


def _processed_from(raw: pd.DataFrame) -> pd.DataFrame:
    """Helper: minimal processing equivalent to the ingest-layer logic.

    ``astype("int64")`` is explicit because Python's ``int`` maps to int32 on
    Windows numpy builds, which would fail the schema's int64 expectation.
    """
    df = raw.drop(columns=["customerID"]).copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df[TARGET_COLUMN] = (df["Churn"] == "Yes").astype("int64")
    return df.drop(columns=["Churn"])


def test_processed_schema_accepts_processed_fixture(telco_sample_df: pd.DataFrame) -> None:
    processed = _processed_from(telco_sample_df)
    PROCESSED_SCHEMA.validate(processed)


def test_processed_schema_rejects_target_outside_binary(telco_sample_df: pd.DataFrame) -> None:
    processed = _processed_from(telco_sample_df)
    processed.loc[0, TARGET_COLUMN] = 2
    with pytest.raises(pa.errors.SchemaError):
        PROCESSED_SCHEMA.validate(processed)


def test_processed_schema_rejects_remaining_customer_id(telco_sample_df: pd.DataFrame) -> None:
    processed = _processed_from(telco_sample_df)
    processed["customerID"] = "X"
    with pytest.raises(pa.errors.SchemaError):
        PROCESSED_SCHEMA.validate(processed)
