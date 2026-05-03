"""End-to-end data ingestion: download → load → preprocess → splits.

Three single-responsibility functions composed by :func:`ingest_to_splits`.
Splitting lets monitoring re-use ``load_raw`` (for drift baselines) and lets
tests target each stage in isolation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn.data.download import download_telco
from churn.data.schema import PROCESSED_SCHEMA, RAW_SCHEMA, TARGET_COLUMN
from churn.data.splits import Splits, make_splits


def load_raw(path: Path) -> pd.DataFrame:
    """Read the Telco CSV from ``path`` and validate against :data:`RAW_SCHEMA`.

    ``TotalCharges`` is read as string and then coerced with ``errors="coerce"``
    so the source CSV's whitespace cells (tenure-0 customers) become NaN
    rather than raising or silently casting to 0.
    """
    df = pd.read_csv(path, dtype={"TotalCharges": "string"})
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return RAW_SCHEMA.validate(df, lazy=True)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a raw-validated dataframe into a modeling-ready one.

    Operations:
      * Drop ``customerID`` (PII / non-feature).
      * Fill ``TotalCharges`` NaN with 0.0 — these correspond to tenure-0
        brand-new customers who have not been billed yet, so 0 is correct
        domain semantics rather than a statistical imputation.
      * Encode ``Churn`` (Yes/No) as ``churn`` ∈ {0, 1} ``int64`` (explicit
        dtype: Python ``int`` maps to int32 on Windows numpy, which the
        schema would reject).

    The returned frame is validated against :data:`PROCESSED_SCHEMA`.
    """
    out = df.drop(columns=["customerID"]).copy()
    out["TotalCharges"] = out["TotalCharges"].fillna(0.0).astype(float)
    out[TARGET_COLUMN] = (out["Churn"] == "Yes").astype("int64")
    out = out.drop(columns=["Churn"])
    return PROCESSED_SCHEMA.validate(out, lazy=True)


def ingest_to_splits(
    raw_path: Path | None = None,
    test_size: float = 0.20,
    val_size: float = 0.20,
    random_state: int = 42,
) -> Splits:
    """Top-level pipeline: ensure the dataset is present, load, preprocess, split.

    If ``raw_path`` is None, the dataset is downloaded (or reused) under
    ``data/raw/`` per :func:`churn.data.download.download_telco`.
    """
    path = raw_path if raw_path is not None else download_telco()
    raw = load_raw(path)
    processed = preprocess(raw)
    return make_splits(
        processed,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )
