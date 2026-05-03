from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from churn.data.ingest import load_raw, preprocess
from churn.data.schema import TARGET_COLUMN
from churn.features.pipeline import (
    CATEGORICAL_COLUMNS,
    FEATURE_PIPELINE_VERSION,
    NUMERICAL_COLUMNS,
    build_feature_pipeline,
    load_pipeline,
    save_pipeline,
)


@pytest.fixture
def fixture_csv_path() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures" / "telco_sample.csv"


@pytest.fixture
def processed_df(fixture_csv_path: Path) -> pd.DataFrame:
    return preprocess(load_raw(fixture_csv_path))


def test_build_returns_fresh_unfitted_pipeline():
    pipe = build_feature_pipeline()
    assert isinstance(pipe, ColumnTransformer)
    # An unfitted ColumnTransformer has not yet populated ``transformers_``
    # (the post-fit attribute holding actual fitted estimators).
    assert not hasattr(pipe, "transformers_")


def test_pipeline_fits_and_transforms_on_processed_data(processed_df: pd.DataFrame):
    pipe = build_feature_pipeline()
    X = processed_df.drop(columns=[TARGET_COLUMN])
    out = pipe.fit_transform(X)
    assert out.ndim == 2
    assert out.shape[0] == len(processed_df)
    # Output is dense float (sparse_threshold=0.0).
    assert np.issubdtype(out.dtype, np.floating)


def test_output_column_count_matches_branch_contributions(processed_df: pd.DataFrame):
    pipe = build_feature_pipeline()
    X = processed_df.drop(columns=[TARGET_COLUMN])
    out = pipe.fit_transform(X)

    n_numerical = len(NUMERICAL_COLUMNS)
    # Tenure-bucket branch: number of distinct buckets observed in fit data.
    bucket_branch = pipe.named_transformers_["tenure_bucket"]
    n_bucket = bucket_branch.named_steps["encode"].categories_[0].shape[0]
    # Categorical branch: sum of unique values per column observed in fit data.
    cat_branch = pipe.named_transformers_["categorical"]
    n_categorical = sum(c.shape[0] for c in cat_branch.categories_)

    assert out.shape[1] == n_numerical + n_bucket + n_categorical


def test_get_feature_names_out_is_callable_after_fit(processed_df: pd.DataFrame):
    pipe = build_feature_pipeline()
    X = processed_df.drop(columns=[TARGET_COLUMN])
    pipe.fit(X)
    names = pipe.get_feature_names_out()
    assert len(names) == pipe.transform(X).shape[1]
    # Verbose names are prefixed with the branch name.
    assert any(n.startswith("numerical__") for n in names)
    assert any(n.startswith("categorical__") for n in names)
    assert any(n.startswith("tenure_bucket__") for n in names)


def test_transform_is_deterministic_run_to_run(processed_df: pd.DataFrame):
    X = processed_df.drop(columns=[TARGET_COLUMN])

    a = build_feature_pipeline().fit_transform(X)
    b = build_feature_pipeline().fit_transform(X)
    np.testing.assert_array_equal(a, b)


def test_transform_handles_unseen_category_via_ignore(processed_df: pd.DataFrame):
    X_train = processed_df.drop(columns=[TARGET_COLUMN])
    pipe = build_feature_pipeline().fit(X_train)

    X_inference = X_train.iloc[:5].copy()
    X_inference["PaymentMethod"] = "Bitcoin (autonomous)"  # genuinely unseen

    out = pipe.transform(X_inference)
    assert out.shape == (5, pipe.transform(X_train.iloc[:5]).shape[1])
    # The unseen category zero-encodes the four PaymentMethod columns.
    names = pipe.get_feature_names_out()
    pm_idx = [i for i, n in enumerate(names) if "PaymentMethod" in n]
    assert (out[:, pm_idx] == 0).all()


def test_save_and_load_roundtrip_preserves_transform(processed_df: pd.DataFrame, tmp_path: Path):
    X = processed_df.drop(columns=[TARGET_COLUMN])
    pipe = build_feature_pipeline().fit(X)
    expected = pipe.transform(X)

    path = tmp_path / "feature_pipeline.joblib"
    saved_path = save_pipeline(pipe, path)
    assert saved_path.exists()

    loaded = load_pipeline(path)
    actual = loaded.transform(X)
    np.testing.assert_array_equal(expected, actual)


def test_load_pipeline_rejects_non_pipeline_artifact(tmp_path: Path):
    import joblib

    path = tmp_path / "bogus.joblib"
    joblib.dump({"not": "a pipeline"}, path)
    with pytest.raises(TypeError, match="ColumnTransformer"):
        load_pipeline(path)


def test_feature_pipeline_version_is_set():
    assert FEATURE_PIPELINE_VERSION == "v1"


def test_required_columns_are_disjoint():
    """Numerical and categorical lists must not double-count a column."""
    overlap = set(NUMERICAL_COLUMNS) & set(CATEGORICAL_COLUMNS)
    assert overlap == set()
