from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from churn.data.ingest import load_raw, preprocess
from churn.data.splits import make_splits
from churn.evaluation.compare import COMPARE_COLUMNS, compare_models
from churn.features.pipeline import build_feature_pipeline
from churn.models.logreg import LogRegModel
from churn.models.tabular_mlp import TabularMLPModel
from churn.models.xgboost_model import XGBoostModel

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "telco_sample.csv"


@pytest.fixture(scope="module")
def fitted_models_and_holdout():
    """Train cheap versions of all three models on the fixture; return on test split."""
    df = preprocess(load_raw(FIXTURE_PATH))
    splits = make_splits(df, random_state=42)
    pipeline = build_feature_pipeline().fit(splits.X_train)

    X_train_t = pipeline.transform(splits.X_train).astype(np.float64)
    X_test_t = pipeline.transform(splits.X_test).astype(np.float64)
    y_train = splits.y_train.to_numpy().astype(np.int64)
    y_test = splits.y_test.to_numpy().astype(np.int64)

    models = {
        "logreg": LogRegModel(max_iter=200).fit(X_train_t, y_train),
        "xgboost": XGBoostModel(n_estimators=20).fit(X_train_t, y_train),
        "tabular_mlp": TabularMLPModel(epochs=2, batch_size=64).fit(X_train_t, y_train),
    }
    return models, X_test_t, y_test


def test_compare_returns_one_row_per_model(fitted_models_and_holdout):
    models, X, y = fitted_models_and_holdout
    out = compare_models(models, X, y, n_latency_samples=10)
    assert list(out.index) == ["logreg", "xgboost", "tabular_mlp"]


def test_compare_columns_match_canonical_set(fitted_models_and_holdout):
    models, X, y = fitted_models_and_holdout
    out = compare_models(models, X, y, n_latency_samples=10)
    assert list(out.columns) == COMPARE_COLUMNS


def test_quality_metrics_are_finite_and_bounded(fitted_models_and_holdout):
    models, X, y = fitted_models_and_holdout
    out = compare_models(models, X, y, n_latency_samples=10)
    for metric in ("roc_auc", "pr_auc", "f1", "brier"):
        assert (out[metric] >= 0.0).all()
        assert (out[metric] <= 1.0).all()
    assert (out["log_loss"] >= 0.0).all()


def test_latency_percentiles_are_monotonically_increasing(fitted_models_and_holdout):
    models, X, y = fitted_models_and_holdout
    out = compare_models(models, X, y, n_latency_samples=20)
    # p50 <= p95 <= p99 by definition of order statistics.
    assert (out["p50_ms"] <= out["p95_ms"]).all()
    assert (out["p95_ms"] <= out["p99_ms"]).all()
    # All latencies positive (we measured something).
    assert (out["p50_ms"] > 0).all()


def test_size_kb_is_positive_for_all_models(fitted_models_and_holdout):
    models, X, y = fitted_models_and_holdout
    out = compare_models(models, X, y, n_latency_samples=5)
    assert (out["size_kb"] > 0).all()


def test_empty_dict_raises():
    with pytest.raises(ValueError, match="at least one"):
        compare_models({}, np.zeros((4, 4)), np.array([0, 1, 0, 1]))


def test_works_with_subset_of_registry(fitted_models_and_holdout):
    """Comparison helper accepts any subset — useful for ablations / champion vs challenger."""
    models, X, y = fitted_models_and_holdout
    subset = {"logreg": models["logreg"], "xgboost": models["xgboost"]}
    out = compare_models(subset, X, y, n_latency_samples=5)
    assert len(out) == 2
    assert set(out.index) == {"logreg", "xgboost"}
