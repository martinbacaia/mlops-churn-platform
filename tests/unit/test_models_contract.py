"""Parametrized tests that the **same** contract holds for every implementation.

The point of these tests is the platform claim: any of the three models can sit
behind the registry without the rest of the system noticing. If a future
implementation breaks any of these invariants, the platform layer is broken
even if the model itself trains and predicts in isolation.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np
import numpy.typing as npt
import pytest

from churn.data.ingest import load_raw, preprocess
from churn.data.schema import TARGET_COLUMN
from churn.features.pipeline import build_feature_pipeline
from churn.models.base import Model
from churn.models.logreg import LogRegModel
from churn.models.registry import MODEL_REGISTRY
from churn.models.tabular_mlp import TabularMLPModel
from churn.models.xgboost_model import XGBoostModel

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "telco_sample.csv"


@pytest.fixture(scope="module")
def feature_matrix() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Real post-feature-pipeline output on the 200-row fixture."""
    df = preprocess(load_raw(FIXTURE_PATH))
    X_raw = df.drop(columns=[TARGET_COLUMN])
    pipe = build_feature_pipeline().fit(X_raw)
    X = pipe.transform(X_raw)
    y = df[TARGET_COLUMN].to_numpy()
    return X, y


# Factories keep epoch / n_estimator counts low so the parametrized suite stays fast.
ModelFactory = Callable[[], Model]

MODEL_FACTORIES = [
    pytest.param(lambda: LogRegModel(max_iter=200), id="logreg"),
    pytest.param(lambda: XGBoostModel(n_estimators=30), id="xgboost"),
    pytest.param(lambda: TabularMLPModel(epochs=3, batch_size=64), id="tabular_mlp"),
]


@pytest.fixture(params=MODEL_FACTORIES)
def make_model(request: pytest.FixtureRequest) -> ModelFactory:
    return request.param


def test_fit_returns_self(make_model: ModelFactory, feature_matrix):
    X, y = feature_matrix
    m = make_model()
    assert m.fit(X, y) is m


def test_predict_proba_shape_is_n_by_2(make_model: ModelFactory, feature_matrix):
    X, y = feature_matrix
    proba = make_model().fit(X, y).predict_proba(X)
    assert proba.shape == (len(y), 2)


def test_predict_proba_values_in_unit_interval(make_model: ModelFactory, feature_matrix):
    X, y = feature_matrix
    proba = make_model().fit(X, y).predict_proba(X)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_predict_proba_rows_sum_to_one(make_model: ModelFactory, feature_matrix):
    X, y = feature_matrix
    proba = make_model().fit(X, y).predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_predict_returns_int64_in_binary(make_model: ModelFactory, feature_matrix):
    X, y = feature_matrix
    pred = make_model().fit(X, y).predict(X)
    assert pred.dtype == np.int64
    assert set(pred.tolist()).issubset({0, 1})


def test_classes_attribute_after_fit(make_model: ModelFactory, feature_matrix):
    X, y = feature_matrix
    m = make_model().fit(X, y)
    np.testing.assert_array_equal(m.classes_, np.array([0, 1]))


def test_n_features_in_matches_input(make_model: ModelFactory, feature_matrix):
    X, y = feature_matrix
    m = make_model().fit(X, y)
    assert m.n_features_in_ == X.shape[1]


def test_deterministic_given_seed(make_model: ModelFactory, feature_matrix):
    X, y = feature_matrix
    a = make_model().fit(X, y).predict_proba(X)
    b = make_model().fit(X, y).predict_proba(X)
    # XGBoost and LogReg are bit-exact; TabularMLP is bit-exact on CPU
    # thanks to ``set_torch_deterministic`` inside ``fit``.
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_joblib_roundtrip_preserves_predictions(
    make_model: ModelFactory, feature_matrix, tmp_path: Path
):
    X, y = feature_matrix
    m = make_model().fit(X, y)
    expected = m.predict_proba(X)

    path = tmp_path / "model.joblib"
    joblib.dump(m, path)
    loaded = joblib.load(path)
    actual = loaded.predict_proba(X)
    np.testing.assert_allclose(expected, actual, atol=1e-6)


def test_get_params_contains_random_state(make_model: ModelFactory):
    """All three models accept ``random_state`` so the platform can pin determinism uniformly."""
    m = make_model()
    assert "random_state" in m.get_params()


# --- Registry-level invariants ------------------------------------------------


def test_registry_has_all_three_models():
    assert set(MODEL_REGISTRY.keys()) == {"logreg", "xgboost", "tabular_mlp"}


def test_registry_keys_match_class_name_attr():
    for key, cls in MODEL_REGISTRY.items():
        assert cls.name == key, f"registry key {key!r} != cls.name {cls.name!r}"


def test_registry_classes_are_model_subclasses():
    for cls in MODEL_REGISTRY.values():
        assert issubclass(cls, Model)
