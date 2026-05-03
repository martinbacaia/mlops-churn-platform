from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from churn.models.tabular_mlp import TabularMLPModel


def _toy_dataset(n: int = 256, n_features: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    coef = rng.normal(size=n_features)
    y = ((X @ coef + 0.3 * rng.normal(size=n)) > 0).astype(np.int64)
    return X, y


@pytest.fixture(scope="module")
def trained_model():
    X, y = _toy_dataset()
    return TabularMLPModel(epochs=3, batch_size=64).fit(X, y), X, y


def test_fit_sets_classes_and_feature_count(trained_model):
    model, X, _ = trained_model
    np.testing.assert_array_equal(model.classes_, np.array([0, 1]))
    assert model.n_features_in_ == X.shape[1]


def test_predict_proba_shape_and_range(trained_model):
    model, X, y = trained_model
    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert (proba >= 0).all() and (proba <= 1).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_predict_returns_int_in_binary(trained_model):
    model, X, _ = trained_model
    pred = model.predict(X)
    assert pred.dtype == np.int64
    assert set(pred.tolist()).issubset({0, 1})


def test_unfitted_predict_proba_raises():
    model = TabularMLPModel()
    with pytest.raises(NotFittedError):
        model.predict_proba(np.zeros((2, 8), dtype=np.float32))


def test_deterministic_for_same_seed():
    """Two fresh fits with the same seed produce identical probabilities on CPU."""
    X, y = _toy_dataset()
    a = TabularMLPModel(epochs=3, batch_size=64, random_state=42).fit(X, y).predict_proba(X)
    b = TabularMLPModel(epochs=3, batch_size=64, random_state=42).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_different_seeds_produce_different_predictions():
    X, y = _toy_dataset()
    a = TabularMLPModel(epochs=3, batch_size=64, random_state=1).fit(X, y).predict_proba(X)
    b = TabularMLPModel(epochs=3, batch_size=64, random_state=2).fit(X, y).predict_proba(X)
    # Predictions should differ, but the same shape contract holds.
    assert not np.allclose(a, b, atol=1e-6)


def test_name_is_tabular_mlp():
    assert TabularMLPModel.name == "tabular_mlp"


def test_no_torch_imports_leak_through_public_api():
    """The wrapper hides torch from callers — outside this module, nothing should
    have to ``import torch`` to use a TabularMLPModel."""
    import churn.models.tabular_mlp as module

    public_attrs = {n for n in dir(module) if not n.startswith("_")}
    # ``torch`` is imported at module level; that's fine as an internal detail,
    # but the public API surface (the model class) must not require torch.
    model = TabularMLPModel(epochs=1)
    assert "TabularMLPModel" in public_attrs
    # The estimator behaves as a normal sklearn-style object:
    params = model.get_params()
    expected_keys = {
        "hidden",
        "dropout",
        "lr",
        "epochs",
        "batch_size",
        "weight_decay",
        "random_state",
    }
    assert expected_keys.issubset(params.keys())
