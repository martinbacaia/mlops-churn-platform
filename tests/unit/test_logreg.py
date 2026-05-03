from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from churn.models.logreg import LogRegModel


def _toy_dataset(n: int = 200, n_features: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    # Linear ground truth so logreg can learn it cleanly.
    coef = rng.normal(size=n_features)
    logits = X @ coef
    y = (logits > 0).astype(np.int64)
    return X, y


def test_fit_returns_self_and_sets_classes():
    X, y = _toy_dataset()
    model = LogRegModel().fit(X, y)
    np.testing.assert_array_equal(model.classes_, np.array([0, 1]))
    assert model.n_features_in_ == X.shape[1]


def test_predict_proba_shape_and_range():
    X, y = _toy_dataset()
    model = LogRegModel().fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert (proba >= 0).all() and (proba <= 1).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-7)


def test_predict_returns_int_in_binary():
    X, y = _toy_dataset()
    model = LogRegModel().fit(X, y)
    pred = model.predict(X)
    assert pred.dtype == np.int64
    assert set(pred.tolist()).issubset({0, 1})


def test_unfitted_predict_proba_raises():
    model = LogRegModel()
    with pytest.raises(NotFittedError):
        model.predict_proba(np.zeros((2, 8)))


def test_deterministic_for_same_seed():
    X, y = _toy_dataset()
    a = LogRegModel(random_state=42).fit(X, y).predict_proba(X)
    b = LogRegModel(random_state=42).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(a, b)


def test_name_is_logreg():
    assert LogRegModel.name == "logreg"


def test_get_params_round_trip_via_sklearn_clone():
    """sklearn's clone() relies on get_params/set_params; ensures Optuna integration."""
    from sklearn.base import clone

    original = LogRegModel(C=0.3, max_iter=500, random_state=7, class_weight=None)
    cloned = clone(original)
    assert cloned.get_params() == original.get_params()
    assert cloned is not original
