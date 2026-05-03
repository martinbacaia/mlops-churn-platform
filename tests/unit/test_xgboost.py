from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from churn.models.xgboost_model import XGBoostModel


def _toy_dataset(n: int = 300, n_features: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    coef = rng.normal(size=n_features)
    y = ((X @ coef + 0.3 * rng.normal(size=n)) > 0).astype(np.int64)
    return X, y


def test_fit_sets_classes_and_feature_count():
    X, y = _toy_dataset()
    model = XGBoostModel(n_estimators=20).fit(X, y)
    np.testing.assert_array_equal(model.classes_, np.array([0, 1]))
    assert model.n_features_in_ == X.shape[1]


def test_predict_proba_shape_and_range():
    X, y = _toy_dataset()
    model = XGBoostModel(n_estimators=20).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert (proba >= 0).all() and (proba <= 1).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_predict_returns_int_in_binary():
    X, y = _toy_dataset()
    model = XGBoostModel(n_estimators=20).fit(X, y)
    pred = model.predict(X)
    assert pred.dtype == np.int64
    assert set(pred.tolist()).issubset({0, 1})


def test_unfitted_predict_raises():
    model = XGBoostModel()
    with pytest.raises(NotFittedError):
        model.predict_proba(np.zeros((2, 10)))


def test_deterministic_for_same_seed():
    X, y = _toy_dataset()
    a = XGBoostModel(n_estimators=20, random_state=42).fit(X, y).predict_proba(X)
    b = XGBoostModel(n_estimators=20, random_state=42).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(a, b)


def test_class_imbalance_handled_via_scale_pos_weight():
    """A 90/10 imbalanced toy: the model should still learn the minority class."""
    rng = np.random.default_rng(0)
    n = 400
    n_features = 6
    coef = rng.normal(size=n_features)
    X = rng.normal(size=(n, n_features))
    p = 1.0 / (1.0 + np.exp(-X @ coef))
    # Force imbalance by sampling positives at lower rate.
    y = (rng.random(n) < (0.5 * p)).astype(np.int64)

    model = XGBoostModel(n_estimators=50).fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    # The model should have learned a non-trivial separation, not collapse to one class.
    assert proba.std() > 0.05


def test_name_is_xgboost():
    assert XGBoostModel.name == "xgboost"
