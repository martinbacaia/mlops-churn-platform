from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from churn.models.base import Model


class _ToyModel(Model):
    """Stateless mock returning a constant prior — used to exercise the contract."""

    name = "toy"

    def __init__(self, prior: float = 0.5) -> None:
        self.prior = prior

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> _ToyModel:
        self.fitted_ = True
        return self

    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray[np.floating[Any]]:
        n = len(np.asarray(X))
        return np.column_stack([np.full(n, 1.0 - self.prior), np.full(n, self.prior)])


def test_cannot_instantiate_abstract_model():
    with pytest.raises(TypeError):
        Model()


def test_concrete_subclass_satisfies_full_contract():
    m = _ToyModel(prior=0.7)
    X = np.zeros((4, 3))
    y = np.array([0, 1, 0, 1])
    assert m.fit(X, y) is m
    proba = m.predict_proba(X)
    assert proba.shape == (4, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_predict_default_thresholds_at_half():
    m = _ToyModel(prior=0.7).fit(np.zeros((4, 3)), np.array([0, 1, 0, 1]))
    np.testing.assert_array_equal(
        m.predict(np.zeros((4, 3))),
        np.array([1, 1, 1, 1]),
    )

    m_low = _ToyModel(prior=0.3).fit(np.zeros((4, 3)), np.array([0, 1, 0, 1]))
    np.testing.assert_array_equal(
        m_low.predict(np.zeros((4, 3))),
        np.array([0, 0, 0, 0]),
    )


def test_predict_threshold_override_changes_decision():
    m = _ToyModel(prior=0.3).fit(np.zeros((4, 3)), np.array([0, 1, 0, 1]))
    aggressive = m.predict(np.zeros((4, 3)), threshold=0.2)
    np.testing.assert_array_equal(aggressive, np.array([1, 1, 1, 1]))


def test_predict_validates_proba_shape_returned_by_subclass():
    class _BadShapeModel(Model):
        name = "bad"

        def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> _BadShapeModel:
            return self

        def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray[np.floating[Any]]:
            return np.zeros(len(np.asarray(X)))  # wrong: 1-D

    m = _BadShapeModel().fit(np.zeros((4, 3)), np.array([0, 1, 0, 1]))
    with pytest.raises(ValueError, match="shape"):
        m.predict(np.zeros((4, 3)))


def test_subclass_must_define_name_class_attr():
    """Forgetting ``name`` is allowed at definition but should be obvious downstream."""

    class _NoName(Model):
        # Intentionally no ``name`` set.

        def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> _NoName:
            return self

        def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray[np.floating[Any]]:
            n = len(np.asarray(X))
            return np.zeros((n, 2))

    # Accessing ``name`` on a subclass that did not set it raises AttributeError.
    with pytest.raises(AttributeError):
        _ = _NoName.name
