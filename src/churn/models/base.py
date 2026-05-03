"""Abstract :class:`Model` contract that every churn classifier implements.

The platform is **model-agnostic**: training, evaluation, registry promotion,
serving, and behavior tests all interact with concrete classifiers exclusively
through this interface. Swap one for another at any point and the rest of the
system does not notice.

The contract is deliberately small:

* ``name`` — class-level identifier used as an MLflow tag.
* ``fit(X, y)`` — train; returns self.
* ``predict_proba(X)`` — probability over both classes, shape ``(n, 2)``.

A default :meth:`Model.predict` is provided in terms of ``predict_proba`` so
subclasses rarely need to define it. Save / load are intentionally **not** part
of the contract: every implementation is picklable, so the registry layer
serializes them uniformly with joblib.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClassifierMixin


class Model(BaseEstimator, ClassifierMixin, ABC):  # type: ignore[misc]
    """Platform-wide classifier contract.

    Concrete subclasses must:

    * Set the class-level :attr:`name` to a unique string (used as an MLflow tag
      and as the key under which the class is registered in
      :data:`churn.models.registry.MODEL_REGISTRY`).
    * Implement :meth:`fit`, returning ``self``.
    * Implement :meth:`predict_proba`, returning shape ``(n, 2)`` with column 1
      being the positive-class (churn) probability.

    The default :meth:`predict` is a threshold over ``predict_proba``; rarely
    overridden.
    """

    name: ClassVar[str]

    @abstractmethod
    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> Model: ...

    @abstractmethod
    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray[np.floating[Any]]: ...

    def predict(self, X: npt.ArrayLike, threshold: float = 0.5) -> npt.NDArray[np.int64]:
        proba = self.predict_proba(X)
        if proba.ndim != 2 or proba.shape[1] != 2:
            raise ValueError(f"predict_proba must return shape (n, 2); got {proba.shape}.")
        return (proba[:, 1] >= threshold).astype(np.int64)
