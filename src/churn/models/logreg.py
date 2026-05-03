"""Logistic-regression baseline implementation of :class:`Model`."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from churn.models.base import Model


class LogRegModel(Model):
    """Linear classifier on the post-feature-pipeline matrix.

    Useful as an interpretable baseline: coefficients are inspectable and the
    model's calibration is usually decent without any post-hoc fitting.
    ``class_weight="balanced"`` by default because the Telco target is ~26 %
    positive — leaving it None silently lets the model under-predict churn.
    """

    name: ClassVar[str] = "logreg"

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
        class_weight: str | None = "balanced",
    ) -> None:
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.class_weight = class_weight

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> LogRegModel:
        self._estimator = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            class_weight=self.class_weight,
            solver="lbfgs",
        )
        self._estimator.fit(np.asarray(X), np.asarray(y))
        self.classes_ = self._estimator.classes_
        self.n_features_in_ = self._estimator.n_features_in_
        return self

    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray[np.floating[Any]]:
        check_is_fitted(self, "_estimator")
        proba: npt.NDArray[np.floating[Any]] = self._estimator.predict_proba(np.asarray(X))
        return proba
