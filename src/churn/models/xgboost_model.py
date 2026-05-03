"""XGBoost gradient-boosted-trees implementation of :class:`Model`."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
import xgboost as xgb
from sklearn.utils.validation import check_is_fitted

from churn.models.base import Model


class XGBoostModel(Model):
    """Gradient-boosted trees behind the platform contract.

    Defaults are tuned for tabular classification of moderate size — fast to
    train (``tree_method="hist"``, ``n_jobs=1`` for determinism) and shallow
    enough to resist overfitting the ~7K-row Telco dataset. The class imbalance
    is handled via ``scale_pos_weight``, computed from the fit data so the
    same hyperparameters generalize if the training prevalence shifts later.
    """

    name: ClassVar[str] = "xgboost"

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        # n_jobs=1 keeps tree construction order deterministic across runs;
        # the model is small, so multi-threading doesn't materially help.
        self.n_jobs = n_jobs

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> XGBoostModel:
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        n_pos = float((y_arr == 1).sum())
        n_neg = float((y_arr == 0).sum())
        # ``scale_pos_weight`` rebalances the loss function the way
        # ``class_weight="balanced"`` does for sklearn linear models.
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        self._estimator = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            tree_method="hist",
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
        )
        self._estimator.fit(X_arr, y_arr)
        self.classes_ = self._estimator.classes_
        self.n_features_in_ = self._estimator.n_features_in_
        return self

    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray[np.floating[Any]]:
        check_is_fitted(self, "_estimator")
        proba: npt.NDArray[np.floating[Any]] = self._estimator.predict_proba(np.asarray(X))
        return proba
