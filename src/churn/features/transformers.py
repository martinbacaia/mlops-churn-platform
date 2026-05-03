"""Custom sklearn-compatible transformers for the churn feature pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TenureBucketizer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Discretize a single tenure column into customer-lifecycle buckets.

    The default boundaries reflect Telco-style customer phases:

    * **0** — first year (months 0–11), where churn risk is highest
    * **1** — second year (12–23), where promotional pricing typically ends
    * **2** — multi-year mid-tenure (24–47)
    * **3** — long-loyal (48+)

    Output is a single integer column. Pair it with a one-hot encoder downstream
    if a downstream linear model needs the buckets disjoint.

    The transformer is stateless apart from sklearn's ``n_features_in_`` bookkeeping;
    ``fit`` records that count so refit-after-clone behaves correctly.
    """

    def __init__(self, bins: tuple[int, ...] = (12, 24, 48)) -> None:
        # Internal cut-points (exclusive lower edges of buckets 1, 2, 3 ...).
        self.bins = bins

    def fit(
        self, X: pd.DataFrame | pd.Series[Any] | npt.NDArray[Any], y: Any = None
    ) -> TenureBucketizer:
        arr = self._to_1d(X)
        self.n_features_in_ = 1
        # Validate that the input is numeric — mirrors sklearn's eager-error contract.
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(f"TenureBucketizer expects numeric input; got dtype {arr.dtype}.")
        return self

    def transform(
        self, X: pd.DataFrame | pd.Series[Any] | npt.NDArray[Any]
    ) -> npt.NDArray[np.int64]:
        arr = self._to_1d(X)
        bucketed = np.digitize(arr, self.bins, right=False).astype(np.int64)
        return bucketed.reshape(-1, 1)

    def get_feature_names_out(
        self, input_features: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.object_]:
        # Single output column regardless of input naming. Stable name lets the
        # downstream OHE produce predictable column names like ``tenure_bucket_0``.
        return np.array(["tenure_bucket"], dtype=object)

    @staticmethod
    def _to_1d(X: pd.DataFrame | pd.Series[Any] | npt.NDArray[Any]) -> npt.NDArray[Any]:
        if isinstance(X, pd.DataFrame):
            if X.shape[1] != 1:
                raise ValueError(f"TenureBucketizer expects a single column; got {X.shape[1]}.")
            return X.iloc[:, 0].to_numpy()
        if isinstance(X, pd.Series):
            return X.to_numpy()
        arr = np.asarray(X)
        if arr.ndim == 2:
            if arr.shape[1] != 1:
                raise ValueError(
                    f"TenureBucketizer expects a single column; got shape {arr.shape}."
                )
            arr = arr.ravel()
        return arr
