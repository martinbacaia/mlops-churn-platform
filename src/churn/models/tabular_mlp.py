"""Tabular MLP (PyTorch) wrapped to satisfy the sklearn-compatible :class:`Model` contract.

The point of this module is not to beat XGBoost — the MLP is small (two hidden
layers) and tabular data with mostly-categorical features is gradient-boosted
trees' home turf. The point is to **prove the abstraction**: an entirely
different runtime (PyTorch) sits behind exactly the same fit / predict_proba /
predict surface as the linear and tree models, and the rest of the platform
can't tell the difference.

Determinism is non-negotiable: ``set_torch_deterministic`` is called at the top
of every ``fit`` so a re-fit with the same seed reproduces bit-for-bit on CPU.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
import torch
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from churn.models.base import Model
from churn.seeds import set_torch_deterministic


class _MLP(nn.Module):
    """Pure PyTorch module — kept private so callers never see it.

    Architecture: ``input → hidden[0] → hidden[1] → 1`` with ReLU + dropout
    between each linear layer. The output layer emits a single logit (no
    sigmoid here — :class:`torch.nn.BCEWithLogitsLoss` applies it numerically
    stably during training, and inference applies sigmoid explicitly).
    """

    def __init__(
        self,
        n_features: int,
        hidden: tuple[int, int] = (64, 32),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(x)
        return out


class TabularMLPModel(Model):
    """sklearn-compatible wrapper around :class:`_MLP`.

    Implements the full :class:`Model` contract while hiding all PyTorch
    machinery. Anything outside this module that touches a TabularMLPModel
    sees an estimator with ``fit / predict / predict_proba / get_params /
    set_params``; no ``import torch`` is required elsewhere.
    """

    name: ClassVar[str] = "tabular_mlp"

    def __init__(
        self,
        hidden: tuple[int, int] = (64, 32),
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 256,
        weight_decay: float = 0.0,
        random_state: int = 42,
    ) -> None:
        # sklearn's BaseEstimator requires ``__init__`` to be a pure assignment;
        # the network is constructed in ``fit`` once ``n_features_in_`` is known.
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.random_state = random_state

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> TabularMLPModel:
        set_torch_deterministic(self.random_state)
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X_arr.shape[1]

        self._model = _MLP(self.n_features_in_, self.hidden, self.dropout)
        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.BCEWithLogitsLoss()

        dataset = TensorDataset(torch.from_numpy(X_arr), torch.from_numpy(y_arr))
        # The DataLoader's RNG is fixed independently so shuffle order is
        # reproducible even if other torch RNGs are touched between epochs.
        generator = torch.Generator().manual_seed(self.random_state)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )

        self._model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self._model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
        return self

    @torch.no_grad()
    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray[np.floating[Any]]:
        check_is_fitted(self, "_model")
        self._model.eval()
        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
        logits = self._model(X_t)
        p1 = torch.sigmoid(logits).numpy().ravel()
        return np.column_stack([1.0 - p1, p1])
