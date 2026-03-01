"""
Hybrid Quantum-Classical model for Q-volution 2026 (Quandela track).

This module defines a PyTorch MLP regressor that consumes fixed quantum
reservoir features produced by the MerLin-based `QuantumReservoir`.

The design follows the previous competition architecture:
- Quantum reservoir is fixed / non-trainable.
- Classical head (MLP) is trained via gradient descent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import RidgeCV
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset

from .quantum_reservoir import QuantumReservoir

try:
    from lightgbm import LGBMRegressor

    HAS_LGBM = True
except ImportError:  # pragma: no cover - optional dependency
    LGBMRegressor = None  # type: ignore
    HAS_LGBM = False


@dataclass
class MLPConfig:
    """
    Configuration for the classical head.

    Supports three backends:
    - 'ridge' : scikit-learn RidgeCV (default, recommended)
    - 'mlp'   : PyTorch MLPRegressor
    - 'lgbm'  : LightGBM LGBMRegressor
    """

    hidden_dims: Tuple[int, ...] = (64, 32)
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    n_epochs: int = 50
    device: str = "cpu"

    # 'ridge' (RidgeCV) is the robust default for RC
    classical_backend: str = "ridge"  # 'ridge', 'mlp', 'lgbm'
    # Candidate alphas for RidgeCV; tuned via built-in CV
    ridge_alphas: Tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    lgbm_num_leaves: int = 31
    lgbm_learning_rate: float = 0.05
    lgbm_n_estimators: int = 100


class MLPRegressor(nn.Module):
    """Simple feed-forward network for regression."""

    def __init__(self, input_dim: int, config: MLPConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class HybridQMLModel:
    """
    Hybrid QML model: fixed quantum reservoir + trainable MLP head.

    API
    ---
    - fit(X_train, y_train)
    - evaluate(X_test, y_test)
    - predict(X)
    """

    def __init__(
        self,
        quantum_reservoir: QuantumReservoir,
        mlp_config: MLPConfig,
    ) -> None:
        self.quantum_reservoir = quantum_reservoir
        self.mlp_config = mlp_config

        self.device = torch.device(mlp_config.device)
        self.backend = mlp_config.classical_backend.lower()

        # Models for different backends
        self.model_torch: MLPRegressor | None = None
        self.model_sklearn: Any | None = None

    def _ensure_mlp(self, feature_dim: int) -> None:
        if self.model_torch is None:
            self.model_torch = MLPRegressor(feature_dim, self.mlp_config).to(
                self.device
            )

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x.astype(np.float32)).to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the classical head on top of quantum reservoir features.
        """
        # Step 1: extract quantum features (no gradients).
        X_q = self.quantum_reservoir.get_reservoir_states(X_train)

        backend = self.backend

        if backend == "mlp":
            feature_dim = X_q.shape[1]
            self._ensure_mlp(feature_dim)
            assert self.model_torch is not None

            X_tensor = self._to_tensor(X_q)
            y_tensor = self._to_tensor(y_train).view(-1)

            dataset = TensorDataset(X_tensor, y_tensor)
            loader = TorchDataLoader(
                dataset, batch_size=self.mlp_config.batch_size, shuffle=True
            )

            optimizer = optim.AdamW(
                self.model_torch.parameters(),
                lr=self.mlp_config.learning_rate,
                weight_decay=self.mlp_config.weight_decay,
            )
            loss_fn = nn.MSELoss()

            self.model_torch.train()
            for epoch in range(self.mlp_config.n_epochs):
                epoch_loss = 0.0
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()
                    preds = self.model_torch(batch_X)
                    loss = loss_fn(preds, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * batch_X.size(0)

                epoch_loss /= len(dataset)
                # Optional: simple progress print for debugging
                print(f"[Epoch {epoch + 1}/{self.mlp_config.n_epochs}] "
                      f"Train MSE: {epoch_loss:.6f}")

        elif backend == "ridge":
            # Standard RC readout: RidgeCV on reservoir features
            self.model_sklearn = RidgeCV(alphas=self.mlp_config.ridge_alphas)
            self.model_sklearn.fit(X_q, y_train.astype(np.float64))

        elif backend == "lgbm":
            if not HAS_LGBM:
                raise ImportError(
                    "LightGBM backend selected but lightgbm is not installed. "
                    "Install with `pip install lightgbm`."
                )
            self.model_sklearn = LGBMRegressor(
                num_leaves=self.mlp_config.lgbm_num_leaves,
                learning_rate=self.mlp_config.lgbm_learning_rate,
                n_estimators=self.mlp_config.lgbm_n_estimators,
            )
            self.model_sklearn.fit(X_q, y_train.astype(np.float64))

        else:
            raise ValueError(
                f"Unknown classical_backend '{backend}'. "
                "Supported: 'mlp', 'ridge', 'lgbm'."
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the trained hybrid model.
        """
        X_q = self.quantum_reservoir.get_reservoir_states(X)

        backend = self.backend

        if backend == "mlp":
            if self.model_torch is None:
                raise RuntimeError("MLP backend not trained. Call fit() first.")
            X_tensor = self._to_tensor(X_q)
            self.model_torch.eval()
            with torch.no_grad():
                preds = self.model_torch(X_tensor)
            return preds.detach().cpu().numpy().astype(np.float32)

        elif backend in {"ridge", "lgbm"}:
            if self.model_sklearn is None:
                raise RuntimeError(f"{backend} backend not trained. Call fit() first.")
            preds = self.model_sklearn.predict(X_q)
            return preds.astype(np.float32)

        else:
            raise ValueError(
                f"Unknown classical_backend '{backend}'. "
                "Supported: 'mlp', 'ridge', 'lgbm'."
            )

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate model performance with basic regression metrics.
        """
        preds = self.predict(X)
        y_true = y.astype(np.float32)

        mse = float(np.mean((preds - y_true) ** 2))
        mae = float(np.mean(np.abs(preds - y_true)))
        rmse = float(np.sqrt(mse))

        return {"mse": mse, "rmse": rmse, "mae": mae}

