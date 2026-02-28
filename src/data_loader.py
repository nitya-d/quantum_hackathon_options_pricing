"""
Data loading and preprocessing for the Q-volution 2026 Quandela track.

This module adapts the previous QRC data pipeline to the new Parquet
swaption dataset delivered for the hackathon.

Key responsibilities
--------------------
- Load `data/019c9f39-e697-7fa1-9725-d93bdd138124.parquet`.
- Select a specific (Tenor, Maturity) column as the target price series.
- Build supervised time-series windows `(X, y)` with a configurable lookback.
- Apply optional log-return transformation and z-score normalization (train only).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


DEFAULT_DATA_PATH = Path("data") / "019c9f39-e697-7fa1-9725-d93bdd138124.parquet"


@dataclass
class DataConfig:
    """Configuration for loading and preprocessing the swaption dataset."""

    data_path: Path = DEFAULT_DATA_PATH
    # Can be a single column or a list of columns for multivariate inputs
    target_column: Union[str, Sequence[str]] = "Tenor : 10; Maturity : 10"
    date_column: str = "Date"
    lookback_window: int = 8
    test_size: float = 0.2
    use_log_returns: bool = True
    normalize_method: str = "zscore"  # currently only 'zscore' is implemented
    max_samples: Optional[int] = None
    # Target dimensionality for PCA / reservoir input (typically = n_modes)
    n_modes: int = 8

    def __post_init__(self) -> None:
        if not (0.0 < self.test_size < 1.0):
            raise ValueError("test_size must be in (0, 1).")
        if self.lookback_window <= 0:
            raise ValueError("lookback_window must be positive.")
        if self.n_modes <= 0:
            raise ValueError("n_modes must be positive.")


class DataLoader:
    """
    Parquet-based data loader for the swaption dataset.

    The loader produces:
    - X_train, X_test: arrays of shape [n_samples, lookback_window]
    - y_train, y_test: arrays of shape [n_samples]

    These are designed to match the QuantumReservoir interface, which expects
    classical inputs with dimensionality <= number of photonic modes.
    """

    def __init__(self, config: DataConfig) -> None:
        self.config = config

        # Normalization parameters (computed from training data only)
        self.feature_mean_: Optional[float] = None
        self.feature_std_: Optional[float] = None
        # PCA model (fitted on training set only, if used)
        self.pca_: Optional[PCA] = None

    # ------------------------------------------------------------------
    # Core loading utilities
    # ------------------------------------------------------------------
    def _load_raw_series(self) -> pd.DataFrame:
        """Load and sort the target price series (one or more columns)."""
        data_path = self.config.data_path
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        df = pd.read_parquet(data_path)

        if self.config.date_column not in df.columns:
            raise ValueError(
                f"Expected date column '{self.config.date_column}' in dataset. "
                f"Available columns: {list(df.columns)}"
            )

        # Normalize target_column to a list of column names
        if isinstance(self.config.target_column, str):
            target_cols = [self.config.target_column]
        else:
            target_cols = list(self.config.target_column)

        missing = [c for c in target_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Target column(s) {missing} not found. "
                f"Please choose among: {list(df.columns)}"
            )

        # Ensure proper datetime sorting
        df = df.copy()
        df[self.config.date_column] = pd.to_datetime(
            df[self.config.date_column], dayfirst=True, errors="coerce"
        )
        df = df.sort_values(self.config.date_column)
        df = df.dropna(subset=[self.config.date_column] + target_cols)

        return df[target_cols].astype(float)

    def _to_log_returns(self, values: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Convert price series or frame to log returns."""
        returns = np.log(values / values.shift(1))
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        # Always return a DataFrame for downstream consistency
        return returns if isinstance(returns, pd.DataFrame) else returns.to_frame()

    def _build_windows(
        self, frame: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build supervised windows from a (possibly multivariate) time series.

        For each index t >= lookback_window:
            X_t = frame[t - lookback_window : t, :]
            y_t = frame[t, 0]   # first column as target
        """
        values = frame.to_numpy(dtype=np.float64)  # shape: [T, C]
        lookback = self.config.lookback_window

        X_list = []
        y_list = []

        for t in range(lookback, len(values)):
            window = values[t - lookback : t, :]  # [lookback, C]
            target = values[t, 0]  # use first column as scalar target
            if np.any(~np.isfinite(window)) or not np.isfinite(target):
                continue
            # Flatten [lookback, C] -> [lookback * C]
            X_list.append(window.reshape(-1))
            y_list.append(target)

        if not X_list:
            raise ValueError("Not enough valid data to build time-series windows.")

        X = np.stack(X_list, axis=0)
        y = np.asarray(y_list, dtype=np.float64)
        return X, y

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_and_preprocess(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load the Parquet dataset and return train/test splits.

        Returns
        -------
        X_train, X_test, y_train, y_test : np.ndarray
            Arrays with shapes:
            - X_*: [n_samples, lookback_window]
            - y_*: [n_samples]
        """
        raw = self._load_raw_series()

        if self.config.use_log_returns:
            frame = self._to_log_returns(raw)
        else:
            frame = raw.copy()

        X, y = self._build_windows(frame)

        # Chronological train/test split
        n_samples = X.shape[0]
        split_idx = int((1.0 - self.config.test_size) * n_samples)
        split_idx = max(split_idx, 1)

        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_test = X[split_idx:]
        y_test = y[split_idx:]

        # Optional downsampling for quick experiments
        if self.config.max_samples is not None:
            max_n = self.config.max_samples
            X_train = X_train[:max_n]
            y_train = y_train[:max_n]

        # Fit normalization on training features only (scalar z-score)
        if self.config.normalize_method == "zscore":
            self.feature_mean_ = float(X_train.mean())
            self.feature_std_ = float(X_train.std(ddof=0) + 1e-8)

            X_train = (X_train - self.feature_mean_) / self.feature_std_
            X_test = (X_test - self.feature_mean_) / self.feature_std_
        else:
            raise ValueError(
                f"Unsupported normalize_method: {self.config.normalize_method}. "
                "Only 'zscore' is currently implemented."
            )

        # Dimensionality reduction with PCA to respect mode constraints.
        # If flattened window dimension exceeds n_modes, compress to n_modes.
        original_dim = X_train.shape[1]
        target_dim = self.config.n_modes

        if original_dim > target_dim:
            self.pca_ = PCA(n_components=target_dim)
            X_train = self.pca_.fit_transform(X_train)
            X_test = self.pca_.transform(X_test)
        else:
            self.pca_ = None

        # Cast to float32 for compatibility with PyTorch / MerLin
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)

        return X_train, X_test, y_train, y_test

