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
from typing import Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_DATA_PATH = Path("data") / "019c9f39-e697-7fa1-9725-d93bdd138124.parquet"


@dataclass
class DataConfig:
    """Configuration for loading and preprocessing the swaption dataset."""

    data_path: Path = DEFAULT_DATA_PATH
    target_column: str = "Tenor : 10; Maturity : 10"
    date_column: str = "Date"
    lookback_window: int = 8
    test_size: float = 0.2
    use_log_returns: bool = True
    normalize_method: str = "zscore"  # currently only 'zscore' is implemented
    max_samples: Optional[int] = None

    def __post_init__(self) -> None:
        if not (0.0 < self.test_size < 1.0):
            raise ValueError("test_size must be in (0, 1).")
        if self.lookback_window <= 0:
            raise ValueError("lookback_window must be positive.")


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

    # ------------------------------------------------------------------
    # Core loading utilities
    # ------------------------------------------------------------------
    def _load_raw_series(self) -> pd.Series:
        """Load and sort the target price series from the Parquet file."""
        data_path = self.config.data_path
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        df = pd.read_parquet(data_path)

        if self.config.date_column not in df.columns:
            raise ValueError(
                f"Expected date column '{self.config.date_column}' in dataset. "
                f"Available columns: {list(df.columns)}"
            )

        if self.config.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.config.target_column}' not found. "
                f"Please choose one of: {list(df.columns)}"
            )

        # Ensure proper datetime sorting
        df = df.copy()
        df[self.config.date_column] = pd.to_datetime(
            df[self.config.date_column], dayfirst=True, errors="coerce"
        )
        df = df.sort_values(self.config.date_column)
        df = df.dropna(subset=[self.config.date_column, self.config.target_column])

        return df[self.config.target_column].astype(float)

    def _to_log_returns(self, prices: pd.Series) -> pd.Series:
        """Convert price series to log returns."""
        returns = np.log(prices / prices.shift(1))
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        return returns

    def _build_windows(
        self, series: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build supervised windows from a 1D time series.

        For each index t >= lookback_window:
            X_t = series[t - lookback_window : t]
            y_t = series[t]
        """
        values = series.to_numpy(dtype=np.float64)
        lookback = self.config.lookback_window

        X_list = []
        y_list = []

        for t in range(lookback, len(values)):
            window = values[t - lookback : t]
            target = values[t]
            if np.any(~np.isfinite(window)) or not np.isfinite(target):
                continue
            X_list.append(window)
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
        prices = self._load_raw_series()

        if self.config.use_log_returns:
            series = self._to_log_returns(prices)
        else:
            series = prices.copy()

        X, y = self._build_windows(series)

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

        # Fit normalization on training features only
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

        # Cast to float32 for compatibility with PyTorch / MerLin
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)

        return X_train, X_test, y_train, y_test

