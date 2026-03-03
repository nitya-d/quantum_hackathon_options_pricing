"""
Data loading and preprocessing for the Q-volution 2026 Quandela track.

This module loads train and test CSV files (separate files), selects
date and target columns, builds supervised time-series windows, and
applies optional log-return transformation and z-score normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


DEFAULT_TRAIN_PATH = Path("data") / "train.csv"
DEFAULT_TEST_PATH = Path("data") / "test_template.csv"


@dataclass
class DataConfig:
    """Configuration for loading and preprocessing the swaption dataset."""

    train_path: Path = DEFAULT_TRAIN_PATH
    test_path: Path = DEFAULT_TEST_PATH
    # Can be a single column or a list of columns for multivariate inputs
    target_column: Union[str, Sequence[str]] = "Tenor : 10; Maturity : 10"
    date_column: str = "Date"
    lookback_window: int = 8
    use_log_returns: bool = True
    normalize_method: str = "zscore"  # currently only 'zscore' is implemented
    max_samples: Optional[int] = None
    # Target dimensionality for PCA / reservoir input (typically = n_modes)
    n_modes: int = 8

    def __post_init__(self) -> None:
        if self.lookback_window <= 0:
            raise ValueError("lookback_window must be positive.")
        if self.n_modes <= 0:
            raise ValueError("n_modes must be positive.")
        if isinstance(self.train_path, str):
            self.train_path = Path(self.train_path)
        if isinstance(self.test_path, str):
            self.test_path = Path(self.test_path)


class DataLoader:
    """
    CSV-based data loader with separate train and test files.

    Produces:
    - X_train, X_test: arrays of shape [n_samples, n_features] (after PCA if applied)
    - y_train: array of shape [n_samples]
    - y_test: array of shape [n_test] or None if test file has no labels
    """

    def __init__(self, config: DataConfig) -> None:
        self.config = config

        self.feature_mean_: Optional[float] = None
        self.feature_std_: Optional[float] = None
        self.target_mean_: Optional[float] = None
        self.target_std_: Optional[float] = None
        self.pca_: Optional[PCA] = None
        # Last raw price in training set (for log-return → price reconstruction)
        self.last_price_before_test_: Optional[float] = None

    # ------------------------------------------------------------------
    # Core loading utilities
    # ------------------------------------------------------------------
    def _load_csv(
        self, path: Path, required_targets: bool = True
    ) -> pd.DataFrame:
        """
        Load a CSV or Excel file and return a DataFrame with date_column and target columns only.
        Drops optional non-numeric columns (e.g. 'Type'). Sorted by date.
        """
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        path_str = str(path)
        if path_str.lower().endswith(".csv"):
            df = pd.read_csv(path)
        else:
            # Excel: read first sheet
            df = pd.read_excel(path, sheet_name=0, engine="openpyxl")

        if self.config.date_column not in df.columns:
            raise ValueError(
                f"Expected date column '{self.config.date_column}' in {path.name}. "
                f"Available columns: {list(df.columns)}"
            )

        if isinstance(self.config.target_column, str):
            target_cols = [self.config.target_column]
        else:
            target_cols = list(self.config.target_column)

        missing = [c for c in target_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Target column(s) {missing} not found in {path.name}. "
                f"Please choose among: {list(df.columns)}"
            )

        keep = [self.config.date_column] + target_cols
        df = df[keep].copy()
        df[self.config.date_column] = pd.to_datetime(
            df[self.config.date_column], dayfirst=True, errors="coerce"
        )
        df = df.sort_values(self.config.date_column)

        if required_targets:
            df = df.dropna(subset=target_cols)
        # else: keep rows even if target is NaN (for test with missing labels)

        return df

    def _to_log_returns(self, values: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Convert price series or frame to log returns."""
        returns = np.log(values / values.shift(1))
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        return returns if isinstance(returns, pd.DataFrame) else returns.to_frame()

    def _build_windows(
        self,
        frame: pd.DataFrame,
        allow_nan_target: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build supervised windows from a (possibly multivariate) time series.
        If allow_nan_target is True, rows with NaN target are still included;
        y will contain np.nan for those indices.
        """
        values = frame.to_numpy(dtype=np.float64)  # shape: [T, C]
        lookback = self.config.lookback_window

        X_list = []
        y_list = []

        for t in range(lookback, len(values)):
            window = values[t - lookback : t, :]
            target = values[t, 0]
            if np.any(~np.isfinite(window)):
                continue
            if not allow_nan_target and not np.isfinite(target):
                continue
            X_list.append(window.reshape(-1))
            y_list.append(target if np.isfinite(target) else np.nan)

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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Load train and test CSVs and return (X_train, X_test, y_train, y_test).
        y_test is None if the test file has no valid target labels.
        """
        if isinstance(self.config.target_column, str):
            target_cols = [self.config.target_column]
        else:
            target_cols = list(self.config.target_column)

        # ----- Train -----
        raw_train = self._load_csv(self.config.train_path, required_targets=True)
        train_values = raw_train[target_cols]
        if self.config.use_log_returns:
            frame_train = self._to_log_returns(train_values)
        else:
            frame_train = train_values.copy()

        X_train, y_train = self._build_windows(frame_train, allow_nan_target=False)

        # Last raw price in train (for reconstructing test prices from log returns)
        if self.config.use_log_returns and len(raw_train) > 0:
            self.last_price_before_test_ = float(raw_train[target_cols].iloc[-1, 0])
        else:
            self.last_price_before_test_ = None

        if self.config.max_samples is not None:
            max_n = min(self.config.max_samples, X_train.shape[0])
            X_train = X_train[:max_n]
            y_train = y_train[:max_n]

        # ----- Test -----
        raw_test = self._load_csv(self.config.test_path, required_targets=False)
        raw_test = raw_test.dropna(subset=target_cols)

        test_values = raw_test[target_cols]
        if self.config.use_log_returns:
            frame_test = self._to_log_returns(test_values)
        else:
            frame_test = test_values.copy()

        # If test frame is empty or too short, we have no test samples
        if len(frame_test) < self.config.lookback_window:
            # Placeholder shape; will be resized after PCA to match train
            X_test = np.zeros((0, X_train.shape[1]), dtype=np.float32)
            y_test = None
        else:
            X_test, y_test_arr = self._build_windows(
                frame_test, allow_nan_target=True
            )
            has_labels = np.any(np.isfinite(y_test_arr))
            y_test = y_test_arr if has_labels else None

        # ----- Normalization (fit on train only) -----
        if self.config.normalize_method == "zscore":
            self.feature_mean_ = float(X_train.mean())
            self.feature_std_ = float(X_train.std(ddof=0) + 1e-8)

            X_train = (X_train - self.feature_mean_) / self.feature_std_
            if X_test.shape[0] > 0:
                X_test = (X_test - self.feature_mean_) / self.feature_std_

            self.target_mean_ = float(y_train.mean())
            self.target_std_ = float(y_train.std(ddof=0) + 1e-8)

            y_train = (y_train - self.target_mean_) / self.target_std_
            if y_test is not None:
                y_test = (y_test - self.target_mean_) / self.target_std_
        else:
            raise ValueError(
                f"Unsupported normalize_method: {self.config.normalize_method}. "
                "Only 'zscore' is currently implemented."
            )

        # ----- PCA (fit on train only) -----
        original_dim = X_train.shape[1]
        target_dim = self.config.n_modes

        if original_dim > target_dim:
            self.pca_ = PCA(n_components=target_dim)
            X_train = self.pca_.fit_transform(X_train)
            if X_test.shape[0] > 0:
                X_test = self.pca_.transform(X_test)
            else:
                X_test = np.zeros((0, target_dim), dtype=np.float32)
        else:
            self.pca_ = None
            if X_test.shape[0] == 0:
                X_test = np.zeros((0, original_dim), dtype=np.float32)

        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.float32)
        if y_test is not None:
            y_test = y_test.astype(np.float32)

        return X_train, X_test, y_train, y_test
