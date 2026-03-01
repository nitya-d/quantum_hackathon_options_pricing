"""
Entry point for Q-volution 2026 (Quandela Track B) QRC pipeline.

This script wires together:
- Parquet-based swaption data loader
- MerLin-based photonic Quantum Reservoir
- PyTorch MLP head for option price prediction
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data_loader import DataConfig, DataLoader
from src.model import HybridQMLModel, MLPConfig
from src.quantum_reservoir import QuantumReservoir, ReservoirConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Q-volution 2026 Quandela Track - Quantum Reservoir Computing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data parameters
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(Path("data") / "019c9f39-e697-7fa1-9725-d93bdd138124.parquet"),
        help="Path to Parquet swaption dataset.",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="Tenor : 10; Maturity : 10",
        help="Parquet column to use as target price series.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=8,
        help="Lookback window size for time-series features.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of samples reserved for testing.",
    )
    parser.add_argument(
        "--no_log_returns",
        action="store_true",
        help="Disable log-return transformation (use raw prices).",
    )

    # Quantum reservoir parameters (photonic constraints)
    parser.add_argument(
        "--n_modes",
        type=int,
        default=8,
        help="Number of photonic modes in the reservoir (<= 20).",
    )
    parser.add_argument(
        "--n_photons",
        type=int,
        default=4,
        help="Number of photons injected into the circuit (<= 10).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Number of entangling layers in the reservoir.",
    )
    parser.add_argument(
        "--data_scaling_factor",
        type=float,
        default=1.0,
        help="Global scaling applied before angle encoding.",
    )

    # Classical head
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[64, 32],
        help="Hidden layer sizes for the MLP regressor.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate in the MLP.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for AdamW optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for AdamW optimizer.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="ridge",
        choices=["mlp", "ridge", "lgbm"],
        help="Classical backend to use on top of quantum features.",
    )

    # Misc
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device for PyTorch (cpu or cuda).",
    )

    return parser.parse_args()


def build_configs(args: argparse.Namespace) -> Tuple[DataConfig, ReservoirConfig, MLPConfig]:
    # Ensure reservoir respects hardware constraints and matches lookback window.
    n_modes = max(args.n_modes, args.lookback)

    data_config = DataConfig(
        data_path=Path(args.data_path),
        target_column=args.target_column,
        lookback_window=args.lookback,
        test_size=args.test_size,
        use_log_returns=not args.no_log_returns,
        n_modes=n_modes,
    )

    res_config = ReservoirConfig(
        n_modes=n_modes,
        n_photons=args.n_photons,
        depth=args.depth,
        data_scaling_factor=args.data_scaling_factor,
        device=args.device,
    )

    mlp_config = MLPConfig(
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        device=args.device,
        classical_backend=args.model_type,
    )

    return data_config, res_config, mlp_config


def main() -> None:
    args = parse_args()
    data_config, res_config, mlp_config = build_configs(args)

    print("=== Loading data ===")
    loader = DataLoader(data_config)
    X_train, X_test, y_train, y_test = loader.load_and_preprocess()
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    print("=== Building quantum reservoir ===")
    reservoir = QuantumReservoir(res_config)

    print("=== Training hybrid model ===")
    model = HybridQMLModel(reservoir, mlp_config)
    model.fit(X_train, y_train)

    print("=== Evaluating on test set ===")
    # Predict in normalized space
    y_pred_norm = model.predict(X_test)

    # Inverse-transform targets and predictions back to original scale
    target_mean = getattr(loader, "target_mean_", None)
    target_std = getattr(loader, "target_std_", None)

    if target_mean is not None and target_std is not None:
        y_test_orig = y_test * target_std + target_mean
        y_pred_orig = y_pred_norm * target_std + target_mean
    else:
        y_test_orig = y_test
        y_pred_orig = y_pred_norm

    # Compute metrics on original scale
    mse = float(np.mean((y_pred_orig - y_test_orig) ** 2))
    mae = float(np.mean(np.abs(y_pred_orig - y_test_orig)))
    rmse = float(np.sqrt(mse))

    print("Test metrics:")
    print(f"  mse: {mse:.6f}")
    print(f"  rmse: {rmse:.6f}")
    print(f"  mae: {mae:.6f}")

    # ------------------------------------------------------------------
    # Plot and save prediction results
    # ------------------------------------------------------------------
    print("=== Plotting results ===")

    # If we used log returns, reconstruct price trajectories for plotting
    last_price = getattr(loader, "last_price_before_test_", None)
    if last_price is not None and last_price > 0:
        true_prices = last_price * np.exp(np.cumsum(y_test_orig))
        pred_prices = last_price * np.exp(np.cumsum(y_pred_orig))
        plot_ylabel = "Option price (reconstructed)"
        plot_title_ts = "Test set: true vs predicted (reconstructed prices)"
        plot_title_sc = "True vs predicted (test set, reconstructed prices)"
        y_plot_true = true_prices
        y_plot_pred = pred_prices
    else:
        plot_ylabel = "Target (original scale)"
        plot_title_ts = "Test set: true vs predicted"
        plot_title_sc = "True vs predicted (test set)"
        y_plot_true = y_test_orig
        y_plot_pred = y_pred_orig

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Time-series plot: true vs predicted
    fig_ts, ax_ts = plt.subplots(figsize=(8, 4))
    ax_ts.plot(y_plot_true, label="True", linewidth=1.5)
    ax_ts.plot(y_plot_pred, label="Predicted", linewidth=1.2, alpha=0.8)
    ax_ts.set_xlabel("Test sample index")
    ax_ts.set_ylabel(plot_ylabel)
    ax_ts.set_title(plot_title_ts)
    ax_ts.legend()
    fig_ts.tight_layout()
    fig_ts.savefig(results_dir / "test_timeseries.png", dpi=300)
    plt.close(fig_ts)

    # Scatter plot: true vs predicted
    fig_sc, ax_sc = plt.subplots(figsize=(4, 4))
    ax_sc.scatter(y_plot_true, y_plot_pred, alpha=0.6, s=10)
    min_val = float(min(y_plot_true.min(), y_plot_pred.min()))
    max_val = float(max(y_plot_true.max(), y_plot_pred.max()))
    ax_sc.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)
    ax_sc.set_xlabel("True price" if last_price else "True target")
    ax_sc.set_ylabel("Predicted price" if last_price else "Predicted target")
    ax_sc.set_title(plot_title_sc)
    fig_sc.tight_layout()
    fig_sc.savefig(results_dir / "test_scatter.png", dpi=300)
    plt.close(fig_sc)


if __name__ == "__main__":
    main()

