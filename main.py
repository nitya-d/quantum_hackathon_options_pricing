"""Main Script (Qiskit-Fall-Fest-shaped) for Q-volution 2026.

The goal is to mirror the folder and code structure of the previous
`qiskit-fall-fest2025_paris-saclay_quandela` repo while using:

- MerLin (merlinquantum) photonic reservoir instead of Qiskit
- Q-volution CSV/Excel datasets instead of the old XLSX dataset

This file intentionally follows the old structure:
- parse_arguments()
- run_experiment(config)
- tune.py imports run_experiment
- results/ contains prediction_plot.png, residuals.png, etc.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.data_loader import DataLoader
from src.model import HybridQMLModel
from src.quantum_reservoir import QuantumReservoir


# Hyperparameter: data scaling factor for angle encoding
# Maps normalized data (Z-scores) to rotation angles.
# π/4 maps Z-score ±3 -> ±2.356 rad (manageable range, less wrap-around).
DATA_SCALING_FACTOR = np.pi / 4.0


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments (kept compatible with old repo)."""
    parser = argparse.ArgumentParser(
        description="Quantum Machine Learning for Option Price Prediction (Q-volution)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --data_file data/train.csv
  python main.py --data_file data/train.csv --n_qubits 8 --depth 3 --lookback 8
  python main.py --data_file data/train.csv --regressor lgbm --test_size 0.3
        """,
    )

    # Data parameters (single-file, old-style)
    parser.add_argument(
        "--data_file",
        type=str,
        default=str(Path("data") / "train.csv"),
        help="Path to data file (CSV or Excel)",
    )
    parser.add_argument(
        "--price_column",
        type=str,
        default=None,
        help="Name of the price column (defaults to 'Tenor : 10; Maturity : 10')",
    )
    parser.add_argument("--tenor", type=float, default=None, help="Tenor value to filter by")
    parser.add_argument("--maturity", type=float, default=None, help="Maturity value to filter by")

    parser.add_argument(
        "--use_log_returns",
        action="store_true",
        help="Use log returns instead of raw prices (default: True)",
    )
    parser.add_argument(
        "--no_log_returns",
        dest="use_log_returns",
        action="store_false",
        help="Disable log returns (use raw prices)",
    )
    parser.set_defaults(use_log_returns=True)

    parser.add_argument(
        "--lookback",
        type=int,
        default=8,
        help="Lookback window size for time-series (default: 8)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of data for testing (default: 0.2)",
    )

    # Quantum reservoir parameters (keep names from old repo)
    parser.add_argument("--n_qubits", type=int, default=8, help="Number of modes (legacy name) (<=20)")
    parser.add_argument("--depth", type=int, default=3, help="Reservoir depth")
    parser.add_argument(
        "--encoding",
        type=str,
        default="angle",
        choices=["angle", "amplitude"],
        help="Encoding type (amplitude not supported on target QPU)",
    )
    parser.add_argument(
        "--entanglement",
        type=str,
        default="linear",
        choices=["linear", "circular", "full"],
        help="Entanglement pattern (kept for compatibility; MerLin uses fixed interferometer)",
    )
    parser.add_argument("--shots", type=int, default=1024, help="Shots (kept for compatibility)")

    # Classical regressor parameters
    parser.add_argument(
        "--regressor",
        type=str,
        default="lgbm",
        choices=["linear", "ridge", "mlp", "lgbm"],
        help="Classical regressor type (default: lgbm)",
    )

    parser.add_argument(
        "--normalize_method",
        type=str,
        default="zscore",
        choices=["zscore"],
        help="Normalization method (zscore)",
    )

    # Output parameters
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: results/)")
    parser.add_argument("--visualize", action="store_true", default=True, help="Generate visualization plots")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples for debugging/tuning")

    return parser.parse_args()


def create_output_dir(output_dir: Optional[str]) -> Path:
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, *, title: str, save_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    x = np.arange(len(y_true))
    plt.plot(x, y_true, "b-", label="Actual", alpha=0.7, linewidth=2)
    plt.plot(x, y_pred, "r--", label="Predicted", alpha=0.7, linewidth=2)
    plt.xlabel("Sample Index")
    plt.ylabel("Price")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, *, save_path: Path) -> None:
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(residuals, "o", alpha=0.6, markersize=4)
    axes[0].axhline(0, color="r", linestyle="--", alpha=0.7)
    axes[0].set_title("Residuals vs Index")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Residual")
    axes[1].hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    axes[1].set_title("Residuals Histogram")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_results_summary(train_metrics: dict, test_metrics: dict, model_info: dict) -> None:
    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)
    print("\nModel Configuration:")
    for k, v in model_info.items():
        print(f"  {k}: {v}")
    print("\nTraining Metrics:")
    for k, v in train_metrics.items():
        if k == "mape":
            print(f"  {k.upper()}: {v:.4f}%")
        else:
            print(f"  {k.upper()}: {v:.6f}")
    print("\nTest Metrics:")
    for k, v in test_metrics.items():
        if k == "mape":
            print(f"  {k.upper()}: {v:.4f}%")
        else:
            print(f"  {k.upper()}: {v:.6f}")
    print("\n" + "=" * 80 + "\n")


def run_experiment(config: dict) -> dict:
    """Run a single experiment (kept compatible with old tune.py)."""
    np.random.seed(int(config["seed"]))
    output_dir = create_output_dir(config.get("output_dir"))
    verbose = bool(config.get("verbose", True))

    if config.get("encoding") == "amplitude":
        raise ValueError("Amplitude encoding is not supported (hardware constraint).")

    if verbose:
        print("\n" + "=" * 80)
        print("  QUANTUM MACHINE LEARNING FOR OPTION PRICE PREDICTION (Q-volution)")
        print("=" * 80)
        print("\nConfiguration:")
        for k in [
            "data_file",
            "tenor",
            "maturity",
            "use_log_returns",
            "n_qubits",
            "depth",
            "encoding",
            "entanglement",
            "lookback",
            "regressor",
            "test_size",
        ]:
       	    print(f"  {k}: {config.get(k)}")
        print(f"  data_scaling_factor: {config['data_scaling_factor']:.4f}")

    # Step 1: Data
    data_loader = DataLoader(
        normalize_method=config["normalize_method"],
        lookback_window=int(config["lookback"]),
        test_size=float(config["test_size"]),
        random_seed=int(config["seed"]),
    )

    data_file = Path(config["data_file"])
    if not data_file.is_absolute():
        data_file = Path(__file__).parent / data_file

    if verbose:
        try:
            pairs = data_loader.get_available_pairs(data_file)
            if pairs:
                print(f"\nAvailable (Tenor, Maturity) pairs: {len(pairs)}")
        except Exception as e:
            warnings.warn(f"Could not parse tenor/maturity pairs: {e}")

    X_train, X_test, y_train, y_test, test_initial_prices = data_loader.prepare_data(
        data_file,
        price_column=config.get("price_column"),
        tenor=config.get("tenor"),
        maturity=config.get("maturity"),
        use_log_returns=bool(config["use_log_returns"]),
        max_samples=config.get("max_samples"),
    )

    if verbose:
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Input shape: {X_train.shape}")

    # Step 2: Quantum reservoir (MerLin)
    quantum_reservoir = QuantumReservoir(
        n_qubits=int(config["n_qubits"]),
        n_photons=4,
        depth=int(config["depth"]),
        encoding_type=str(config["encoding"]),
        entanglement_pattern=str(config["entanglement"]),
        random_seed=int(config["seed"]),
        shots=int(config["shots"]),
        data_scaling_factor=float(config["data_scaling_factor"]),
    )

    if config.get("visualize", False):
        try:
            quantum_reservoir.visualize_circuit(str(output_dir / "quantum_reservoir_circuit.png"))
        except Exception as e:
            if verbose:
                print(f"Warning: Could not visualize circuit: {e}")

    # Step 3: Train classical head
    model = HybridQMLModel(quantum_reservoir=quantum_reservoir, regressor_type=str(config["regressor"]))
    model.fit(X_train, y_train, verbose=verbose)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Step 4.5: Price reconstruction (if using log returns)
    y_test_pred_prices = None
    y_test_actual_prices = None
    if config["use_log_returns"] and test_initial_prices is not None and len(X_test) > 0:
        y_test_pred_denorm = data_loader.denormalize(y_test_pred)
        y_test_denorm = data_loader.denormalize(y_test)
        initial_price = float(test_initial_prices[0])
        y_test_actual_prices = initial_price * np.exp(np.cumsum(y_test_denorm))
        prev_true = np.zeros_like(y_test_denorm, dtype=np.float64)
        prev_true[0] = initial_price
        if len(prev_true) > 1:
            prev_true[1:] = y_test_actual_prices[:-1]
        y_test_pred_prices = prev_true * np.exp(y_test_pred_denorm)

    # Step 4: Evaluation
    train_metrics = model.evaluate(X_train, y_train, metrics=["mse", "mae", "r2", "rmse"])
    test_metrics = model.evaluate(X_test, y_test, metrics=["mse", "mae", "r2", "rmse"])

    if y_test_pred_prices is not None and y_test_actual_prices is not None:
        test_metrics["mape"] = calculate_mape(y_test_actual_prices, y_test_pred_prices)

    # Step 5: Visualization + results_summary.txt
    if config.get("visualize", False):
        if y_test_pred_prices is not None and y_test_actual_prices is not None:
            # Single canonical plot for test prices (kept as prediction_plot.png)
            plot_predictions(
                y_test_actual_prices,
                y_test_pred_prices,
                title="Test Set: Predicted vs Actual Prices (Reconstructed)",
                save_path=output_dir / "prediction_plot.png",
            )
            plot_residuals(
                y_test_actual_prices,
                y_test_pred_prices,
                save_path=output_dir / "residuals.png",
            )
            plot_predictions(
                y_train,
                y_train_pred,
                title="Training Set: Predictions vs Actual (Log Returns)",
                save_path=output_dir / "train_predictions.png",
            )
        else:
            plot_predictions(
                y_train,
                y_train_pred,
                title="Training Set: Predictions vs Actual",
                save_path=output_dir / "train_predictions.png",
            )
            plot_predictions(
                y_test,
                y_test_pred,
                title="Test Set: Predictions vs Actual",
                save_path=output_dir / "test_predictions.png",
            )
            plot_predictions(
                y_test,
                y_test_pred,
                title="Predicted vs Actual",
                save_path=output_dir / "prediction_plot.png",
            )
            plot_residuals(
                y_test,
                y_test_pred,
                save_path=output_dir / "residuals.png",
            )

    model_info = {
        "n_qubits": config["n_qubits"],
        "circuit_depth": config["depth"],
        "encoding": config["encoding"],
        "entanglement": config["entanglement"],
        "lookback_window": config["lookback"],
        "regressor": config["regressor"],
        "use_log_returns": config["use_log_returns"],
        "tenor": config.get("tenor"),
        "maturity": config.get("maturity"),
        "training_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
    }
    if verbose:
        print_results_summary(train_metrics, test_metrics, model_info)
        results_file = output_dir / "results_summary.txt"
        with open(results_file, "w") as f:
            f.write("QUANTUM MACHINE LEARNING RESULTS\n")
            f.write("=" * 80 + "\n\n")
            f.write("Model Configuration:\n")
            for k, v in model_info.items():
                f.write(f"  {k}: {v}\n")
            f.write("\nTraining Metrics:\n")
            for k, v in train_metrics.items():
                if k == "mape":
                    f.write(f"  {k.upper()}: {v:.4f}%\n")
                else:
                    f.write(f"  {k.upper()}: {v:.6f}\n")
            f.write("\nTest Metrics:\n")
            for k, v in test_metrics.items():
                if k == "mape":
                    f.write(f"  {k.upper()}: {v:.4f}%\n")
                else:
                    f.write(f"  {k.upper()}: {v:.6f}\n")

    return {
        "r2": float(test_metrics.get("r2", float("nan"))),
        "mse": float(test_metrics["mse"]),
        "mae": float(test_metrics["mae"]),
        "rmse": float(test_metrics["rmse"]),
        "mape": test_metrics.get("mape", None),
    }


def main() -> None:
    args = parse_arguments()
    config = {
        "data_file": args.data_file,
        "tenor": args.tenor,
        "maturity": args.maturity,
        "use_log_returns": args.use_log_returns,
        "n_qubits": args.n_qubits,
        "depth": args.depth,
        "encoding": args.encoding,
        "entanglement": args.entanglement,
        "lookback": args.lookback,
        "regressor": args.regressor,
        "normalize_method": args.normalize_method,
        "test_size": args.test_size,
        "data_scaling_factor": DATA_SCALING_FACTOR,
        "shots": args.shots,
        "seed": args.seed,
        "output_dir": args.output_dir,
        "visualize": args.visualize,
        "price_column": args.price_column,
        "max_samples": args.max_samples,
        "verbose": True,
    }
    metrics = run_experiment(config)
    print("\nFinal Results:")
    print(f"  Test R²: {metrics['r2']:.6f}")
    print(f"  Test MSE: {metrics['mse']:.6f}")
    print(f"  Test MAE: {metrics['mae']:.6f}")
    print(f"  Test RMSE: {metrics['rmse']:.6f}")
    if metrics.get("mape") is not None:
        print(f"  Test MAPE: {metrics['mape']:.4f}%")


if __name__ == "__main__":
    main()

