"""
Hyperparameter Tuning Script (Qiskit-Fall-Fest-shaped) for Q-volution 2026.

This script mirrors the structure of the previous qiskit-fall-fest repository:
- Imports `run_experiment` from main.py
- Performs a simple grid search / sampling over key hyperparameters
- Writes tuning_results.json and tuning_heatmap.png under results/
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from main import DATA_SCALING_FACTOR, run_experiment


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hyperparameter Tuning for Q-volution QRC Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=str(Path("data") / "train.csv"),
        help="Path to data file (default: data/train.csv)",
    )
    parser.add_argument("--tenor", type=float, default=None, help="Tenor value")
    parser.add_argument("--maturity", type=float, default=None, help="Maturity value")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: results/)")
    parser.add_argument("--max_samples", type=int, default=1000, help="Limit samples for quick tuning")
    parser.add_argument("--max_combinations", type=int, default=None, help="Maximum number of combinations to test")
    parser.add_argument(
        "--smart_sampling",
        action="store_true",
        default=True,
        help="Enable smart sampling (default: True)",
    )
    parser.add_argument(
        "--no_smart_sampling",
        dest="smart_sampling",
        action="store_false",
        help="Disable smart sampling (test all combinations)",
    )
    return parser.parse_args()


def plot_tuning_results(all_results: List[Dict[str, Any]], results_dir: Path) -> None:
    if not all_results:
        print("Warning: No results to plot.")
        return

    n_qubits_list = sorted({r["config"]["n_qubits"] for r in all_results})
    lookback_list = sorted({r["config"]["lookback"] for r in all_results})

    heatmap_matrix = np.full((len(n_qubits_list), len(lookback_list)), np.nan)
    for i, n_qubits in enumerate(n_qubits_list):
        for j, lookback in enumerate(lookback_list):
            relevant = [
                r for r in all_results
                if r["config"]["n_qubits"] == n_qubits and r["config"]["lookback"] == lookback
            ]
            if relevant:
                heatmap_matrix[i, j] = max(float(r["metrics"]["r2"]) for r in relevant)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        heatmap_matrix,
        cmap="RdYlGn",
        aspect="auto",
        vmin=np.nanmin(heatmap_matrix),
        vmax=np.nanmax(heatmap_matrix),
    )
    ax.set_xticks(np.arange(len(lookback_list)))
    ax.set_yticks(np.arange(len(n_qubits_list)))
    ax.set_xticklabels(lookback_list)
    ax.set_yticklabels(n_qubits_list)
    ax.set_xlabel("Lookback Window", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Qubits (Modes)", fontsize=12, fontweight="bold")
    ax.set_title("Hyperparameter Tuning Results: Best R² Score", fontsize=14, fontweight="bold")

    for i in range(len(n_qubits_list)):
        for j in range(len(lookback_list)):
            if not np.isnan(heatmap_matrix[i, j]):
                ax.text(j, i, f"{heatmap_matrix[i, j]:.4f}", ha="center", va="center", color="black", fontsize=10)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("R² Score", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plot_path = results_dir / "tuning_heatmap.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Tuning heatmap saved to: {plot_path}")


def main() -> None:
    args = parse_arguments()
    results_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Search space (kept similar to old repo; n_qubits==modes in photonics)
    search_space = {
        "n_qubits": [4, 6, 8, 10, 12, 16, 20],
        "lookback": [3, 4, 6, 8],
        "depth": [2, 3, 4],
        "entanglement": ["linear"],
        "scaling_mult": [0.5, 1.0, 1.5],
        "regressor": ["ridge", "lgbm"],
    }

    # Ensure lookback <= n_qubits for a fair comparison (optional heuristic)
    valid = []
    for combo in itertools.product(
        search_space["n_qubits"],
        search_space["lookback"],
        search_space["depth"],
        search_space["entanglement"],
        search_space["scaling_mult"],
        search_space["regressor"],
    ):
        n_qubits, lookback, depth, entanglement, scaling_mult, regressor = combo
        if lookback <= n_qubits:
            valid.append(combo)

    if args.max_combinations is not None and args.max_combinations < len(valid):
        random.shuffle(valid)
        valid = valid[: args.max_combinations]

    all_results: List[Dict[str, Any]] = []
    best_r2 = -1e9
    best_config: Dict[str, Any] = {}

    for idx, (n_qubits, lookback, depth, entanglement, scaling_mult, regressor) in enumerate(valid, start=1):
        cfg = {
            "data_file": args.data_file,
            "tenor": args.tenor,
            "maturity": args.maturity,
            "use_log_returns": True,
            "n_qubits": int(n_qubits),
            "depth": int(depth),
            "encoding": "angle",
            "entanglement": entanglement,
            "lookback": int(lookback),
            "regressor": regressor,
            "normalize_method": "zscore",
            "test_size": 0.2,
            "data_scaling_factor": float(DATA_SCALING_FACTOR) * float(scaling_mult),
            "shots": 1024,
            "seed": 42,
            "output_dir": str(results_dir),
            "visualize": False,
            "price_column": None,
            "max_samples": args.max_samples,
            "verbose": False,
        }

        metrics = run_experiment(cfg)
        result = {"config": cfg, "metrics": metrics}
        all_results.append(result)

        r2 = float(metrics.get("r2", float("nan")))
        if np.isfinite(r2) and r2 > best_r2:
            best_r2 = r2
            best_config = cfg.copy()

        if idx % 10 == 0:
            print(f"[{idx}/{len(valid)}] best_r2={best_r2:.6f}")

    # Save tuning results
    tuning_path = results_dir / "tuning_results.json"
    with open(tuning_path, "w") as f:
        json.dump({"all_results": all_results, "best_r2": best_r2, "best_config": best_config}, f, indent=2)
    print(f"Tuning results saved to: {tuning_path}")

    # Save best_params.json (same name as old repo)
    best_params_path = results_dir / "best_params.json"
    with open(best_params_path, "w") as f:
        json.dump({"best_r2": best_r2, "best_config": best_config}, f, indent=2)
    print(f"Best params saved to: {best_params_path}")

    plot_tuning_results(all_results, results_dir)


if __name__ == "__main__":
    main()

