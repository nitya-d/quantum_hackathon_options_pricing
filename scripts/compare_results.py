"""
Compare qiskit-fall-fest and q-volution results side-by-side.

This is a lightweight helper that parses `results_summary.txt` produced by both
projects and prints a compact comparison table.

Usage:
  .venv/bin/python scripts/compare_results.py \
    --qiskit_results ../qiskit-fall-fest2025_paris-saclay_quandela/results \
    --qvolution_results results
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple


def parse_summary(path: Path) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, float]]:
    """
    Returns: (model_info, train_metrics, test_metrics)
    """
    text = path.read_text(encoding="utf-8", errors="replace")

    def parse_block(block_name: str) -> Dict[str, str]:
        # Grab lines after "<block_name>:" until next blank line or end
        m = re.search(rf"{re.escape(block_name)}:\n(.*?)(\n\n|$)", text, flags=re.S)
        if not m:
            return {}
        block = m.group(1)
        out: Dict[str, str] = {}
        for line in block.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
        return out

    def to_float_metrics(d: Dict[str, str]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, v in d.items():
            v_clean = v.replace("%", "").strip()
            try:
                out[k.lower()] = float(v_clean)
            except ValueError:
                continue
        return out

    model_info = parse_block("Model Configuration")
    train_metrics = to_float_metrics(parse_block("Training Metrics"))
    test_metrics = to_float_metrics(parse_block("Test Metrics"))
    return model_info, train_metrics, test_metrics


def fmt(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{x:.6f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare results_summary.txt across two runs")
    ap.add_argument("--qiskit_results", type=str, required=True, help="Path to qiskit results directory")
    ap.add_argument("--qvolution_results", type=str, required=True, help="Path to q-volution results directory")
    args = ap.parse_args()

    qiskit_summary = Path(args.qiskit_results) / "results_summary.txt"
    qvol_summary = Path(args.qvolution_results) / "results_summary.txt"
    if not qiskit_summary.exists():
        raise FileNotFoundError(qiskit_summary)
    if not qvol_summary.exists():
        raise FileNotFoundError(qvol_summary)

    qiskit_info, qiskit_train, qiskit_test = parse_summary(qiskit_summary)
    qvol_info, qvol_train, qvol_test = parse_summary(qvol_summary)

    keys = ["mse", "rmse", "mae", "r2", "mape"]
    print("=== Model config (qiskit) ===")
    for k in ["n_qubits", "circuit_depth", "encoding", "entanglement", "lookback_window", "regressor"]:
        if k in qiskit_info:
            print(f"{k}: {qiskit_info[k]}")
    print("\n=== Model config (q-volution) ===")
    for k in ["n_qubits", "circuit_depth", "encoding", "entanglement", "lookback_window", "regressor"]:
        if k in qvol_info:
            print(f"{k}: {qvol_info[k]}")

    print("\n=== Metrics (train) ===")
    print("metric     qiskit        q-volution")
    for k in keys:
        print(f"{k:<9} {fmt(qiskit_train.get(k)):<12} {fmt(qvol_train.get(k)):<12}")

    print("\n=== Metrics (test) ===")
    print("metric     qiskit        q-volution")
    for k in keys:
        print(f"{k:<9} {fmt(qiskit_test.get(k)):<12} {fmt(qvol_test.get(k)):<12}")


if __name__ == "__main__":
    main()

