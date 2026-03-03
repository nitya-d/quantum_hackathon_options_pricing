"""
Test Evaluation — Hybrid Temporal QRC
======================================

Evaluates the QRC ensemble (from src/hybrid_temporal_QRC.py) against
the ground-truth test set in data/test.xlsx.

Usage
-----
  cd quantum_hackathon_options_pricing

  # Run the QRC pipeline, evaluate, and save results
  python test_eval.py

  # Load previously saved predictions (skip re-running the QRC)
  python test_eval.py --load-saved

  # Use the full seed sweep before evaluating
  python test_eval.py --sweep --seeds 30 --topk 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# ── Add src/ to path so we can import from hybrid_temporal_QRC ──
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from hybrid_temporal_QRC import (  # noqa: E402
    QRCConfig,
    load_and_preprocess,
    run_known_ensemble,
    run_sweep,
)

EPS = 1e-8


# ──────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────

def qlike(true: np.ndarray, pred: np.ndarray) -> float:
    """QLIKE loss — primary metric for volatility forecasting."""
    pred = np.clip(pred, EPS, None)
    true = np.clip(true, EPS, None)
    ratio = true / pred
    return float(np.mean(ratio - np.log(ratio) - 1))


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(true.flatten(), pred.flatten())))
    mae  = float(np.mean(np.abs(true - pred)))
    ql   = qlike(true, pred)
    return {"RMSE": rmse, "MAE": mae, "QLIKE": ql}


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="QRC Test Evaluation")
    ap.add_argument(
        "--load-saved", action="store_true",
        help="Load predictions from models/qrc_test_pred.npy instead of re-running",
    )
    ap.add_argument(
        "--sweep", action="store_true",
        help="Use full seed sweep (default: known-best ensemble seeds)",
    )
    ap.add_argument("--seeds", type=int, default=30,
                    help="Number of seeds for sweep (default: 30)")
    ap.add_argument("--topk", type=int, default=3,
                    help="Top-k seeds selected for ensemble (default: 3)")
    args = ap.parse_args()

    print()
    print("=" * 62)
    print("  QRC TEST EVALUATION")
    print("  Hybrid Temporal Quantum Reservoir Computing")
    print("=" * 62)

    # ── Load ground truth ──
    test_path = ROOT_DIR / "data" / "test.xlsx"
    test_df   = pd.read_excel(test_path, index_col=0)
    test_true = test_df.values.astype(np.float64)
    n_test    = len(test_df)
    feature_cols = test_df.columns.tolist()

    print(f"\n  Ground truth : {test_path.name}")
    print(f"  Shape        : {test_df.shape}")
    print(f"  Dates        : {test_df.index[0]} → {test_df.index[-1]}")

    # ── Get predictions ──
    save_path = ROOT_DIR / "models" / "qrc_test_pred.npy"

    if args.load_saved:
        if not save_path.exists():
            raise FileNotFoundError(
                f"No saved predictions found at {save_path}.\n"
                "Run without --load-saved to generate them first."
            )
        raw = np.load(str(save_path))
        test_preds = raw[:n_test]
        print(f"\n  Loaded saved predictions: {save_path.relative_to(ROOT_DIR)}")
        print(f"  Shape : {test_preds.shape}")

    else:
        cfg = QRCConfig(
            sweep_mode=args.sweep,
            sweep_n_seeds=args.seeds,
            ensemble_k=args.topk,
            n_test_days=n_test,
        )
        mode_label = "SWEEP" if args.sweep else "KNOWN-BEST ENSEMBLE"
        print(f"\n  Mode : {mode_label}")
        print(f"  n_modes={cfg.n_modes}  n_photons={cfg.n_photons}  "
              f"n_pca={cfg.n_pca}  n_steps={cfg.n_steps}")
        print()

        print("Loading & preprocessing data...")
        prices, prices_scaled, angles, scaler, _ = load_and_preprocess(cfg)
        print()

        if args.sweep:
            print("Running seed sweep...")
            results, ens_pred = run_sweep(cfg, angles, prices_scaled, scaler)
        else:
            print(f"Running {len(cfg.ensemble_seeds)} ensemble members "
                  f"(seeds {cfg.ensemble_seeds})...")
            results, ens_pred = run_known_ensemble(cfg, angles, prices_scaled, scaler)

        test_preds = ens_pred[:n_test]

        # Save for reuse
        save_path.parent.mkdir(exist_ok=True)
        np.save(str(save_path), ens_pred[:cfg.n_test_days])
        print(f"\n  Saved predictions → {save_path.relative_to(ROOT_DIR)}")

    # ── Sanity check ──
    if test_preds.shape != test_true.shape:
        raise ValueError(
            f"Shape mismatch: predictions={test_preds.shape} vs "
            f"ground truth={test_true.shape}"
        )

    # ──────────────────────────────────────────────────────────
    # OVERALL METRICS
    # ──────────────────────────────────────────────────────────
    overall = compute_metrics(test_preds, test_true)

    print()
    print("=" * 62)
    print("  OVERALL TEST METRICS")
    print("=" * 62)
    print(f"  RMSE  : {overall['RMSE']:.6f}")
    print(f"  MAE   : {overall['MAE']:.6f}")
    print(f"  QLIKE : {overall['QLIKE']:.6f}   ← primary metric")

    # ──────────────────────────────────────────────────────────
    # PER-DAY BREAKDOWN
    # ──────────────────────────────────────────────────────────
    print()
    print("  Per-day breakdown:")
    print(f"  {'Date':<20} {'RMSE':>10} {'MAE':>10} {'QLIKE':>10}")
    print("  " + "-" * 54)

    day_rows = []
    for i, date in enumerate(test_df.index):
        m = compute_metrics(test_preds[i : i + 1], test_true[i : i + 1])
        day_rows.append({"Date": date, **m})
        print(f"  {str(date):<20}  {m['RMSE']:>9.6f}  {m['MAE']:>9.6f}  {m['QLIKE']:>9.6f}")

    # ──────────────────────────────────────────────────────────
    # PER-INSTRUMENT SUMMARY
    # ──────────────────────────────────────────────────────────
    per_inst_qlike = np.array([
        qlike(test_true[:, j], test_preds[:, j])
        for j in range(test_true.shape[1])
    ])
    worst_idx = int(per_inst_qlike.argmax())
    best_idx  = int(per_inst_qlike.argmin())

    print()
    print("  Per-instrument QLIKE summary:")
    print(f"    mean   : {per_inst_qlike.mean():.6f}")
    print(f"    median : {np.median(per_inst_qlike):.6f}")
    print(f"    best   : {per_inst_qlike[best_idx]:.6f}  ({feature_cols[best_idx]})")
    print(f"    worst  : {per_inst_qlike[worst_idx]:.6f}  ({feature_cols[worst_idx]})")

    # ──────────────────────────────────────────────────────────
    # SAVE OUTPUTS
    # ──────────────────────────────────────────────────────────
    out_dir = ROOT_DIR / "results"
    out_dir.mkdir(exist_ok=True)

    # Predictions
    pred_df = pd.DataFrame(test_preds, index=test_df.index, columns=feature_cols)
    pred_df.index.name = "Date"
    pred_df.to_csv(str(out_dir / "test_predictions.csv"))
    pred_df.to_excel(str(out_dir / "test_predictions.xlsx"))

    # Per-day metrics
    metrics_df = pd.DataFrame(day_rows)
    metrics_df.to_csv(str(out_dir / "test_metrics.csv"), index=False)

    # Per-instrument QLIKE
    inst_df = pd.DataFrame({
        "Instrument": feature_cols,
        "QLIKE": per_inst_qlike,
    }).sort_values("QLIKE")
    inst_df.to_csv(str(out_dir / "per_instrument_qlike.csv"), index=False)

    print()
    print(f"  Saved → results/test_predictions.csv")
    print(f"  Saved → results/test_predictions.xlsx")
    print(f"  Saved → results/test_metrics.csv")
    print(f"  Saved → results/per_instrument_qlike.csv")
    print("=" * 62)
    print()


if __name__ == "__main__":
    main()
