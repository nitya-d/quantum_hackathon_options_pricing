"""
Quantum Reservoir Computing — Swaption Price Forecasting
=========================================================

Combined best practices from all team members:

  Nitya   — Memory-loop temporal encoding with hidden modes
             Sequential day encoding through the circuit preserves temporal
             correlations via persistent hidden-mode interference.

  Tomoya  — Clean OOP config with hardware-constraint validation
             Dataclass config, MZI-style entangling layers, proper limits.

  Aziz    — PCA dimensionality reduction + angle encoding pipeline
             224 instruments → N_PCA components scaled to [0, π].

Architecture
------------
  [Entangle] → [Encode day t-2 on input modes]  →
  [Entangle] → [Encode day t-1 on input modes]  →  ← t-2 persists in hidden modes
  [Entangle] → [Encode day t   on input modes]  →
  [Entangle] → [Measure mode expectations]       → n_modes features

The quantum circuit is FIXED (random parameters, frozen).
Zero trainable quantum params — only a Ridge readout is trained.
Multi-seed ensemble: average predictions from top-k reservoirs (by val QLIKE).

Produces
--------
  models/qrc_test_pred.npy  (n_test_days, 224)
  → Loaded by notebooks/05_Test_Evaluation.ipynb for test metrics.

⚠  test.xlsx is NEVER loaded or referenced here.

Usage
-----
  cd quantum_hackathon_options_pricing
  python src/qrc.py                  # default: run top-3 known seeds
  python src/qrc.py --sweep          # full 30-seed × 3-window sweep (~12 min)
  python src/qrc.py --sweep --seeds 50 --topk 5   # custom sweep
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from merlin import (
    CircuitBuilder,
    ComputationSpace,
    MeasurementStrategy,
    QuantumLayer,
)

# Resolve project root from script location
ROOT_DIR = Path(__file__).resolve().parent.parent


# ──────────────────────────────────────────────────────────
# 1. CONFIGURATION  (Tomoya-style dataclass + constraint checks)
# ──────────────────────────────────────────────────────────
@dataclass
class QRCConfig:
    """
    Configuration for the QRC pipeline.

    Hardware constraints (Quandela photonic QPU):
      Simulation : ≤ 20 modes, ≤ 10 photons
      QPU        : ≤ 24 modes, ≤ 12 photons
      Encoding   : angle / phase only (no amplitude encoding)
    """

    # ── Data ──
    data_path: str = "data/level1.parquet"
    save_dir: str = "models"
    window: int = 20          # lookback window (trading days)
    horizon: int = 10         # forecast horizon (days)
    val_size: int = 30        # held-out validation days (chronological)

    # ── Quantum circuit ──
    n_modes: int = 10         # photonic modes
    n_photons: int = 5        # photon count
    n_pca: int = 4            # PCA components → input modes [0..n_pca-1]
    n_steps: int = 3          # temporal depth (consecutive days per evaluation)

    # ── Ensemble (default: known-best seeds from sweep) ──
    ensemble_seeds: List[int] = field(default_factory=lambda: [5, 15, 16])
    ensemble_windows: List[int] = field(default_factory=lambda: [20, 20, 20])
    n_test_days: int = 6      # forecast days to save for test evaluation

    # ── Sweep (optional, activated with --sweep) ──
    sweep_mode: bool = False
    sweep_n_seeds: int = 30
    sweep_windows: List[int] = field(default_factory=lambda: [15, 20, 25])
    ensemble_k: int = 3       # top-k members for ensemble selection

    def __post_init__(self) -> None:
        assert self.n_modes <= 20, f"n_modes={self.n_modes} exceeds sim limit of 20"
        assert self.n_photons <= 10, f"n_photons={self.n_photons} exceeds sim limit of 10"
        assert self.n_pca <= self.n_modes, f"n_pca must be ≤ n_modes"
        self.input_modes: List[int] = list(range(self.n_pca))
        self.hidden_modes: List[int] = list(range(self.n_pca, self.n_modes))


# ──────────────────────────────────────────────────────────
# 2. DATA PREPARATION  (Aziz's PCA + our train-only fitting)
# ──────────────────────────────────────────────────────────
def load_and_preprocess(
    cfg: QRCConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, List[str]]:
    """
    Load swaption surface, fit StandardScaler + PCA on TRAIN split only,
    scale PCA output to [0, π] for angle encoding.

    Returns (prices, prices_scaled, angles, scaler, price_cols).
    """
    path = ROOT_DIR / cfg.data_path
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"], format="mixed")
    df = df.sort_values("Date").reset_index(drop=True)
    price_cols = [c for c in df.columns if c != "Date"]
    prices = df[price_cols].astype(float).values  # (494, 224)

    # StandardScaler — fit on TRAIN only
    scaler = StandardScaler()
    scaler.fit(prices[: -cfg.val_size])
    prices_scaled = scaler.transform(prices)

    # PCA: 224 → n_pca — fit on TRAIN only
    pca = PCA(n_components=cfg.n_pca, random_state=42)
    pca.fit(prices_scaled[: -cfg.val_size])
    prices_pca = pca.transform(prices_scaled)

    # Angle encoding: [0, π] — min/max from TRAIN only
    pca_min = prices_pca[: -cfg.val_size].min(axis=0)
    pca_max = prices_pca[: -cfg.val_size].max(axis=0)
    angles = (prices_pca - pca_min) / (pca_max - pca_min + 1e-8) * np.pi

    print(f"  Data shape       : {prices.shape}")
    print(f"  PCA variance     : {pca.explained_variance_ratio_.sum():.3f}")
    print(f"  Angle range      : [{angles.min():.3f}, {angles.max():.3f}] rad")
    print(f"  Train / Val split: {len(prices) - cfg.val_size} / {cfg.val_size}")

    return prices, prices_scaled, angles, scaler, price_cols


# ──────────────────────────────────────────────────────────
# 3. RESERVOIR BUILDER  (Nitya's memory-loop architecture)
# ──────────────────────────────────────────────────────────
def build_reservoir(cfg: QRCConfig, seed: int) -> QuantumLayer:
    """
    Build a fixed photonic reservoir with memory-loop encoding.

    For n_steps=3 the circuit is:
      [Entangle] → [Encode day t−2 on input modes 0..3]   →
      [Entangle] → [Encode day t−1 on input modes 0..3]   →  ← t−2 in hidden 4..9
      [Entangle] → [Encode day t   on input modes 0..3]   →
      [Entangle] → [MODE_EXPECTATIONS]  → n_modes features

    Key trick: ``trainable=True`` so parameters get registered by MerLin,
    then we randomise with ``uniform_(0, 2π)`` and freeze with
    ``requires_grad = False``.  (trainable=False → params default to 0
    → identity interferometers → dead features.)
    """
    builder = CircuitBuilder(n_modes=cfg.n_modes)

    for _ in range(cfg.n_steps):
        builder.add_entangling_layer(trainable=True)
        builder.add_angle_encoding(modes=cfg.input_modes)
    builder.add_entangling_layer(trainable=True)  # final mixing

    reservoir = QuantumLayer(
        input_size=cfg.n_steps * cfg.n_pca,
        builder=builder,
        n_photons=cfg.n_photons,
        measurement_strategy=MeasurementStrategy.MODE_EXPECTATIONS,
        computation_space=ComputationSpace.UNBUNCHED,
        dtype=torch.float64,
    )

    # RANDOMIZE then FREEZE — true reservoir (random but fixed)
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in reservoir.parameters():
            p.uniform_(0, 2 * np.pi)
            p.requires_grad = False

    return reservoir


# ──────────────────────────────────────────────────────────
# 4. FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────
def extract_features(
    reservoir: QuantumLayer, angles: np.ndarray, cfg: QRCConfig
) -> np.ndarray:
    """Pass every overlapping n_steps-day block through the fixed circuit."""
    n_blocks = len(angles) - cfg.n_steps + 1
    q_dim = reservoir.output_size
    features = np.empty((n_blocks, q_dim))

    with torch.no_grad():
        for i in range(n_blocks):
            block = angles[i : i + cfg.n_steps].flatten()
            x = torch.tensor(block[np.newaxis, :], dtype=torch.float64)
            features[i] = reservoir(x).numpy().flatten()

    return features


# ──────────────────────────────────────────────────────────
# 5. WINDOWED DATASET + METRICS
# ──────────────────────────────────────────────────────────
def build_windows(
    q_feats: np.ndarray,
    targets_scaled: np.ndarray,
    window: int,
    horizon: int,
    offset: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X, Y) pairs from quantum features and scaled targets."""
    X, Y = [], []
    for i in range(len(q_feats) - window - horizon + 1):
        X.append(q_feats[i : i + window].flatten())
        start = i + offset + window
        Y.append(targets_scaled[start : start + horizon].flatten())
    return np.array(X), np.array(Y)


def compute_metrics(pred: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
    """RMSE, MAE, and QLIKE (financial loss metric for volatility forecasting)."""
    rmse = float(np.sqrt(((pred - actual) ** 2).mean()))
    mae = float(np.abs(pred - actual).mean())
    eps = 1e-8
    ratio = actual / np.clip(pred, eps, None)
    qlike = float((ratio - np.log(ratio) - 1).mean())
    return {"RMSE": rmse, "MAE": mae, "QLIKE": qlike}


# ──────────────────────────────────────────────────────────
# 6. TRAIN + EVALUATE ONE RESERVOIR
# ──────────────────────────────────────────────────────────
def run_single(
    cfg: QRCConfig,
    seed: int,
    window: int,
    angles: np.ndarray,
    prices_scaled: np.ndarray,
    scaler: StandardScaler,
) -> Tuple[Dict[str, float], np.ndarray, float]:
    """
    Build one reservoir, extract features, train RidgeCV, return:
      (val_metrics, raw_10day_prediction, ridge_alpha)
    """
    reservoir = build_reservoir(cfg, seed)
    features = extract_features(reservoir, angles, cfg)

    X_all, Y_all = build_windows(
        features, prices_scaled, window, cfg.horizon, offset=cfg.n_steps - 1
    )

    n_total = len(prices_scaled)
    val_start = n_total - cfg.val_size
    train_end = val_start - window - cfg.n_steps + 1 - cfg.horizon + 1

    X_train, Y_train = X_all[:train_end], Y_all[:train_end]
    X_val, Y_val = X_all[train_end:], Y_all[train_end:]

    ridge = RidgeCV(alphas=np.logspace(-3, 6, 50), scoring="neg_mean_squared_error")
    ridge.fit(X_train, Y_train)

    # Validation metrics (original scale)
    Y_val_hat = ridge.predict(X_val)
    vp = [
        scaler.inverse_transform(Y_val_hat[i].reshape(cfg.horizon, 224))
        for i in range(len(Y_val_hat))
    ]
    vt = [
        scaler.inverse_transform(Y_val[i].reshape(cfg.horizon, 224))
        for i in range(len(Y_val))
    ]
    val_metrics = compute_metrics(np.concatenate(vp), np.concatenate(vt))

    # Raw 10-day prediction (original scale) for ensembling
    pred_input = features[-window:].flatten().reshape(1, -1)
    raw_pred = scaler.inverse_transform(
        ridge.predict(pred_input).reshape(cfg.horizon, 224)
    )

    return val_metrics, raw_pred, float(ridge.alpha_)


# ──────────────────────────────────────────────────────────
# 7. ENSEMBLE RUNNERS
# ──────────────────────────────────────────────────────────
def run_known_ensemble(
    cfg: QRCConfig,
    angles: np.ndarray,
    prices_scaled: np.ndarray,
    scaler: StandardScaler,
) -> Tuple[List[Dict], np.ndarray]:
    """Run the pre-selected best seeds and average their predictions."""
    results: List[Dict] = []
    preds: List[np.ndarray] = []

    for seed, win in zip(cfg.ensemble_seeds, cfg.ensemble_windows):
        t1 = time.time()
        print(f"  seed={seed:>2d}  W={win}  ", end="", flush=True)
        val_m, raw_pred, alpha = run_single(
            cfg, seed, win, angles, prices_scaled, scaler
        )
        elapsed = time.time() - t1
        results.append({"Seed": seed, "Window": win, "Alpha": alpha, **val_m})
        preds.append(raw_pred)
        print(f"val_Q={val_m['QLIKE']:.6f}  α={alpha:.4f}  ({elapsed:.1f}s)")

    ens_pred = np.mean(preds, axis=0)
    return results, ens_pred


def run_sweep(
    cfg: QRCConfig,
    angles: np.ndarray,
    prices_scaled: np.ndarray,
    scaler: StandardScaler,
) -> Tuple[List[Dict], np.ndarray]:
    """Full seed sweep: discover best seeds, build ensemble from top-k."""
    warnings.filterwarnings("ignore")

    n_configs = cfg.sweep_n_seeds * len(cfg.sweep_windows)
    print(f"  Sweep: {cfg.sweep_n_seeds} seeds × {len(cfg.sweep_windows)} windows"
          f" = {n_configs} configs\n")

    all_results: List[Dict] = []
    all_preds: Dict[str, np.ndarray] = {}

    for si, seed in enumerate(range(cfg.sweep_n_seeds)):
        t1 = time.time()
        for win in cfg.sweep_windows:
            try:
                val_m, raw_pred, alpha = run_single(
                    cfg, seed, win, angles, prices_scaled, scaler
                )
                key = f"seed={seed} W={win}"
                all_results.append(
                    {"Key": key, "Seed": seed, "Window": win, "Alpha": alpha, **val_m}
                )
                all_preds[key] = raw_pred
            except Exception as e:
                print(f"    seed={seed} W={win} FAILED: {e}")
        elapsed = time.time() - t1
        best_so_far = min(r["QLIKE"] for r in all_results) if all_results else float("inf")
        print(
            f"  [{si + 1:>2d}/{cfg.sweep_n_seeds}] seed={seed:>2d}  "
            f"({elapsed:.1f}s)  best_Q={best_so_far:.6f}"
        )

    # Rank by val QLIKE
    df = pd.DataFrame(all_results).sort_values("QLIKE")
    print(f"\n  Top {min(10, len(df))} by Val QLIKE:")
    print(df.head(10).to_string(index=False))

    # Build ensemble from top-k
    top_keys = df.nsmallest(cfg.ensemble_k, "QLIKE")["Key"].tolist()
    ens_pred = np.mean([all_preds[k] for k in top_keys], axis=0)

    # Convert to results format
    top_results = df.nsmallest(cfg.ensemble_k, "QLIKE").to_dict("records")

    print(f"\n  Ensemble top-{cfg.ensemble_k}: {top_keys}")

    return top_results, ens_pred


# ──────────────────────────────────────────────────────────
# 8. MAIN
# ──────────────────────────────────────────────────────────
def main(cfg: Optional[QRCConfig] = None) -> None:
    if cfg is None:
        cfg = QRCConfig()

    t0 = time.time()

    print()
    print("=" * 62)
    print("  QRC — Quantum Reservoir Computing")
    print("  Swaption Price Forecasting  (Mil'HaQ Fest — Quandela Track)")
    print("=" * 62)
    print()
    print(f"  Modes={cfg.n_modes}  Photons={cfg.n_photons}  "
          f"PCA={cfg.n_pca}  Steps={cfg.n_steps}")
    print(f"  Window={cfg.window}  Horizon={cfg.horizon}  "
          f"Val={cfg.val_size} days")
    if cfg.sweep_mode:
        print(f"  Mode: SWEEP ({cfg.sweep_n_seeds} seeds × "
              f"{len(cfg.sweep_windows)} windows)")
    else:
        print(f"  Mode: ENSEMBLE (seeds {cfg.ensemble_seeds})")
    print()

    # ── Load & preprocess ──
    print("Loading data...")
    prices, prices_scaled, angles, scaler, price_cols = load_and_preprocess(cfg)
    print()

    # ── Run reservoir(s) ──
    if cfg.sweep_mode:
        print("Running seed sweep...")
        results, ens_pred = run_sweep(cfg, angles, prices_scaled, scaler)
    else:
        print(f"Running {len(cfg.ensemble_seeds)} ensemble members...")
        results, ens_pred = run_known_ensemble(cfg, angles, prices_scaled, scaler)

    # ── Ensemble stats ──
    avg_qlike = np.mean([r["QLIKE"] for r in results])
    avg_rmse = np.mean([r["RMSE"] for r in results])

    # ── Save ──
    save_path = ROOT_DIR / cfg.save_dir / "qrc_test_pred.npy"
    save_path.parent.mkdir(exist_ok=True)
    np.save(str(save_path), ens_pred[: cfg.n_test_days])

    elapsed = time.time() - t0

    # ── Report ──
    print()
    print("=" * 62)
    print(f"  ENSEMBLE  ({len(results)} members)")
    print("-" * 62)
    for r in results:
        s = r.get("Seed", r.get("Key", "?"))
        w = r.get("Window", "?")
        print(f"    seed={s!s:>2s}  W={w}  val_Q={r['QLIKE']:.6f}  "
              f"val_RMSE={r['RMSE']:.6f}  α={r['Alpha']:.4f}")
    print("-" * 62)
    print(f"  Avg val QLIKE : {avg_qlike:.6f}")
    print(f"  Avg val RMSE  : {avg_rmse:.6f}")
    print(f"  Saved         : {save_path.relative_to(ROOT_DIR)}"
          f"  shape={ens_pred[:cfg.n_test_days].shape}")
    print(f"  Total time    : {elapsed:.1f}s")
    print()
    print("  → Run notebooks/05_Test_Evaluation.ipynb for test metrics")
    print("  ⚠  test.xlsx was NEVER loaded in this script")
    print("=" * 62)


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────
def parse_args() -> QRCConfig:
    ap = argparse.ArgumentParser(
        description="QRC — Quantum Reservoir Computing for Swaption Forecasting"
    )
    ap.add_argument("--sweep", action="store_true",
                    help="Run full seed sweep instead of known-best ensemble")
    ap.add_argument("--seeds", type=int, default=30,
                    help="Number of seeds to sweep (default: 30)")
    ap.add_argument("--topk", type=int, default=3,
                    help="Top-k members for ensemble (default: 3)")
    ap.add_argument("--modes", type=int, default=10,
                    help="Number of photonic modes (default: 10, max 20)")
    ap.add_argument("--photons", type=int, default=5,
                    help="Number of photons (default: 5, max 10)")
    ap.add_argument("--pca", type=int, default=4,
                    help="PCA components / input modes (default: 4)")
    ap.add_argument("--steps", type=int, default=3,
                    help="Temporal depth — days per circuit eval (default: 3)")

    args = ap.parse_args()

    cfg = QRCConfig(
        n_modes=args.modes,
        n_photons=args.photons,
        n_pca=args.pca,
        n_steps=args.steps,
        sweep_mode=args.sweep,
        sweep_n_seeds=args.seeds,
        ensemble_k=args.topk,
    )
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)