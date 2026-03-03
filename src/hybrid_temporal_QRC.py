'''
**Architecture:**
- 8 modes: 5 input + 3 memory (Li et al. arXiv:2505.13933)
- 3 photons, UNBUNCHED Fock space, depth-3 fixed circuit
- Single LexGrouping projection: Fock → 8 grouped features per timestep
- Raw PCA values concatenated as classical features (25)
- Ridge regression readout, alpha=0.1
'''

import random
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from datasets import load_dataset
import torch

import merlin as ML
from merlin import LexGrouping, MeasurementStrategy, ComputationSpace
from merlin.builder import CircuitBuilder

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

N_INPUT_MODES    = 5    # modes that receive angle-encoded PCA features
N_MEMORY_MODES   = 3    # modes that accumulate temporal context — never re-encoded
N_MODES          = N_INPUT_MODES + N_MEMORY_MODES   # = 8
N_PCA_COMPONENTS = N_INPUT_MODES                    # 1:1 with input modes = 5
N_PHOTONS        = 3
N_GROUPED        = 8    # LexGrouping output size — single projection, no ensemble
CIRCUIT_DEPTH    = 3    # single circuit, fixed depth
LOOKBACK         = 5    # days in the lookback window, processed sequentially
TRAIN_SPLIT      = 0.85
RIDGE_ALPHA      = 0.1  # tuned via alpha sweep (alpha=10 over-regularized quantum features)
EPS              = 1e-6

QUANTUM_FEATURES  = LOOKBACK * N_GROUPED              # 5 × 8 = 40
CLASSICAL_FEATURES= LOOKBACK * N_PCA_COMPONENTS       # 5 × 5 = 25
TOTAL_FEATURES    = QUANTUM_FEATURES + CLASSICAL_FEATURES  # 65

print(f"Config:")
print(f"  Circuit   : {N_MODES} modes ({N_INPUT_MODES} input + {N_MEMORY_MODES} memory), "
      f"{N_PHOTONS} photons, depth={CIRCUIT_DEPTH}")
print(f"  LexGrouping: Fock → {N_GROUPED} features per timestep (single projection)")
print(f"  Features  : {QUANTUM_FEATURES} quantum + {CLASSICAL_FEATURES} classical = {TOTAL_FEATURES}")
print(f"  Ridge alpha: {RIDGE_ALPHA}  (tuned — not default)")
print(f"  Lookback  : {LOOKBACK} timesteps (sequential encoding)")

# ─────────────────────────────────────────────
# 1. LOAD & PREPROCESS DATA
# ─────────────────────────────────────────────

print("Loading dataset...")
ds = pd.read_excel("data/train.xlsx", index_col=0)
df = ds.to_pandas()
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.sort_values("Date").reset_index(drop=True)      # chronological — never shuffle

feature_cols = [c for c in df.columns if c != "Date"]
raw_data     = df[feature_cols].values.astype(np.float32)
print(f"Raw data shape: {raw_data.shape}  ({raw_data.shape[0]} days, {raw_data.shape[1]} vol points)")

# MinMax scale to [0, 1] — needed to invert predictions back to vol space
scaler      = MinMaxScaler()
data_scaled = scaler.fit_transform(raw_data).astype(np.float32)

# PCA: 224 → 5 (one component per input mode)
pca      = PCA(n_components=N_PCA_COMPONENTS)
data_pca = pca.fit_transform(data_scaled).astype(np.float32)

# Rescale PCA output to [0, 1] before angle encoding
pca_scaler = MinMaxScaler()
data_pca   = pca_scaler.fit_transform(data_pca).astype(np.float32)

# Map [0, 1] → [-π, π] for angle encoding
data_pca_encoded = (data_pca * 2 * np.pi) - np.pi

print(f"PCA explained variance : {pca.explained_variance_ratio_.sum()*100:.1f}%")
print(f"Encoding range         : [{data_pca_encoded.min():.2f}, {data_pca_encoded.max():.2f}]")
# ─────────────────────────────────────────────
# 2. BUILD SINGLE FIXED CIRCUIT + LEXGROUPING
# ─────────────────────────────────────────────
# One circuit (depth=3), all parameters fixed (trainable=False).
# One LexGrouping projection: Fock space → N_GROUPED features.
# No ensemble, no multi-seed, fully deterministic.
#
# Layer structure:
#   Layer 1: Entangle(U1) + AngleEncode(input modes 0..4) + Rotate(R1)
#   Layer 2: Entangle(U2) + Rotate(R2)
#   Layer 3: Superposition(depth=1)
#
# Input modes  (0..4) : angle-encoded with PCA features at each timestep
# Memory modes (5..7) : entangled but never re-encoded — carry temporal state

builder = CircuitBuilder(n_modes=N_MODES)

builder.add_entangling_layer(trainable=False, name="U1")
builder.add_angle_encoding(
    modes=list(range(N_INPUT_MODES)),
    name="enc",
    scale=1.0,   # features already in [-π, π]
)
builder.add_rotations(trainable=False, name="R1")

builder.add_entangling_layer(trainable=False, name="U2")
builder.add_rotations(trainable=False, name="R2")

builder.add_superpositions(depth=1, trainable=False)

ql = ML.QuantumLayer(
    input_size=N_INPUT_MODES,
    builder=builder,
    n_photons=N_PHOTONS,
    measurement_strategy=MeasurementStrategy.probs(ComputationSpace.UNBUNCHED),
)

# Single LexGrouping: projects the full Fock distribution → N_GROUPED features
torch.manual_seed(SEED)
grp = LexGrouping(ql.output_size, N_GROUPED)

print(f"Circuit built ✓")
print(f"  Fock space size : {ql.output_size} unbunched states")
print(f"  LexGrouping     : {ql.output_size} → {N_GROUPED} features per timestep")
print(f"  Quantum features: {LOOKBACK} timesteps × {N_GROUPED} = {QUANTUM_FEATURES}")
print(f"  Classical feats : {CLASSICAL_FEATURES}  (raw PCA values)")
print(f"  Total features  : {TOTAL_FEATURES}")

# ─────────────────────────────────────────────
# 3. FEATURE EXTRACTION
# ─────────────────────────────────────────────
# For each 5-day lookback window:
#   For each timestep t (oldest → most recent):
#     Run circuit on x_t → Fock probabilities → LexGrouping → 8 features
#   Concatenate quantum features with raw PCA values (classical)
#
# Total per sample: LOOKBACK × N_GROUPED + CLASSICAL_FEATURES = 65

N = len(data_pca_encoded)
X_list, Y_list = [], []

print(f"Extracting features from {N - LOOKBACK} samples...")
print(f"  Per sample: {LOOKBACK} timesteps × {N_GROUPED} grouped + {CLASSICAL_FEATURES} classical = {TOTAL_FEATURES}")

for i in range(LOOKBACK, N):
    window = data_pca_encoded[i - LOOKBACK:i]   # (LOOKBACK, N_INPUT_MODES)

    quantum_feats = []
    for t in range(LOOKBACK):
        x_t = torch.tensor(window[t], dtype=torch.float32).unsqueeze(0)   # (1, 5)
        with torch.no_grad():
            fock    = ql(x_t)         # (1, fock_size) — Fock probability distribution
            grouped = grp(fock)        # (1, N_GROUPED)  — LexGrouping projection
        quantum_feats.append(grouped.squeeze(0).numpy())   # (N_GROUPED,)

    classical = window.flatten()   # (CLASSICAL_FEATURES,) — raw PCA values

    X_list.append(np.concatenate(quantum_feats + [classical]))
    Y_list.append(data_scaled[i])

    if (i - LOOKBACK) % 50 == 0:
        print(f"  Sample {i - LOOKBACK}/{N - LOOKBACK}...")

X = np.array(X_list, dtype=np.float32)
Y = np.array(Y_list, dtype=np.float32)

# Split for ablation cell reuse
X_quantum_only   = X[:, :QUANTUM_FEATURES]
X_classical_only = X[:, QUANTUM_FEATURES:]

print(f"\nDataset: X={X.shape}, Y={Y.shape}")

# ─────────────────────────────────────────────
# 4. TRAIN / VAL SPLIT  +  RIDGE FIT
# ─────────────────────────────────────────────
# Chronological split — never shuffle time series.

split = int(len(X) * TRAIN_SPLIT)
X_train, X_val = X[:split], X[split:]
Y_train, Y_val = Y[:split], Y[split:]

print(f"Train: {len(X_train)} samples | Val: {len(X_val)} samples")

# Ridge: analytical closed-form solution, same result every run
print(f"\nFitting Ridge(alpha={RIDGE_ALPHA})...")
ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
ridge.fit(X_train, Y_train)
print("Ridge fitted ✓")

# ─────────────────────────────────────────────
# 5. EVALUATE
# ─────────────────────────────────────────────

def qlike_numpy(true: np.ndarray, pred: np.ndarray) -> float:
    pred  = np.clip(pred,  EPS, None)
    true  = np.clip(true,  EPS, None)
    ratio = true / pred
    return float(np.mean(ratio ** 2 - 2 * np.log(ratio) - 1))

val_pred_scaled   = np.clip(ridge.predict(X_val), 0.0, 1.0)
val_pred_original = scaler.inverse_transform(val_pred_scaled)
val_true_original = scaler.inverse_transform(Y_val)

rmse      = np.sqrt(mean_squared_error(val_true_original, val_pred_original))
mae       = np.mean(np.abs(val_true_original - val_pred_original))
qlike_val = qlike_numpy(val_true_original, val_pred_original)

print(f"\n{'='*62}")
print(f"TEMPORAL QRC V2 — SINGLE CIRCUIT, FULL FOCK — RESULTS")
print(f"{'='*62}")
print(f"  Overall RMSE  : {rmse:.6f}")
print(f"  Overall MAE   : {mae:.6f}")
print(f"  Overall QLIKE : {qlike_val:.6f}   <- primary metric")

# Per-maturity breakdown
maturities_list = sorted(set(
    float(c.split("Maturity : ")[1]) for c in feature_cols
))
print(f"\n  Per-maturity QLIKE breakdown:")
for mat in maturities_list:
    mat_idx = [i for i, c in enumerate(feature_cols)
               if float(c.split("Maturity : ")[1]) == mat]
    q = qlike_numpy(val_true_original[:, mat_idx], val_pred_original[:, mat_idx])
    print(f"    Maturity {mat:5.2f}yr -> QLIKE: {q:.6f}")

    

