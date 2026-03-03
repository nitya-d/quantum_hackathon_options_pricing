import random
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import torch

import merlin as ML
from merlin import LexGrouping, MeasurementStrategy, ComputationSpace
from merlin.builder import CircuitBuilder
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

N_INPUT_MODES    = 5
N_MEMORY_MODES   = 3
N_MODES          = N_INPUT_MODES + N_MEMORY_MODES
N_PCA_COMPONENTS = N_INPUT_MODES
N_PHOTONS        = 3
N_GROUPED        = 8
CIRCUIT_DEPTH    = 3
LOOKBACK         = 5
RIDGE_ALPHA      = 0.1
EPS              = 1e-6

print(f"Config: {N_MODES} modes ({N_INPUT_MODES}+{N_MEMORY_MODES}), "
      f"{N_PHOTONS} photons, depth={CIRCUIT_DEPTH}, "
      f"Ridge alpha={RIDGE_ALPHA}")
# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

train_df = pd.read_excel("data/train.xlsx", index_col=0)
test_df  = pd.read_excel("data/test.xlsx", index_col=0)
feature_cols = train_df.columns.tolist()

print(f"Train: {train_df.shape}  ({train_df.index[0]} → {train_df.index[-1]})")
print(f"Test : {test_df.shape}   ({test_df.index[0]} → {test_df.index[-1]})")
print(f"\nLast 5 train dates : {train_df.index.tolist()[-5:]}")
print(f"Test dates         : {test_df.index.tolist()}")

# ─────────────────────────────────────────────
# 2. FIT SCALERS + PCA ON TRAINING DATA
# ─────────────────────────────────────────────

raw_train = train_df.values.astype(np.float32)
raw_test  = test_df.values.astype(np.float32)

# MinMax: fit on train, apply to both
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(raw_train).astype(np.float32)
test_scaled  = scaler.transform(raw_test).astype(np.float32)

# PCA: fit on train
pca = PCA(n_components=N_PCA_COMPONENTS)
train_pca = pca.fit_transform(train_scaled).astype(np.float32)
test_pca  = pca.transform(test_scaled).astype(np.float32)

# PCA scaler: fit on train
pca_scaler = MinMaxScaler()
train_pca = pca_scaler.fit_transform(train_pca).astype(np.float32)
test_pca  = pca_scaler.transform(test_pca).astype(np.float32)

# Angle encode: [0,1] → [-π, π]
train_enc = (train_pca * 2 * np.pi) - np.pi
test_enc  = (test_pca  * 2 * np.pi) - np.pi

print(f"PCA explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")
print(f"Train encoded shape   : {train_enc.shape}")
print(f"Test encoded shape    : {test_enc.shape}")

# ─────────────────────────────────────────────
# 3. BUILD CIRCUIT + LEXGROUPING  (same as training)
# ─────────────────────────────────────────────

builder = CircuitBuilder(n_modes=N_MODES)
builder.add_entangling_layer(trainable=False, name="U1")
builder.add_angle_encoding(modes=list(range(N_INPUT_MODES)), name="enc", scale=1.0)
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

torch.manual_seed(SEED)
grp = LexGrouping(ql.output_size, N_GROUPED)

print(f"Circuit: Fock space {ql.output_size} → LexGrouping {N_GROUPED} features/timestep")

# ─────────────────────────────────────────────
# 4. FEATURE EXTRACTION HELPER
# ─────────────────────────────────────────────

def extract_features(window: np.ndarray) -> np.ndarray:
    """window: (LOOKBACK, N_INPUT_MODES) angle-encoded PCA values.
    Returns: (TOTAL_FEATURES,) = 40 quantum + 25 classical.
    """
    quantum_feats = []
    for t in range(LOOKBACK):
        x_t = torch.tensor(window[t], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            fock    = ql(x_t)
            grouped = grp(fock)
        quantum_feats.append(grouped.squeeze(0).numpy())
    classical = window.flatten()
    return np.concatenate(quantum_feats + [classical])

# ─────────────────────────────────────────────
# 5. EXTRACT FEATURES FOR ALL TRAINING DATA
# ─────────────────────────────────────────────

N = len(train_enc)
X_list, Y_list = [], []

print(f"Extracting features from {N - LOOKBACK} training samples...")
for i in range(LOOKBACK, N):
    window = train_enc[i - LOOKBACK:i]
    X_list.append(extract_features(window))
    Y_list.append(train_scaled[i])
    if (i - LOOKBACK) % 100 == 0:
        print(f"  {i - LOOKBACK}/{N - LOOKBACK}...")

X_train = np.array(X_list, dtype=np.float32)
Y_train = np.array(Y_list, dtype=np.float32)
print(f"\nTraining features: X={X_train.shape}, Y={Y_train.shape}")

# ─────────────────────────────────────────────
# 6. FIT RIDGE ON FULL TRAINING SET
# ─────────────────────────────────────────────

print(f"Fitting Ridge(alpha={RIDGE_ALPHA}) on all {len(X_train)} training samples...")
ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
ridge.fit(X_train, Y_train)
print("Ridge fitted ✓")

# ─────────────────────────────────────────────
# 7. PREDICT TEST DAYS (rolling window)
# ─────────────────────────────────────────────
# Day 1 lookback: last 5 days of training data
# Day 2+ lookback: (last 5 - k days of training) + (first k actual test days)
# Teacher-forcing: use actual test values in the rolling context

N_TEST = len(test_enc)
test_preds_scaled = []

print("Predicting test days:")
for i in range(N_TEST):
    # Build the 5-day lookback window
    if i < LOOKBACK:
        # Take from end of train + beginning of test
        train_part = train_enc[N - (LOOKBACK - i):]
        test_part  = test_enc[:i]
        window = np.concatenate([train_part, test_part], axis=0)
    else:
        window = test_enc[i - LOOKBACK:i]

    assert window.shape == (LOOKBACK, N_INPUT_MODES), f"Window shape error: {window.shape}"

    feat = extract_features(window)
    pred_scaled = np.clip(ridge.predict(feat.reshape(1, -1)), 0.0, 1.0)
    test_preds_scaled.append(pred_scaled[0])

    print(f"  Day {i+1} ({test_df.index[i]}): lookback [{LOOKBACK-min(i,LOOKBACK)} train + {min(i,LOOKBACK)} test days]")

test_preds_scaled = np.array(test_preds_scaled, dtype=np.float32)

# Inverse-transform back to original vol space
test_preds = scaler.inverse_transform(test_preds_scaled)
test_true  = raw_test

print(f"\nPredictions shape: {test_preds.shape}")

# ─────────────────────────────────────────────
# 8. METRICS ON TEST SET
# ─────────────────────────────────────────────

def qlike_numpy(true: np.ndarray, pred: np.ndarray) -> float:
    pred  = np.clip(pred,  EPS, None)
    true  = np.clip(true,  EPS, None)
    ratio = true / pred
    return float(np.mean(ratio ** 2 - 2 * np.log(ratio) - 1))

rmse      = np.sqrt(mean_squared_error(test_true, test_preds))
mae       = np.mean(np.abs(test_true - test_preds))
qlike_val = qlike_numpy(test_true, test_preds)

print(f"{'='*56}")
print(f"HYBRID TEMPORAL QRC — TEST SET RESULTS (6 days)")
print(f"{'='*56}")
print(f"  RMSE  : {rmse:.6f}")
print(f"  MAE   : {mae:.6f}")
print(f"  QLIKE : {qlike_val:.6f}   <- primary metric")
print()

# Per-day breakdown
print("  Per-day QLIKE:")
for i, date in enumerate(test_df.index):
    q = qlike_numpy(test_true[i:i+1], test_preds[i:i+1])
    print(f"    {date}  QLIKE: {q:.6f}")

# ─────────────────────────────────────────────
# 9. SAVE PREDICTIONS
# ─────────────────────────────────────────────

pred_df = pd.DataFrame(test_preds, index=test_df.index, columns=feature_cols)
pred_df.index.name = "Date"
pred_df.to_csv("test_predictions.csv")
pred_df.to_excel("test_predictions.xlsx")

print(f"Saved predictions to test_predictions.csv / .xlsx")
print(f"Shape: {pred_df.shape}")
pred_df.head()