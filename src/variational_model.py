import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datasets import load_dataset
import random

import merlin as ML
from merlin import LexGrouping, MeasurementStrategy, ComputationSpace
from merlin.builder import CircuitBuilder
N_PCA_COMPONENTS  = 16    # PCA: 224 → 16 (fits quantum mode limit)
LOOKBACK          = 5     # Days of history per sample → input = 5×16 = 80
N_MODES           = 16    # Quantum circuit modes (≤ 20 QPU hard limit)
N_PHOTONS         = 4     # Photons in the register
N_GROUPED_OUTPUTS = 32    # LexGrouping: compress Fock space → 32 features
TRAIN_SPLIT       = 0.85
EPOCHS            = 80
LR                = 5e-4
BATCH_SIZE        = 16
DEVICE            = torch.device("cpu")
# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

print("Loading dataset...")
ds = load_dataset(
    "Quandela/Challenge_Swaptions",
    data_files="level-1_Future_prediction/train.csv",
    split="train",
)
df = ds.to_pandas()
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.sort_values("Date").reset_index(drop=True)

feature_cols = [c for c in df.columns if c != "Date"]
raw_data = df[feature_cols].values.astype(np.float32)

print(f"Raw data shape: {raw_data.shape}")
# ─────────────────────────────────────────────
# 2. PREPROCESS
# ─────────────────────────────────────────────

# MinMax scale to [0, 1] — required for angle encoding stability (MerLin docs)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(raw_data).astype(np.float32)

# PCA: 224 → 16
pca = PCA(n_components=N_PCA_COMPONENTS)
data_pca = pca.fit_transform(data_scaled).astype(np.float32)

# Re-scale PCA outputs to [0, 1] so the pre-compression Sigmoid stays well-behaved
pca_scaler = MinMaxScaler()
data_pca = pca_scaler.fit_transform(data_pca).astype(np.float32)

print(f"PCA explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# ─────────────────────────────────────────────
# 3. BUILD LOOKBACK WINDOWS
# ─────────────────────────────────────────────
# Input:  last LOOKBACK days of PCA surface, flattened → (80,)
# Target: full 224-dim surface on the next day

N = len(data_pca)
X_list, Y_list = [], []
for i in range(LOOKBACK, N):
    window = data_pca[i - LOOKBACK:i].flatten()
    X_list.append(window)
    Y_list.append(data_scaled[i])

X = np.array(X_list, dtype=np.float32)   # (N-LOOKBACK, 80)
Y = np.array(Y_list, dtype=np.float32)   # (N-LOOKBACK, 224)

print(f"X shape: {X.shape}  ({LOOKBACK} days × {N_PCA_COMPONENTS} PCA = {X.shape[1]} features)")
print(f"Y shape: {Y.shape}")

# ─────────────────────────────────────────────
# 4. TRAIN / VAL SPLIT  (chronological — never shuffle time series)
# ─────────────────────────────────────────────

split    = int(len(X) * TRAIN_SPLIT)
X_train  = torch.tensor(X[:split], device=DEVICE)
Y_train  = torch.tensor(Y[:split], device=DEVICE)
X_val    = torch.tensor(X[split:], device=DEVICE)
Y_val    = torch.tensor(Y[split:], device=DEVICE)

train_loader = DataLoader(
    TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=False
)

print(f"\nTrain: {len(X_train)} samples | Val: {len(X_val)} samples")

builder = CircuitBuilder(n_modes=N_MODES)
builder.add_entangling_layer(trainable=True, name="U1")
builder.add_angle_encoding(
    modes=list(range(N_MODES)),
    name="input",
    scale=np.pi,
)
builder.add_rotations(trainable=True, name="theta")
builder.add_superpositions(depth=1, trainable=True)

quantum_core = ML.QuantumLayer(
    input_size=N_MODES,
    builder=builder,
    n_photons=N_PHOTONS,
    measurement_strategy=MeasurementStrategy.probs(ComputationSpace.UNBUNCHED),
)

print(f"\nQuantum layer Fock output size : {quantum_core.output_size}")
print(f"After LexGrouping              : {N_GROUPED_OUTPUTS}")

class QRCSwaption(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        # Classical pre-compression: squeeze lookback window → quantum-compatible size
        # Sigmoid ensures output stays in [0,1] for angle encoding
        self.pre_compress = nn.Sequential(
            nn.Linear(input_size, N_MODES),
            nn.Sigmoid(),
        )

        # Quantum feature extraction + Fock space grouping
        self.quantum = nn.Sequential(
            quantum_core,
            LexGrouping(quantum_core.output_size, N_GROUPED_OUTPUTS),
        )

        # Classical readout with regularization
        self.readout = nn.Sequential(
            nn.Linear(N_GROUPED_OUTPUTS, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(256, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_compress(x)   # (B, 80) → (B, 16)
        x = self.quantum(x)        # (B, 16) → (B, 32)
        return self.readout(x)     # (B, 32) → (B, 224)


model = QRCSwaption(
    input_size=N_PCA_COMPONENTS * LOOKBACK,
    output_size=len(feature_cols),
).to(DEVICE)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal params     : {total_params:,}")
print(f"Trainable params : {trainable_params:,}")

# ─────────────────────────────────────────────
# QLIKE LOSS FUNCTION
# ─────────────────────────────────────────────
#
# QLIKE (Quasi-Likelihood) is the standard loss for volatility forecasting.

# Formula (volatility form):
#   QLIKE(σ_true, σ_pred) = mean( (σ_true/σ_pred)² - 2·log(σ_true/σ_pred) - 1 )

EPS = 1e-6

def qlike_loss(pred: 'torch.Tensor', target: 'torch.Tensor') -> 'torch.Tensor':
    """
    QLIKE loss in volatility form, computed on MinMax-scaled values.
    Both pred and target are in [0,1].
    Returns a scalar tensor suitable for .backward().
    """
    pred   = pred.clamp(min=EPS)
    target = target.clamp(min=EPS)
    ratio  = target / pred
    return (ratio ** 2 - 2 * torch.log(ratio) - 1).mean()


def qlike_numpy(true: 'np.ndarray', pred: 'np.ndarray') -> float:
    """
    QLIKE in original volatility space — used for final evaluation reporting.
    """
    pred  = np.clip(pred,  EPS, None)
    true  = np.clip(true,  EPS, None)
    ratio = true / pred
    return float(np.mean(ratio ** 2 - 2 * np.log(ratio) - 1))


# Sanity check: perfect predictions should give QLIKE ≈ 0
x = torch.full((4, len(feature_cols)), 0.2)
assert qlike_loss(x, x).item() < 1e-4, "QLIKE(x,x) should be ≈ 0"
print("QLIKE loss ready ✓  (sanity check passed: QLIKE(x,x) ≈ 0)")

# ─────────────────────────────────────────────
# 7. TRAIN
# ─────────────────────────────────────────────

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Halve LR when val loss stops improving — helps avoid getting stuck
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=8, 
)

loss_fn = qlike_loss   # QLIKE: penalises underprediction more than overprediction
best_val_loss = float("inf")
best_state    = None

print("\nTraining...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(xb)

    epoch_loss /= len(X_train)

    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(X_val), Y_val).item()

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}


    print(f"  Epoch {epoch:3d}/{EPOCHS} | "
              f"train QLIKE: {epoch_loss:.6f} | "
              f"val QLIKE: {val_loss:.6f} | "
              f"best QLIKE: {best_val_loss:.6f}")

model.load_state_dict(best_state)
print(f"\nRestored best model (best QLIKE: {best_val_loss:.6f})")
# ─────────────────────────────────────────────
# 8. EVALUATE
# ─────────────────────────────────────────────

model.eval()
with torch.no_grad():
    val_pred_np = model(X_val).numpy()

val_true_np       = Y_val.numpy()
val_pred_original = scaler.inverse_transform(val_pred_np)
val_true_original = scaler.inverse_transform(val_true_np)

rmse = np.sqrt(mean_squared_error(val_true_original, val_pred_original))
mae  = np.mean(np.abs(val_true_original - val_pred_original))

print(f"\n{'='*55}")
print(f"VALIDATION RESULTS (original volatility scale)")
print(f"{'='*55}")
qlike_val = qlike_numpy(val_true_original, val_pred_original)

print(f"  Overall RMSE  : {rmse:.6f}")
print(f"  Overall MAE   : {mae:.6f}")
print(f"  Overall QLIKE : {qlike_val:.6f}   ← primary metric")
print(f"  (Volatility range ≈ 0.02 – 0.45)")

# Per-maturity QLIKE breakdown
# Short maturities (low vol) are weighted more heavily by QLIKE
maturities_list = sorted(set(
    float(c.split("Maturity : ")[1]) for c in feature_cols
))
print(f"\n  Per-maturity QLIKE breakdown:")
for mat in maturities_list:
    mat_idx = [i for i, c in enumerate(feature_cols)
               if float(c.split("Maturity : ")[1]) == mat]
    q = qlike_numpy(val_true_original[:, mat_idx], val_pred_original[:, mat_idx])
    print(f"    Maturity {mat:5.2f}yr → QLIKE: {q:.6f}")

last_window_raw    = raw_data[-LOOKBACK:]
last_window_scaled = scaler.transform(last_window_raw)
last_window_pca    = pca.transform(last_window_scaled)
last_window_pca    = pca_scaler.transform(last_window_pca).astype(np.float32)
last_window_flat   = last_window_pca.flatten()[np.newaxis, :]    # (1, 80)
last_day_tensor    = torch.tensor(last_window_flat, device=DEVICE)

model.eval()
with torch.no_grad():
    next_day_scaled = model(last_day_tensor).numpy()

next_day_pred = scaler.inverse_transform(next_day_scaled)
print(f"\nPredicted next-day surface (first 5 values): {next_day_pred[0, :5]}")
