"""
QML Model — Level 1: Future Swaption Surface Prediction
Version 3 — Barren Plateau Fixes

Two improvements from the literature to address the initialization instability
that causes most runs to land at QLIKE ~0.020 instead of ~0.012:

  FIX 1 — Identity-Block Initialization
    Quantum rotation parameters (theta) are initialized near zero so the
    circuit starts close to the identity transformation. The optimizer then
    builds complexity gradually from a well-behaved starting point, avoiding
    the flat gradient regions (barren plateaus) that trap random initializations.
    Source: Grant et al. (2019) "An initialization strategy for addressing
    barren plateaus in parametrized quantum circuits"

  FIX 2 — Layerwise (Two-Stage) Training
    Stage 1: Freeze all quantum parameters. Train only classical layers
             (pre_compress + readout) until convergence (~40 epochs).
             This gives the classical parts a strong, meaningful signal
             before the quantum circuit starts receiving gradients.
    Stage 2: Unfreeze everything. Fine-tune the full model end-to-end
             at a lower learning rate (1e-4) for up to 80 more epochs.
    Rationale: when the readout is still random, the gradients flowing
    back into the quantum circuit are pure noise — which causes barren
    plateaus to form. Warming up the classical layers first gives the
    quantum circuit a clean gradient signal from day one of Stage 2.
"""

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datasets import load_dataset

import merlin as ML
from merlin import LexGrouping, MeasurementStrategy, ComputationSpace
from merlin.builder import CircuitBuilder

# ─────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
print(f"Random seed fixed: {SEED}")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

N_PCA_COMPONENTS  = 16
LOOKBACK          = 5
N_MODES           = 16
N_PHOTONS         = 4
N_GROUPED_OUTPUTS = 32
TRAIN_SPLIT       = 0.85
BATCH_SIZE        = 16
DEVICE            = torch.device("cpu")

# Two-stage training config
STAGE1_EPOCHS     = 40     # classical warmup (quantum frozen)
STAGE1_LR         = 5e-4
STAGE2_EPOCHS     = 80     # full fine-tuning (all params)
STAGE2_LR         = 1e-4   # lower LR for fine-tuning
EARLY_STOP        = 15     # patience for early stopping in stage 2

# Identity-block init config
THETA_INIT        = 0.01   # near-zero rotation angles → circuit ≈ identity at start

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

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(raw_data).astype(np.float32)

pca = PCA(n_components=N_PCA_COMPONENTS)
data_pca = pca.fit_transform(data_scaled).astype(np.float32)

pca_scaler = MinMaxScaler()
data_pca = pca_scaler.fit_transform(data_pca).astype(np.float32)

print(f"PCA explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# ─────────────────────────────────────────────
# 3. BUILD LOOKBACK WINDOWS
# ─────────────────────────────────────────────

N = len(data_pca)
X_list, Y_list = [], []
for i in range(LOOKBACK, N):
    window = data_pca[i - LOOKBACK:i].flatten()
    X_list.append(window)
    Y_list.append(data_scaled[i])

X = np.array(X_list, dtype=np.float32)   # (N-LOOKBACK, 80)
Y = np.array(Y_list, dtype=np.float32)   # (N-LOOKBACK, 224)

print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")

# ─────────────────────────────────────────────
# 4. TRAIN / VAL SPLIT
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

# ─────────────────────────────────────────────
# 5. QUANTUM CIRCUIT
# ─────────────────────────────────────────────

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

# ─────────────────────────────────────────────
# 6. MODEL
# ─────────────────────────────────────────────

class QRCSwaption(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        self.pre_compress = nn.Sequential(
            nn.Linear(input_size, N_MODES),
            nn.Sigmoid(),
        )

        self.quantum = nn.Sequential(
            quantum_core,
            LexGrouping(quantum_core.output_size, N_GROUPED_OUTPUTS),
        )

        self.readout = nn.Sequential(
            nn.Linear(N_GROUPED_OUTPUTS, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(256, len(feature_cols)),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_compress(x)
        x = self.quantum(x)
        return self.readout(x)

    def freeze_quantum(self):
        """Freeze all quantum parameters for Stage 1 warmup."""
        for name, param in self.named_parameters():
            if any(k in name for k in ["quantum", "U1", "theta", "superpos"]):
                param.requires_grad = False
        frozen = sum(1 for p in self.parameters() if not p.requires_grad)
        print(f"  Quantum params frozen ({frozen} tensors)")

    def unfreeze_all(self):
        """Unfreeze everything for Stage 2 fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        print(f"  All params unfrozen")


model = QRCSwaption(
    input_size=N_PCA_COMPONENTS * LOOKBACK,
    output_size=len(feature_cols),
).to(DEVICE)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal params     : {total_params:,}")
print(f"Trainable params : {trainable_params:,}")

# ─────────────────────────────────────────────
# FIX 1 — IDENTITY-BLOCK INITIALIZATION
# ─────────────────────────────────────────────
#
# Initialize all rotation parameters (theta, U1, superpositions) near zero.
# When rotation angles are ~0, the circuit acts approximately as the identity:
# photons pass through with minimal interference.
#
# Why this helps: random initialization scatters parameters across the loss
# landscape, often landing in flat barren plateau regions where gradients
# vanish. Starting near zero keeps the initial circuit in a well-behaved
# region where gradients are non-zero and training can begin immediately.

print("\nApplying identity-block initialization to quantum parameters...")
n_init = 0
with torch.no_grad():
    for name, param in model.named_parameters():
        if any(k in name for k in ["theta", "U1", "superpos", "rotation"]):
            param.fill_(THETA_INIT)
            n_init += 1
            print(f"  Initialized {name:<45} → {THETA_INIT}")

print(f"  {n_init} quantum parameter tensors initialized near zero ✓")

# ─────────────────────────────────────────────
# QLIKE LOSS
# ─────────────────────────────────────────────

EPS = 1e-6

def qlike_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred   = pred.clamp(min=EPS)
    target = target.clamp(min=EPS)
    ratio  = target / pred
    return (ratio ** 2 - 2 * torch.log(ratio) - 1).mean()

def qlike_numpy(true: np.ndarray, pred: np.ndarray) -> float:
    pred  = np.clip(pred,  EPS, None)
    true  = np.clip(true,  EPS, None)
    ratio = true / pred
    return float(np.mean(ratio ** 2 - 2 * np.log(ratio) - 1))

x = torch.full((4, len(feature_cols)), 0.2)
assert qlike_loss(x, x).item() < 1e-4
print("\nQLIKE loss ready ✓")

# ─────────────────────────────────────────────
# FIX 2 — LAYERWISE TRAINING
# ─────────────────────────────────────────────
#
# STAGE 1: Classical warmup — quantum frozen
#   Train only pre_compress + readout for STAGE1_EPOCHS epochs.
#   When the readout is random, gradients flowing into the quantum
#   circuit are pure noise. Warming up the classical layers first
#   means the quantum circuit receives meaningful gradients in Stage 2.
#
# STAGE 2: Full fine-tuning — all parameters
#   Unfreeze the quantum parameters and train end-to-end at a lower
#   learning rate. The classical layers are already well-initialized,
#   so the quantum circuit can learn from a stable gradient signal.

# ── STAGE 1: Classical warmup ──────────────────

print(f"\n{'='*55}")
print(f"STAGE 1 — Classical warmup (quantum frozen)")
print(f"  Epochs : {STAGE1_EPOCHS}  |  LR : {STAGE1_LR}")
print(f"{'='*55}")

model.freeze_quantum()

optimizer_s1 = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=STAGE1_LR
)
scheduler_s1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_s1, mode='min', factor=0.5, patience=8,
)

best_val_s1   = float("inf")
best_state_s1 = None

for epoch in range(1, STAGE1_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        optimizer_s1.zero_grad()
        pred = model(xb)
        loss = qlike_loss(pred, yb)
        loss.backward()
        optimizer_s1.step()
        epoch_loss += loss.item() * len(xb)

    epoch_loss /= len(X_train)

    model.eval()
    with torch.no_grad():
        val_loss = qlike_loss(model(X_val), Y_val).item()

    scheduler_s1.step(val_loss)

    if val_loss < best_val_s1:
        best_val_s1   = val_loss
        best_state_s1 = {k: v.clone() for k, v in model.state_dict().items()}

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{STAGE1_EPOCHS} | "
              f"train QLIKE: {epoch_loss:.6f} | "
              f"val QLIKE: {val_loss:.6f} | "
              f"best: {best_val_s1:.6f}")

model.load_state_dict(best_state_s1)
print(f"\nStage 1 complete — best val QLIKE: {best_val_s1:.6f}")

# ── STAGE 2: Full fine-tuning ──────────────────

print(f"\n{'='*55}")
print(f"STAGE 2 — Full fine-tuning (all params unfrozen)")
print(f"  Epochs : up to {STAGE2_EPOCHS}  |  LR : {STAGE2_LR}")
print(f"  Early stopping patience : {EARLY_STOP}")
print(f"{'='*55}")

model.unfreeze_all()

optimizer_s2 = torch.optim.Adam(model.parameters(), lr=STAGE2_LR)
scheduler_s2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_s2, mode='min', factor=0.5, patience=8,
)

best_val_s2       = best_val_s1   # start from Stage 1 best
best_state_s2     = {k: v.clone() for k, v in model.state_dict().items()}
epochs_no_improve = 0

for epoch in range(1, STAGE2_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        optimizer_s2.zero_grad()
        pred = model(xb)
        loss = qlike_loss(pred, yb)
        loss.backward()
        optimizer_s2.step()
        epoch_loss += loss.item() * len(xb)

    epoch_loss /= len(X_train)

    model.eval()
    with torch.no_grad():
        val_loss = qlike_loss(model(X_val), Y_val).item()

    scheduler_s2.step(val_loss)

    if val_loss < best_val_s2:
        best_val_s2       = val_loss
        best_state_s2     = {k: v.clone() for k, v in model.state_dict().items()}
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{STAGE2_EPOCHS} | "
              f"train QLIKE: {epoch_loss:.6f} | "
              f"val QLIKE: {val_loss:.6f} | "
              f"best QLIKE: {best_val_s2:.6f} | "
              f"no_improve: {epochs_no_improve}/{EARLY_STOP}")

    if epochs_no_improve >= EARLY_STOP:
        print(f"\n  Early stopping at epoch {epoch}")
        break

model.load_state_dict(best_state_s2)
print(f"\nStage 2 complete — best val QLIKE: {best_val_s2:.6f}")
print(f"\nOverall improvement: {best_val_s1:.6f} (classical only) → "
      f"{best_val_s2:.6f} (full model)")

# ─────────────────────────────────────────────
# 8. EVALUATE
# ─────────────────────────────────────────────

model.eval()
with torch.no_grad():
    val_pred_np = model(X_val).numpy()

val_true_np       = Y_val.numpy()
val_pred_original = scaler.inverse_transform(val_pred_np)
val_true_original = scaler.inverse_transform(val_true_np)

rmse      = np.sqrt(mean_squared_error(val_true_original, val_pred_original))
mae       = np.mean(np.abs(val_true_original - val_pred_original))
qlike_val = qlike_numpy(val_true_original, val_pred_original)

print(f"\n{'='*55}")
print(f"VALIDATION RESULTS (original volatility scale)")
print(f"{'='*55}")
print(f"  Overall RMSE  : {rmse:.6f}")
print(f"  Overall MAE   : {mae:.6f}")
print(f"  Overall QLIKE : {qlike_val:.6f}   <- primary metric")
print(f"  (Volatility range ~ 0.02 - 0.45)")

maturities_list = sorted(set(
    float(c.split("Maturity : ")[1]) for c in feature_cols
))
print(f"\n  Per-maturity QLIKE breakdown:")
for mat in maturities_list:
    mat_idx = [i for i, c in enumerate(feature_cols)
               if float(c.split("Maturity : ")[1]) == mat]
    q = qlike_numpy(val_true_original[:, mat_idx], val_pred_original[:, mat_idx])
    print(f"    Maturity {mat:5.2f}yr -> QLIKE: {q:.6f}")

# ─────────────────────────────────────────────
# 9. PREDICT NEXT DAY
# ─────────────────────────────────────────────

last_window_raw    = raw_data[-LOOKBACK:]
last_window_scaled = scaler.transform(last_window_raw)
last_window_pca    = pca.transform(last_window_scaled)
last_window_pca    = pca_scaler.transform(last_window_pca).astype(np.float32)
last_window_flat   = last_window_pca.flatten()[np.newaxis, :]
last_day_tensor    = torch.tensor(last_window_flat, device=DEVICE)

model.eval()
with torch.no_grad():
    next_day_scaled = model(last_day_tensor).numpy()

next_day_pred = scaler.inverse_transform(next_day_scaled)
print(f"\nPredicted next-day surface (first 5 values): {next_day_pred[0, :5]}")
