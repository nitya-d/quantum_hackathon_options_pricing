"""
Classical Baseline Models — Level 1: Future Swaption Surface Prediction
Two architectures for fair comparison with the hybrid quantum model:

  1. MLP  — direct classical replacement of the quantum layer
             same preprocessing, same readout, same QLIKE loss
             only difference: Linear(16→32)+ReLU instead of QuantumLayer+LexGrouping

  2. LSTM — explicitly models temporal structure of the surface time series
             processes the LOOKBACK window as a sequence, not a flat vector
             stronger classical baseline since it's architecturally better suited

Both models use:
  - Identical preprocessing (MinMaxScaler, PCA 224→16, LOOKBACK=5)
  - Identical train/val split (chronological, 85/15)
  - Identical QLIKE loss
  - Identical readout (256→256→224)
  - SEED=42 for reproducibility
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

# ─────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
print(f"Random seed fixed: {SEED}")

# ─────────────────────────────────────────────
# CONFIG  (identical to quantum model)
# ─────────────────────────────────────────────

N_PCA_COMPONENTS  = 16
LOOKBACK          = 5
TRAIN_SPLIT       = 0.85
EPOCHS            = 150
LR                = 5e-4
BATCH_SIZE        = 16
DEVICE            = torch.device("cpu")

# LSTM-specific
LSTM_HIDDEN       = 64    # hidden size per LSTM layer
LSTM_LAYERS       = 2     # stacked LSTM layers

# ─────────────────────────────────────────────
# 1. LOAD DATA  (identical to quantum model)
# ─────────────────────────────────────────────
print("Loading dataset...")
ds = pd.read_excel("data/train.xlsx", index_col=0)
df = ds.to_pandas()
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.sort_values("Date").reset_index(drop=True)

feature_cols = [c for c in df.columns if c != "Date"]
raw_data = df[feature_cols].values.astype(np.float32)
print(f"Raw data shape: {raw_data.shape}")

# ─────────────────────────────────────────────
# 2. PREPROCESS  (identical to quantum model)
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
# MLP input : flattened window → (LOOKBACK*N_PCA, )  = (80,)
# LSTM input : sequential window → (LOOKBACK, N_PCA) = (5, 16)
# Target     : full 224-dim surface on the next day

N = len(data_pca)
X_flat_list, X_seq_list, Y_list = [], [], []
for i in range(LOOKBACK, N):
    window = data_pca[i - LOOKBACK:i]            # (5, 16)
    X_flat_list.append(window.flatten())         # (80,)  for MLP
    X_seq_list.append(window)                    # (5,16) for LSTM
    Y_list.append(data_scaled[i])                # (224,)

X_flat = np.array(X_flat_list, dtype=np.float32)  # (N, 80)
X_seq  = np.array(X_seq_list,  dtype=np.float32)  # (N, 5, 16)
Y      = np.array(Y_list,      dtype=np.float32)  # (N, 224)

print(f"MLP  input shape : {X_flat.shape}")
print(f"LSTM input shape : {X_seq.shape}")
print(f"Target shape     : {Y.shape}")

# ─────────────────────────────────────────────
# 4. TRAIN / VAL SPLIT  (chronological)
# ─────────────────────────────────────────────

split = int(len(Y) * TRAIN_SPLIT)

X_flat_train = torch.tensor(X_flat[:split], device=DEVICE)
X_flat_val   = torch.tensor(X_flat[split:], device=DEVICE)
X_seq_train  = torch.tensor(X_seq[:split],  device=DEVICE)
X_seq_val    = torch.tensor(X_seq[split:],  device=DEVICE)
Y_train      = torch.tensor(Y[:split],      device=DEVICE)
Y_val        = torch.tensor(Y[split:],      device=DEVICE)

mlp_loader  = DataLoader(TensorDataset(X_flat_train, Y_train),
                         batch_size=BATCH_SIZE, shuffle=False)
lstm_loader = DataLoader(TensorDataset(X_seq_train,  Y_train),
                         batch_size=BATCH_SIZE, shuffle=False)

print(f"\nTrain: {len(Y_train)} samples | Val: {len(Y_val)} samples")
# ─────────────────────────────────────────────
# QLIKE LOSS  (identical to quantum model)
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

# ─────────────────────────────────────────────
# TRAINING UTILITY
# ─────────────────────────────────────────────

def train_model(model, loader, X_val_tensor, name):
    """Generic training loop with early stopping — same for both baselines."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8,
    )
    best_val_loss     = float("inf")
    best_state        = None
    early_stop        = 15
    epochs_no_improve = 0

    print(f"\nTraining {name}...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for *xb_parts, yb in loader:
            xb = xb_parts[0]
            optimizer.zero_grad()
            pred = model(xb)
            loss = qlike_loss(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(yb)

        epoch_loss /= len(Y_train)

        model.eval()
        with torch.no_grad():
            val_loss = qlike_loss(model(X_val_tensor), Y_val).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            best_state        = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                  f"train QLIKE: {epoch_loss:.6f} | "
                  f"val QLIKE: {val_loss:.6f} | "
                  f"best: {best_val_loss:.6f} | "
                  f"no_improve: {epochs_no_improve}/{early_stop}")

        if epochs_no_improve >= early_stop:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    print(f"  Best val QLIKE: {best_val_loss:.6f}")
    return model

# ─────────────────────────────────────────────
# MODEL 1 — MLP BASELINE
# ─────────────────────────────────────────────
#
# Direct classical replacement of the quantum layer.
# Quantum block was: QuantumLayer(16→Fock) + LexGrouping(→32)
# MLP replacement  : Linear(16→32) + ReLU
# Everything else is identical to the quantum model.
#
#   [80]  flattened lookback window
#     |
#   Linear(80→16) + Sigmoid      <- same pre-compression as quantum model
#     |
#   Linear(16→32) + ReLU         <- classical replacement of quantum+LexGrouping
#     |
#   Linear(32→256) + BN + ReLU + Dropout(0.3)
#   Linear(256→256) + BN + ReLU + Dropout(0.3)
#   Linear(256→224) + Sigmoid

class MLPBaseline(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        self.pre_compress = nn.Sequential(
            nn.Linear(input_size, N_PCA_COMPONENTS),
            nn.Sigmoid(),
        )

        # Classical replacement of quantum+LexGrouping
        self.classical_block = nn.Sequential(
            nn.Linear(N_PCA_COMPONENTS, 32),
            nn.ReLU(),
        )

        self.readout = nn.Sequential(
            nn.Linear(32, 256),
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
        x = self.pre_compress(x)      # (B, 80) -> (B, 16)
        x = self.classical_block(x)   # (B, 16) -> (B, 32)
        return self.readout(x)        # (B, 32) -> (B, 224)


mlp_model = MLPBaseline(
    input_size=N_PCA_COMPONENTS * LOOKBACK,
    output_size=len(feature_cols),
).to(DEVICE)

mlp_params = sum(p.numel() for p in mlp_model.parameters())
print(f"\nMLP params: {mlp_params:,}")

mlp_model = train_model(mlp_model, mlp_loader, X_flat_val, "MLP Baseline")
# ─────────────────────────────────────────────
# MODEL 2 — LSTM BASELINE
# ─────────────────────────────────────────────
#
# A stronger classical baseline — the LSTM explicitly models temporal
# dependencies in the sequence, rather than treating the lookback window
# as a flat vector. This is architecturally better suited to time series
# than both MLP and the quantum model.
#
#   [5, 16]  sequential lookback window (LOOKBACK, N_PCA)
#     |
#   LSTM(input=16, hidden=64, layers=2, dropout=0.3)
#     |
#   Take last hidden state → (B, 64)
#     |
#   Linear(64→256) + BN + ReLU + Dropout(0.3)
#   Linear(256→256) + BN + ReLU + Dropout(0.3)
#   Linear(256→224) + Sigmoid

class LSTMBaseline(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,     # 16 PCA features per timestep
            hidden_size=LSTM_HIDDEN,   # 64
            num_layers=LSTM_LAYERS,    # 2 stacked layers
            batch_first=True,
            dropout=0.3,
        )

        self.readout = nn.Sequential(
            nn.Linear(LSTM_HIDDEN, 256),
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
        # x: (B, LOOKBACK, N_PCA) = (B, 5, 16)
        out, _ = self.lstm(x)          # out: (B, 5, 64)
        last   = out[:, -1, :]         # take last timestep: (B, 64)
        return self.readout(last)      # (B, 224)


lstm_model = LSTMBaseline(
    input_size=N_PCA_COMPONENTS,
    output_size=len(feature_cols),
).to(DEVICE)

lstm_params = sum(p.numel() for p in lstm_model.parameters())
print(f"\nLSTM params: {lstm_params:,}")

lstm_model = train_model(lstm_model, lstm_loader, X_seq_val, "LSTM Baseline")
# ─────────────────────────────────────────────
# EVALUATION — BOTH MODELS
# ─────────────────────────────────────────────

def evaluate(model, X_val_tensor, name):
    model.eval()
    with torch.no_grad():
        val_pred_np = model(X_val_tensor).numpy()

    val_true_np       = Y_val.numpy()
    val_pred_original = scaler.inverse_transform(val_pred_np)
    val_true_original = scaler.inverse_transform(val_true_np)

    rmse      = np.sqrt(mean_squared_error(val_true_original, val_pred_original))
    mae       = np.mean(np.abs(val_true_original - val_pred_original))
    qlike_val = qlike_numpy(val_true_original, val_pred_original)

    print(f"\n{'='*55}")
    print(f"{name} — VALIDATION RESULTS")
    print(f"{'='*55}")
    print(f"  Overall RMSE  : {rmse:.6f}")
    print(f"  Overall MAE   : {mae:.6f}")
    print(f"  Overall QLIKE : {qlike_val:.6f}   <- primary metric")

    maturities_list = sorted(set(
        float(c.split("Maturity : ")[1]) for c in feature_cols
    ))
    print(f"\n  Per-maturity QLIKE breakdown:")
    for mat in maturities_list:
        mat_idx = [i for i, c in enumerate(feature_cols)
                   if float(c.split("Maturity : ")[1]) == mat]
        q = qlike_numpy(val_true_original[:, mat_idx], val_pred_original[:, mat_idx])
        print(f"    Maturity {mat:5.2f}yr -> QLIKE: {q:.6f}")

    return {"rmse": rmse, "mae": mae, "qlike": qlike_val}


mlp_results  = evaluate(mlp_model,  X_flat_val, "MLP Baseline")
lstm_results = evaluate(lstm_model, X_seq_val,  "LSTM Baseline")

# ─────────────────────────────────────────────
# SAVE MODELS + PREPROCESSING
# ─────────────────────────────────────────────
from pathlib import Path
import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent
models_dir = ROOT_DIR / "models"
models_dir.mkdir(exist_ok=True)

torch.save(mlp_model.state_dict(),  str(models_dir / "mlp_best.pt"))
torch.save(lstm_model.state_dict(), str(models_dir / "lstm_best.pt"))

# Save preprocessing so test_evaluation.py can reconstruct identical transforms
joblib.dump({
    "scaler": scaler,
    "pca": pca,
    "pca_scaler": pca_scaler,
    "feature_cols": feature_cols,
    "config": {
        "N_PCA_COMPONENTS": N_PCA_COMPONENTS,
        "LOOKBACK": LOOKBACK,
        "LSTM_HIDDEN": LSTM_HIDDEN,
        "LSTM_LAYERS": LSTM_LAYERS,
    },
}, str(models_dir / "classical_preprocessing.pkl"))

print(f"\nSaved → models/mlp_best.pt")
print(f"Saved → models/lstm_best.pt")
print(f"Saved → models/classical_preprocessing.pkl")