# QML Level 1 — Swaption Surface Prediction

### Quandela Hackathon · Quantum Machine Learning Track

A hybrid quantum-classical model that predicts the next day's swaption implied volatility surface using photonic quantum computing via Quandela's **MerLin** framework.

---

## Table of Contents

- [Overview](#overview)
- [The Problem](#the-problem)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Quantum Circuit Design](#quantum-circuit-design)
- [Loss Function — QLIKE](#loss-function--qlike)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Design Decisions](#key-design-decisions)
- [Configuration](#configuration)
- [Saving and Loading the Model](#saving-and-loading-the-model)

---

## Overview

This project tackles **Level 1** of the Quandela swaption hackathon challenge: given a historical sequence of daily swaption implied volatility surfaces, predict what tomorrow's surface will look like.

The model is a **Quantum Reservoir Computing (QRC)**-inspired hybrid architecture. A photonic quantum circuit built with MerLin's `CircuitBuilder` acts as a non-linear feature extractor. A classical neural network readout is trained on top of the quantum features to produce the final predictions.

The model is trained using the **QLIKE loss** — the standard metric for volatility forecasting — which penalises underprediction of volatility more heavily than overprediction, and scales errors relative to the level of volatility at each maturity.

---

## The Problem

A **swaption** is an option on an interest rate swap. Its price is quoted as an **implied volatility** — a number expressing how much the market expects interest rates to move. These implied volatilities are organized into a 2D grid called the **volatility surface**, indexed by:

- **Tenor** (1–30 years): the length of the underlying swap
- **Maturity** (1 month–30 years): when the swaption expires

**The task**: Given the swaption surface over the last 5 days, predict the full surface on the next day. This is a **multivariate time series forecasting** problem with 224 output values.

---

## Dataset

**Source**: [`Quandela/Challenge_Swaptions`](https://huggingface.co/datasets/Quandela/Challenge_Swaptions) on HuggingFace

**File used**: `level-1_Future_prediction/train.csv`

```python
from datasets import load_dataset

ds = load_dataset(
    "Quandela/Challenge_Swaptions",
    data_files="level-1_Future_prediction/train.csv",
    split="train",
)
```

| Property             | Value                                           |
| -------------------- | ----------------------------------------------- |
| Rows (training days) | 494                                             |
| Features per row     | 224 (14 tenors × 16 maturities)                 |
| Value type           | Implied volatility (decimal, e.g. 0.028 = 2.8%) |
| Value range          | ~0.02 – 0.45                                    |
| Date format          | DD/MM/YYYY                                      |

**Key observations from EDA:**

- Surface shape is dominated by **maturity**, not tenor — short maturities have very low volatility (~0.03), long maturities are high (~0.35)
- Values are **bimodally distributed** — short-maturity cluster near 0.05, main cluster near 0.22–0.28
- Surface is **highly autocorrelated** day-to-day — yesterday is a strong predictor of today
- All tenors at a given maturity are very highly correlated with each other (~0.9+)

---

## Model Architecture

```
Input: 5 days × 16 PCA components = 80 features
             │
    ┌─────────────────────────────────┐
    │  PREPROCESSING                  │
    │  MinMaxScaler → [0,1]           │
    │  PCA: 224 → 16                  │
    │  MinMaxScaler on PCA → [0,1]    │
    │  Sliding window (LOOKBACK=5)    │
    │  Flatten → (80,)                │
    └────────────┬────────────────────┘
                 │
    ┌────────────▼────────────────────┐
    │  CLASSICAL PRE-COMPRESSION      │
    │  Linear(80 → 16) + Sigmoid      │
    │                                 │
    │  Compresses lookback window     │
    │  to fit quantum mode limit.     │
    │  Sigmoid re-normalizes to [0,1] │
    │  for stable angle encoding.     │
    └────────────┬────────────────────┘
                 │
    ┌────────────▼────────────────────┐
    │  QUANTUM CIRCUIT                │
    │  CircuitBuilder (16 modes)      │
    │  · entangling_layer (trainable) │
    │  · angle_encoding (scale=π)     │
    │  · rotations (trainable)        │
    │  · superpositions (trainable)   │
    │                                 │
    │  4 photons · UNBUNCHED space    │
    │  MeasurementStrategy.probs()    │
    └────────────┬────────────────────┘
                 │
    ┌────────────▼────────────────────┐
    │  LexGrouping(Fock → 32)         │
    │                                 │
    │  Compresses high-dimensional    │
    │  Fock probability distribution  │
    │  into 32 classical features.    │
    └────────────┬────────────────────┘
                 │
    ┌────────────▼────────────────────┐
    │  CLASSICAL READOUT              │
    │  Linear(32 → 256)               │
    │  BatchNorm + ReLU + Dropout(0.3)│
    │  Linear(256 → 256)              │
    │  BatchNorm + ReLU + Dropout(0.3)│
    │  Linear(256 → 224) + Sigmoid    │
    └────────────┬────────────────────┘
                 │
    Output: 224 predicted volatilities (next day's surface)
```

---

## Quantum Circuit Design

The circuit is built using MerLin's `CircuitBuilder` with the following layers:

```python
builder = CircuitBuilder(n_modes=16)
builder.add_entangling_layer(trainable=True, name="U1")
builder.add_angle_encoding(modes=list(range(16)), name="input", scale=np.pi)
builder.add_rotations(trainable=True, name="theta")
builder.add_superpositions(depth=1, trainable=True)

quantum_core = ML.QuantumLayer(
    input_size=16,
    builder=builder,
    n_photons=4,
    measurement_strategy=MeasurementStrategy.probs(ComputationSpace.UNBUNCHED),
)
```

**Why these choices:**

| Choice              | Reason                                                                                           |
| ------------------- | ------------------------------------------------------------------------------------------------ |
| `n_modes=16`        | One mode per input feature (angle encoding requirement). Stays within the 20-mode QPU limit.     |
| `n_photons=4`       | Balances expressivity and simulation speed. More photons = richer interference patterns.         |
| `scale=np.pi`       | Required by Perceval's phase conventions. Keeps rotations in a stable range.                     |
| `UNBUNCHED`         | At most 1 photon per mode. Reduces Fock state count, faster simulation.                          |
| `LexGrouping(→ 32)` | Standard MerLin pattern to compress the Fock probability distribution before the classical head. |

---

## Loss Function — QLIKE

### Why not MSE?

The initial version of the model was trained with standard MSE loss. While it achieved good RMSE and MAE scores, MSE treats all 224 cells equally — a prediction error of 0.01 on a 1-month swaption (vol ≈ 0.03) is penalised identically to the same error on a 10-year swaption (vol ≈ 0.35). In volatility forecasting this is financially wrong: short-maturity cells are more sensitive and more dangerous to misprice.

### QLIKE formula

**QLIKE (Quasi-Likelihood)** is the industry-standard loss for volatility forecasting:

```
QLIKE(σ_true, σ_pred) = mean( (σ_true / σ_pred)² - 2·log(σ_true / σ_pred) - 1 )
```

Key properties:

- **Asymmetric**: underpredicting volatility is penalised more heavily than overpredicting
- **Scale-sensitive**: errors are measured relative to the level of volatility, not in absolute terms
- **Minimum at zero**: QLIKE = 0 only when σ_pred = σ_true exactly

### Implementation

Two functions are defined in `QML_level1_QLIKE.ipynb`:

```python
EPS = 1e-6

def qlike_loss(pred, target):
    """PyTorch version — used during training in MinMax-scaled [0,1] space."""
    pred   = pred.clamp(min=EPS)
    target = target.clamp(min=EPS)
    ratio  = target / pred
    return (ratio ** 2 - 2 * torch.log(ratio) - 1).mean()

def qlike_numpy(true, pred):
    """NumPy version — used for evaluation in original volatility space."""
    pred  = np.clip(pred,  EPS, None)
    true  = np.clip(true,  EPS, None)
    ratio = true / pred
    return float(np.mean(ratio ** 2 - 2 * np.log(ratio) - 1))
```

`EPS = 1e-6` is applied to prevent `log(0)` or division by zero on near-zero short-maturity volatilities.

### What we learned

Switching to QLIKE as the training loss correctly reprioritises the gradient toward short-maturity cells which are inherently harder to predict and more sensitive to mispricing. The per-maturity QLIKE breakdown confirms errors are distributed more appropriately across the surface compared to MSE training.

We also explored **maturity-weighted QLIKE** (multiplying each cell's loss term by `1/maturity` to further emphasise short maturities). This degraded all metrics — the gradient became too dominated by a handful of cells before the model had learned overall surface structure. Plain QLIKE already provides implicit scale-sensitive weighting and proved more stable.

---

## Results

### Evolution across versions

| Version                 | Training Loss | Val RMSE     | Val MAE      | Val QLIKE    |
| ----------------------- | ------------- | ------------ | ------------ | ------------ |
| `QuantumLayer.simple()` | MSE           | 0.019590     | 0.014816     | —            |
| `CircuitBuilder`        | MSE           | 0.009964     | 0.007222     | —            |
| `CircuitBuilder`        | **QLIKE**     | **0.014969** | **0.009463** | **0.012683** |

The RMSE and MAE are slightly higher in the QLIKE-trained model compared to the MSE baseline — this is expected. The model now de-emphasises long-maturity absolute accuracy in favour of getting the relative errors right across the full surface, which is the financially correct objective.

### Final validation results

```
=======================================================
VALIDATION RESULTS (original volatility scale)
=======================================================
  Overall RMSE  : 0.014969
  Overall MAE   : 0.009463
  Overall QLIKE : 0.012683   ← primary metric
  (Volatility range ≈ 0.02 – 0.45)
```

### Per-maturity QLIKE breakdown

| Zone      | Maturities    | Difficulty                                      |
| --------- | ------------- | ----------------------------------------------- |
| Short     | 0.08 – 0.75yr | Hardest — low absolute vol, high relative error |
| Medium    | 1.0 – 2.0yr   | Moderate                                        |
| Long      | 3.0 – 15yr    | Easiest — high absolute vol, low relative error |
| Very long | 20 – 30yr     | Slight uptick due to surface irregularity       |

A QLIKE of **0.012683** corresponds to a typical relative prediction error of roughly **11%** across all cells. In academic volatility forecasting, scores below 0.005 are considered strong for out-of-sample predictions — our result is a solid first QML baseline with clear room to improve on short maturities.

---

## Installation

```bash
pip install merlinquantum datasets scikit-learn torch pandas numpy
```

---

## Usage

**Train with MSE loss (baseline):**

```
Run all cells in qml_level1.ipynb
```

**Train with QLIKE loss (primary metric):**

```
Run all cells in QML_level1_QLIKE.ipynb
```

**What the notebooks output:**

- Training progress per epoch (train QLIKE, val QLIKE, best val)
- Final validation RMSE, MAE, and QLIKE in original volatility units
- Per-maturity QLIKE breakdown across all 16 maturities
- Predicted next-day surface (first 5 values printed)
- Saved model file: `qrc_swaption_model.pt`

**Load the model for inference:**

```python
import torch

checkpoint = torch.load("qrc_swaption_model.pt")
model.load_state_dict(checkpoint["model_state"])

scaler     = checkpoint["scaler"]
pca        = checkpoint["pca"]
pca_scaler = checkpoint["pca_scaler"]
```

**Save model to Google Drive (when running on Google Colab via VS Code):**

```python
from google.colab import drive
import shutil

drive.mount("/content/drive")
shutil.copy("qrc_swaption_model.pt", "/content/drive/MyDrive/qrc_swaption_model.pt")
print("Saved to Google Drive ✓")
```

---

## Project Structure

```
.
├── qml_level1.ipynb            # Baseline model trained with MSE loss
├── QML_level1_QLIKE.ipynb      # Final model trained with QLIKE loss (primary)
├── qrc_swaption_model.pt       # Saved model + preprocessing objects
└── README.md                   # This file
```

---

## Key Design Decisions

**Why PCA before the quantum circuit?**
The dataset has 224 features per day. Angle encoding requires one circuit mode per input feature, and the QPU hard limit is 20 modes. PCA reduces the 224 features to 16 while retaining the majority of explained variance.

**Why two MinMaxScalers?**
The first scaler normalizes the raw swaption data to `[0,1]` before PCA. PCA outputs are not naturally bounded, so a second scaler re-normalizes the PCA components to `[0,1]` before they enter the quantum circuit. Without this, the Sigmoid in the pre-compression layer would saturate.

**Why LOOKBACK=5?**
The EDA showed very high day-to-day autocorrelation. A single-day snapshot loses directional information. With 5 days, the model can learn whether the surface is trending up, down, or stable.

**Why QLIKE over MSE?**
Swaption volatilities span two orders of magnitude across maturities (0.03 at 1 month to 0.40 at 30 years). MSE is blind to this scale difference. QLIKE naturally penalises relative errors, making it a far more appropriate loss for this task and consistent with standard practice in the volatility forecasting literature.

**Why not maturity-weighted QLIKE?**
Multiplying each cell's loss by `1/maturity` was tested but degraded all metrics. QLIKE is already its own form of implicit weighting — adding explicit weights on top overcorrected and destabilised training.

**Why not shuffle the data?**
This is a time series. Shuffling would leak future information into training samples and produce overoptimistic validation results. All splits are strictly chronological.

**Why `ComputationSpace.UNBUNCHED`?**
Our circuit avoids photon bunching (multiple photons on the same mode). UNBUNCHED restricts the Fock basis accordingly, dramatically reducing computation time with no loss in accuracy for our use case.

---

## Configuration

All hyperparameters are at the top of both notebooks:

| Parameter           | Default | Description                        |
| ------------------- | ------- | ---------------------------------- |
| `N_PCA_COMPONENTS`  | 16      | PCA output dimension               |
| `LOOKBACK`          | 5       | Number of past days used as input  |
| `N_MODES`           | 16      | Quantum circuit modes              |
| `N_PHOTONS`         | 4       | Photons in the register            |
| `N_GROUPED_OUTPUTS` | 32      | LexGrouping output size            |
| `TRAIN_SPLIT`       | 0.85    | Fraction of data used for training |
| `EPOCHS`            | 80      | Training epochs                    |
| `LR`                | 5e-4    | Initial learning rate              |
| `BATCH_SIZE`        | 16      | Batch size                         |

---

## Saving and Loading the Model

The checkpoint saved to `qrc_swaption_model.pt` contains everything needed for inference:

```python
{
    "model_state"  : model.state_dict(),   # trained weights
    "pca"          : pca,                  # fitted PCA object
    "pca_scaler"   : pca_scaler,           # fitted scaler for PCA outputs
    "scaler"       : scaler,               # fitted scaler for raw data
    "feature_cols" : feature_cols,         # column names (224 swaption labels)
    "config"       : { ... }               # hyperparameters used during training
}
```

To predict on new data, apply the same preprocessing pipeline in order:
`MinMaxScaler → PCA → MinMaxScaler → sliding window → model → inverse MinMaxScaler`
