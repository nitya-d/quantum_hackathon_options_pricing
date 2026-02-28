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

| Property | Value |
|---|---|
| Rows (training days) | 494 |
| Features per row | 224 (14 tenors × 16 maturities) |
| Value type | Implied volatility (decimal, e.g. 0.028 = 2.8%) |
| Value range | ~0.02 – 0.45 |
| Date format | DD/MM/YYYY |

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

| Choice | Reason |
|---|---|
| `n_modes=16` | One mode per input feature (angle encoding requirement). Stays within the 20-mode QPU limit. |
| `n_photons=4` | Balances expressivity and simulation speed. More photons = richer interference patterns. |
| `scale=np.pi` | Required by Perceval's phase conventions. Keeps rotations in a stable range. |
| `UNBUNCHED` | At most 1 photon per mode. Reduces Fock state count, faster simulation. |
| `LexGrouping(→ 32)` | Standard MerLin pattern to compress the Fock probability distribution before the classical head. |

---

## Results

| Metric | Old version (`QuantumLayer.simple`) | Current version (`CircuitBuilder`) |
|---|---|---|
| Val RMSE | 0.019590 | **0.009964** |
| Val MAE | 0.014816 | **0.007222** |

The current model's MAE of **0.0072** means that on average, predictions are off by 0.72 volatility points — roughly **1.5% of the full volatility range** (0.02–0.45). This represents a ~2× improvement over the initial version.

**What drove the improvement:**
- Explicit `CircuitBuilder` with richer circuit structure vs. the generic `QuantumLayer.simple()`
- `scale=π` on angle encoding for phase stability
- `LexGrouping` to properly compress the Fock output
- `LOOKBACK=5` giving the model the surface's temporal trajectory
- Dropout + BatchNorm in the classical readout reducing overfitting
- `ReduceLROnPlateau` scheduler for better convergence

---

## Installation

```bash
pip install merlinquantum datasets scikit-learn torch pandas numpy
```

---

## Usage

**Train the model:**
```bash
python qml_level1.py
```

**What the script outputs:**
- Training progress every 10 epochs (train MSE, val MSE, best val)
- Final validation RMSE and MAE in original volatility units
- Predicted next-day surface (first 5 values printed)
- Saved model file: `qrc_swaption_model.pt`

**Find the saved model:**
```bash
find . -name "qrc_swaption_model.pt"
```

**Load the model for inference:**
```python
import torch

checkpoint = torch.load("qrc_swaption_model.pt")
model.load_state_dict(checkpoint["model_state"])

# Preprocessing objects are also saved
scaler     = checkpoint["scaler"]
pca        = checkpoint["pca"]
pca_scaler = checkpoint["pca_scaler"]
```

---

## Project Structure

```
.
├── qml_level1.py           # Main training script
├── qrc_swaption_model.pt   # Saved model + preprocessing objects (generated after training)
├── analyse_swaptions.py    # EDA script (generates analysis plots)
├── swaption_analysis.png   # EDA output plots
└── README.md               # This file
```

---

## Key Design Decisions

**Why PCA before the quantum circuit?**
The dataset has 224 features per day. Angle encoding requires one circuit mode per input feature, and the QPU hard limit is 20 modes. PCA reduces the 224 features to 16 while retaining the majority of explained variance.

**Why two MinMaxScalers?**
The first scaler normalizes the raw swaption data to `[0,1]` before PCA. PCA outputs are not naturally bounded, so a second scaler re-normalizes the PCA components to `[0,1]` before they enter the quantum circuit. Without this, the Sigmoid in the pre-compression layer would saturate.

**Why LOOKBACK=5?**
The EDA showed very high day-to-day autocorrelation. A single-day snapshot loses directional information. With 5 days, the model can learn whether the surface is trending up, down, or stable. Neighboring tenors are also highly correlated, so multi-day patterns are spatially consistent.

**Why not shuffle the data?**
This is a time series. Shuffling would leak future information into training samples and produce overoptimistic validation results. All splits are strictly chronological.

**Why `ComputationSpace.UNBUNCHED`?**
Our circuit avoids photon bunching (multiple photons on the same mode). UNBUNCHED restricts the Fock basis accordingly, dramatically reducing computation time with no loss in accuracy for our use case.

---

## Configuration

All hyperparameters are in the `CONFIG` section at the top of `qml_level1.py`:

| Parameter | Default | Description |
|---|---|---|
| `N_PCA_COMPONENTS` | 16 | PCA output dimension |
| `LOOKBACK` | 5 | Number of past days used as input |
| `N_MODES` | 16 | Quantum circuit modes |
| `N_PHOTONS` | 4 | Photons in the register |
| `N_GROUPED_OUTPUTS` | 32 | LexGrouping output size |
| `TRAIN_SPLIT` | 0.85 | Fraction of data used for training |
| `EPOCHS` | 80 | Training epochs |
| `LR` | 5e-4 | Initial learning rate |
| `BATCH_SIZE` | 16 | Batch size |

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