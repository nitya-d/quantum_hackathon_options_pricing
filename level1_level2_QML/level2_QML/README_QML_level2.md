# QML Level 2 — Swaption Surface Missing Cell Reconstruction
### Quandela Hackathon · Quantum Machine Learning Track

A hybrid quantum-classical masked autoencoder that reconstructs arbitrary missing cells in a swaption implied volatility surface using photonic quantum computing via Quandela's **MerLin** framework.

---

## Table of Contents

- [Overview](#overview)
- [The Problem](#the-problem)
- [Dataset](#dataset)
- [Key Design Insight](#key-design-insight)
- [Model Architecture](#model-architecture)
- [Quantum Circuit Design](#quantum-circuit-design)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Test Set Inference](#test-set-inference)
- [Project Structure](#project-structure)
- [Key Design Decisions](#key-design-decisions)
- [Configuration](#configuration)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [Differences from Level 1](#differences-from-level-1)

---

## Overview

This project tackles **Level 2** of the Quandela swaption hackathon challenge: given a swaption implied volatility surface with some cells missing, reconstruct the full surface.

The critical challenge is that **any individual (tenor, maturity) cell can be missing** — not just specific rows or columns. This requires a **quantum masked autoencoder**: a model trained by randomly masking cells during training, forcing it to learn the global structure of the surface well enough to impute any arbitrary missing pattern at test time.

---

## The Problem

The test set provides a swaption surface where specific cells are `NaN`. For example:

```
Tenor : 5;  Maturity : 0.0833  → NaN  (missing)
Tenor : 15; Maturity : 0.25    → NaN  (missing)
Tenor : 10; Maturity : 0.5     → NaN  (missing)
Tenor : 1;  Maturity : 1.0     → 0.1432  (observed)
...
```

The missing cells can be at **any position** in the 14×16 tenor/maturity grid. The model must predict the volatility values for all missing positions given the observed ones.

---

## Dataset

**Source**: [`Quandela/Challenge_Swaptions`](https://huggingface.co/datasets/Quandela/Challenge_Swaptions) on HuggingFace

**File used**: `level-2_Missing_data_prediction/train_level2.csv`

```python
from datasets import load_dataset

ds = load_dataset(
    "Quandela/Challenge_Swaptions",
    data_files="level-2_Missing_data_prediction/train_level2.csv",
    split="train",
)
```

| Property | Value |
|---|---|
| Rows (training days) | 489 |
| Features per row | 224 (14 tenors × 16 maturities) |
| Missing values in training | **0** — training data is fully observed |
| Value type | Implied volatility (decimal) |
| Value range | ~0.02 – 0.45 |

**Important**: The training set has no missing values. Missing cells only appear in the test set. We simulate them during training using random masking.

---

## Key Design Insight

### Why a Masked Autoencoder?

The test set has arbitrary individual cells missing — not entire rows or columns. This rules out approaches that target specific maturities or tenors. Instead we need a model that:

1. Can handle **any missing pattern** it hasn't seen before
2. Understands the **global structure** of the surface (correlations across both tenor and maturity dimensions)
3. Clearly distinguishes between **"cell is zero"** and **"cell is missing"**

The masked autoencoder approach solves all three: random masking during training exposes the model to many different missing patterns, the quantum circuit captures global surface correlations, and the binary mask channel resolves the zero/missing ambiguity.

### Why the Binary Mask Matters

Missing cells are filled with `0` in the input. But short-maturity cells genuinely have volatilities close to `0` (~0.03). Without the mask, the model cannot tell the difference between a truly low-volatility cell and a missing cell that was zeroed. The mask provides a second 224-dim binary channel where `1 = observed` and `0 = missing`, making this distinction explicit.

---

## Model Architecture

```
Input: observed surface (224) + binary mask (224) = 448 features
             │
    ┌─────────────────────────────────┐
    │  PREPROCESSING                  │
    │  Fill NaN → 0                   │
    │  MinMaxScaler on surface → [0,1]│
    │  Re-zero masked positions       │
    │  Concatenate [surface | mask]   │
    │  PCA: 448 → 16                  │
    │  MinMaxScaler on PCA → [0,1]    │
    └────────────┬────────────────────┘
                 │
    ┌────────────▼────────────────────┐
    │  QUANTUM CIRCUIT                │
    │  CircuitBuilder (16 modes)      │
    │  · entangling_layer (trainable) │
    │  · angle_encoding (scale=π)     │
    │  · rotations (trainable)        │
    │  · superpositions depth=2       │
    │                                 │
    │  4 photons · UNBUNCHED space    │
    │  MeasurementStrategy.probs()    │
    └────────────┬────────────────────┘
                 │
    ┌────────────▼────────────────────┐
    │  LexGrouping(Fock → 32)         │
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
    Full 224-dim reconstructed surface
    → at test time, only missing cell
      predictions are extracted
```

---

## Quantum Circuit Design

```python
builder = CircuitBuilder(n_modes=16)
builder.add_entangling_layer(trainable=True, name="U1")
builder.add_angle_encoding(modes=list(range(16)), name="input", scale=np.pi)
builder.add_rotations(trainable=True, name="theta")
builder.add_superpositions(depth=2, trainable=True)   # depth=2 for richer interference

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
| `depth=2` superpositions | Spatial interpolation needs richer quantum interference than time series forecasting. Deeper superpositions create more complex entanglement between modes. |
| `probs() + LexGrouping` | Richer output than `mode_expectations()`. The full Fock distribution carries more information about the quantum state, which matters when reconstructing 224 values. |
| `n_photons=4` | Balances expressivity and simulation speed for a 16-mode circuit. |
| `UNBUNCHED` | At most 1 photon per mode. Reduces Fock state count, faster simulation. |
| No pre-compression layer | PCA already outputs exactly 16 components → 16 modes. No extra `Linear(→16)` needed unlike Level 1. |

---

## Results

| Metric | Value |
|---|---|
| Val RMSE (masked cells only) | **0.014704** |
| Val MAE (masked cells only) | **0.010172** |
| Volatility range | 0.02 – 0.45 |
| Error as % of range | ~2.3% |

**Demo predictions on test-style missing cells:**

| Missing Cell | Predicted | True | Error |
|---|---|---|---|
| T5 / M0.083yr | 0.04214 | 0.03588 | +0.00626 |
| T15 / M0.25yr | 0.06320 | 0.05874 | +0.00446 |
| T10 / M0.5yr | 0.09809 | 0.08581 | +0.01228 |

The model correctly learns the ordering of the surface (short maturity = low vol, long maturity = higher vol) and produces predictions in the right range for all tested positions.

---

## Installation

```bash
pip install merlinquantum datasets scikit-learn torch pandas numpy
```

---

## Usage

**Train the model:**
```bash
python qml_level2.py
```

**What the script outputs:**
- Training progress every 10 epochs (loss computed on masked cells only)
- Validation RMSE and MAE on masked cells in original volatility units
- Demo reconstruction for 3 test-style missing cells
- Saved model: `qrc_level2_model.pt`

---

## Test Set Inference

The script includes a `predict_missing()` function designed to work directly with the test set format:

```python
# Load your test row as a dictionary
test_row = {
    "Tenor : 5; Maturity : 0.0833333333333333": np.nan,   # missing
    "Tenor : 15; Maturity : 0.25": np.nan,                # missing
    "Tenor : 10; Maturity : 0.5": np.nan,                 # missing
    "Tenor : 1; Maturity : 1": 0.1432,                    # observed
    # ... all other observed columns
}

predictions = predict_missing(
    row_dict=test_row,
    model=model,
    surface_scaler=surface_scaler,
    pca=pca,
    pca_scaler=pca_scaler,
    feature_cols=feature_cols,
    device=DEVICE,
)

# predictions is a dict: {column_name: predicted_volatility}
# Only contains entries for the NaN columns
print(predictions)
# {
#   "Tenor : 5; Maturity : 0.0833...": 0.04214,
#   "Tenor : 15; Maturity : 0.25": 0.06320,
#   "Tenor : 10; Maturity : 0.5": 0.09809,
# }
```

The function automatically:
- Identifies NaN positions from the dictionary
- Builds the `[surface | mask]` input
- Runs the full preprocessing pipeline
- Returns predictions only for the missing positions

**Loading the saved model for inference:**
```python
import torch

checkpoint = torch.load("qrc_level2_model.pt")
model.load_state_dict(checkpoint["model_state"])

surface_scaler = checkpoint["surface_scaler"]
pca            = checkpoint["pca"]
pca_scaler     = checkpoint["pca_scaler"]
feature_cols   = checkpoint["feature_cols"]
```

---

## Project Structure

```
.
├── qml_level2.py            # Main training script
├── qrc_level2_model.pt      # Saved model + preprocessing objects
├── analyse_level2.py        # EDA script for Level 2 data
├── level2_eda.png           # EDA output plots
└── README_level2.md         # This file
```

---

## Key Design Decisions

**Why random masking during training?**
The training data has zero missing values. We must artificially create the task. By randomly masking 15% of cells (~34 cells per row) at training time, the model sees thousands of different missing patterns and learns the surface structure well enough to impute any arbitrary pattern at test time.

**Why compute loss only on masked cells?**
Observed cells are given to the model in the input. Computing loss on them would be trivial — the model could just copy them — and would drown out the imputation signal. Restricting the loss to masked positions forces the model to genuinely learn to interpolate.

**Why PCA on [surface | mask] together?**
PCA on the concatenated 448-dim vector preserves the joint structure between surface values and their observation pattern. The mask dimensions get folded into the principal components, so the quantum circuit receives a compressed representation that captures both what values are present and which positions are missing.

**Why `MASK_RATIO = 0.15`?**
This masks ~34 cells out of 224, which represents a realistic density of missing values based on the test template format. If the test set turns out to have more or fewer missing cells, this ratio should be adjusted to match.

**Why no pre-compression `Linear(→16)` layer?**
Unlike Level 1 where the lookback window produced 80 features, PCA already outputs exactly 16 components here. These map directly to the 16 quantum modes, eliminating the need for an extra compression step.

---

## Configuration

All hyperparameters are in the `CONFIG` section at the top of `qml_level2.py`:

| Parameter | Default | Description |
|---|---|---|
| `N_PCA_COMPONENTS` | 16 | PCA output dimension |
| `N_MODES` | 16 | Quantum circuit modes |
| `N_PHOTONS` | 4 | Photons in the register |
| `N_GROUPED_OUTPUTS` | 32 | LexGrouping output size |
| `MASK_RATIO` | 0.15 | Fraction of cells masked per training sample |
| `TRAIN_SPLIT` | 0.85 | Fraction of data for training |
| `EPOCHS` | 100 | Training epochs |
| `LR` | 5e-4 | Initial learning rate |
| `BATCH_SIZE` | 16 | Batch size |

---

## Saving and Loading the Model

The checkpoint saved to `qrc_level2_model.pt` contains everything needed for inference:

```python
{
    "model_state"    : model.state_dict(),
    "pca"            : pca,               # fitted on [surface | mask] (448-dim)
    "pca_scaler"     : pca_scaler,        # fitted on PCA outputs
    "surface_scaler" : surface_scaler,    # fitted on raw surface values
    "feature_cols"   : feature_cols,      # 224 column names
    "config"         : { ... }            # hyperparameters
}
```

**Inference pipeline order:**
`Fill NaN → 0 → MinMaxScaler → re-zero masked → concat mask → PCA → MinMaxScaler → model → inverse MinMaxScaler → extract missing positions`

---

## Differences from Level 1

| Aspect | Level 1 | Level 2 |
|---|---|---|
| Task | Time series forecasting | Spatial imputation |
| Input | 5-day lookback window (80 features) | Observed surface + binary mask (448 features) |
| Pre-compression | `Linear(80→16) + Sigmoid` | None (PCA → 16 directly) |
| Measurement | `probs() + LexGrouping` | `probs() + LexGrouping` |
| Superposition depth | 1 | 2 (richer interference for spatial task) |
| Output | Full 224-dim next-day surface | Full 224-dim reconstructed surface |
| Loss | MSE on all 224 outputs | MSE on masked cells only |
| Training trick | Sliding window | Random cell masking |
| RMSE achieved | 0.009964 | 0.014704 |
