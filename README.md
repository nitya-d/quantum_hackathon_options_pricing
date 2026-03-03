## Quantum Machine Learning for Option Price Prediction (Q-volution 2026)

This project implements a **Quantum Reservoir Computing (QRC)** approach for predicting option prices from historical swaption data.  
It is the MerLin/photonics adaptation of the **Qiskit Fall Fest Quandela Track 2** codebase, targeting the Q-volution 2026 Hackathon (Quandela: Option Pricing with QML).

### Overview

The implementation combines:
- **Quantum Reservoir** (`src/quantum_reservoir.py`):  
  Fixed MerLin-based photonic circuit (angle/phase encoding only, ≤20 modes, ≤10 photons) that maps classical features to rich quantum features.
- **Classical Regressor** (`src/model.py`):  
  Trainable classical model (Ridge, MLP, or LightGBM; default: **LightGBM**) that learns to predict option prices from quantum features.
- **Data Pipeline** (`src/data_loader.py`):  
  CSV/Excel loader that builds time-series windows, applies log returns, z-score normalization, and (when used in Q-volution mode) PCA to respect mode limits.

**Latest example run (LightGBM, `data/train.csv`)** – from `results/results_summary.txt`:

- Config: `n_qubits=8`, `depth=3`, `encoding=angle`, `lookback=8`, `regressor=lgbm`
- Training (log returns):  
  - MSE = **0.987733**, RMSE = **0.993847**, MAE = **0.656642**, R² = **0.012267**
- Test (reconstructed prices, 1-step ahead):  
  - MSE = **0.092340**, RMSE = **0.303874**, MAE = **0.221406**, R² = **-0.056195**, MAPE = **0.9639%**

The corresponding plots are written under `results/`:
- `prediction_plot.png`, `train_predictions.png`, `test_predictions.png`, `residuals.png`

### Project structure

- **`data/`**: Training and test data (pandas-friendly CSV or original Excel).
  - **`train.csv`** / `train.xlsx`: Training swaption prices (Date + Tenor/Maturity columns).
  - **`test.csv`** / `test.xlsx`: Small labeled test set (if available).
  - **`test_template.csv`** / `test_template.xlsx`: Test template (Date, Type, Tenor/Maturity); may have no labels.
  - **`sample_Simulated_Swaption_Price.csv`** / `.xlsx`: Sample swaption price data.
  - All CSVs are normalized: **Date** first (ISO `YYYY-MM-DD`, sorted), then **Tenor** columns in lexicographic order, **Type** last if present. To regenerate CSVs from Excel: `python scripts/format_data.py`.
- **`scripts/format_data.py`**: Converts Excel files in `data/` to normalized CSV.
- **`src/data_loader.py`**: Data loader with two APIs:
  - Q-volution API (explicit train/test files)
  - Qiskit-Fall-Fest-compatible API: `get_available_pairs`, `prepare_data`, `denormalize`
- **`src/quantum_reservoir.py`**: MerLin-based photonic quantum reservoir with angle/phase encoding only (no amplitude encoding or state injection).
- **`src/model.py`**: Hybrid QML model: quantum reservoir + classical readout (Ridge, MLP, or LightGBM).
- **`main.py`**: Qiskit-Fall-Fest-shaped entry point with `run_experiment(config)` that writes qiskit-like outputs into `results/`.
- **`tune.py`**: Hyperparameter tuning (grid/sampling) that writes `results/tuning_results.json`, `results/best_params.json`, `results/tuning_heatmap.png`.
- **`run_best_params.py`**: Runs `run_experiment` using `results/best_params.json`.
- **`requirements.txt`**: Dependencies (similar role as the old repo).
- **`plot_data.py`**: Optional utility to visualize swaption time series from a single dataset file.
- **`results/`**: Output directory mirroring the old repo:
  - `prediction_plot.png`, `train_predictions.png`, `test_predictions.png`, `residuals.png`, `results_summary.txt`
  - optional: `tuning_heatmap.png`, `tuning_results.json`, `best_params.json`
- **`plots/`**: Output directory for `plot_data.py` figures (if used).
- **`.venv/`**: Python virtual environment (create with the setup below).

### Python environment

From the project root:

```bash
cd q-volution2026_quandela
python -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements.txt
```

If you prefer not to use `requirements.txt`, install manually:
`pandas matplotlib openpyxl scikit-learn torch merlinquantum perceval-quandela lightgbm`.

### Dataset format

Train and test files (CSV or Excel) should contain:

- **`Date`**: First column in the normalized CSVs; ISO `YYYY-MM-DD`, sorted ascending. Parsed with `pd.to_datetime` in the loader.
- **Tenor columns**: `Tenor : <T>; Maturity : <M>` (e.g. `Tenor : 10; Maturity : 10`), in lexicographic order in the CSVs. The loader can use a single target column or multiple for multivariate input (then PCA reduces to `n_modes`).
- **Type** (optional): Last column in test/sample files (e.g. test_template, sample_Simulated_Swaption_Price).

The test file may be a submission template (only dates, no price values). Then the pipeline trains on the train file and can write predictions to `results/test_predictions.csv` if test feature rows exist. To get pandas-friendly CSVs from the Excel sources, run from the project root: `python scripts/format_data.py`.

### Running the QRC pipeline

Train and evaluate (Qiskit-Fall-Fest-shaped; single-file interface):

```bash
cd q-volution2026_quandela
.venv/bin/python main.py
```

Example with a specific file and classical backend:

```bash
.venv/bin/python main.py --data_file data/train.csv --regressor lgbm --lookback 8 --n_qubits 8
```

### Results (example run)

The command above produces Qiskit-Fall-Fest-style artifacts under `results/`.

- **Summary file**: `results/results_summary.txt`
- **Plots**:
  - `results/prediction_plot.png`
  - `results/train_predictions.png`
  - `results/test_predictions.png`
  - `results/residuals.png`

**Latest run metrics** (from `results/results_summary.txt`):

- **Config**: `n_qubits=8`, `depth=3`, `encoding=angle`, `lookback=8`, `regressor=lgbm`
- **Test**: R2 = **-0.056195**, MSE = **0.092340**, RMSE = **0.303874**, MAE = **0.221406**, MAPE = **0.9639%**

#### Prediction plot (test prices)

![prediction_plot](results/prediction_plot.png)

#### Train predictions (log-returns)

![train_predictions](results/train_predictions.png)

#### Residuals (test prices)

![residuals](results/residuals.png)

**Key CLI options (kept similar to the old repo):**

| Option | Default | Description |
|--------|---------|--------------|
| `--data_file` | `data/train.csv` | Single dataset file used for chrono train/test split. |
| `--price_column` | `None` | Target column; if omitted uses `Tenor : 10; Maturity : 10` by default. |
| `--tenor`, `--maturity` | `None` | Alternative way to select target column (`Tenor : T; Maturity : M`). |
| `--lookback` | `8` | Number of past time steps per window. |
| `--test_size` | `0.2` | Fraction held out for test (chronological). |
| `--n_qubits` | `8` | Legacy name; mapped to photonic `n_modes` (≤ 20). |
| `--depth` | `3` | Reservoir depth (entangling layers). |
| `--regressor` | `lgbm` | Classical readout: `ridge`, `mlp`, or `lgbm`. |
| `--no_log_returns` | — | Use raw prices instead of log returns (default uses log returns). |
| `--output_dir` | `results/` | Where result files are saved. |
| `--max_samples` | `None` | Cap number of samples (useful for tuning/debug). |

**Behaviour:**

- **Split**: Uses a single file and performs a **chronological train/test split** (same as the old repo).
- **Train**: Builds windows, applies log returns (unless `--no_log_returns`), and z-score normalization (train only). The MerLin reservoir extracts quantum features, then the classical regressor is trained.
- **Outputs**: Writes qiskit-like artifacts to `results/`: `prediction_plot.png`, `train_predictions.png`, `test_predictions.png`, `residuals.png`, `results_summary.txt`.

### Architecture

#### 1. Data Preprocessing (`src/data_loader.py`)
- Qiskit-compatible API: `get_available_pairs()`, `prepare_data()`, `denormalize()`
- Selects `Tenor : T; Maturity : M` columns from Excel/CSV and builds time-series windows `(X, y)` with the chosen `lookback` size
- Applies z-score normalization using **train-only** statistics and stores mean/std for inverse-transforming targets

#### 2. Quantum Reservoir (`src/quantum_reservoir.py`)
- **Encoding**: Angle/phase encoding only (amplitude encoding and state injection are intentionally avoided due to hardware/QPU constraints)
- **Reservoir Circuit**: Fixed MerLin `CircuitBuilder` photonic circuit
  - `add_angle_encoding` maps normalized features to mode phases
  - Multiple `add_entangling_layer(model="mzi")` calls create an interferometer-style network
- **Feature Extraction**: `MeasurementStrategy.mode_expectations(FOCK)` provides mode-expectation feature vectors

#### 3. Hybrid Model (`src/model.py`)
- `HybridQMLModel` combines the MerLin reservoir with a classical regressor
- Backends:
  - `ridge`: `Ridge(alpha=1.0)`
  - `mlp`: PyTorch MLPRegressor
  - `lgbm`: LightGBM (`LGBMRegressor`) – default and recommended
- The public API (`fit(X, y)`, `predict(X)`, `evaluate(X, y, metrics=[...])`) closely mirrors the original Qiskit implementation

#### 4. Main Script (`main.py`)
- Follows the same pattern as the original Qiskit Fall Fest script: `parse_arguments()` → `run_experiment(config)` → `main()`
- Provides the same style of CLI and produces the same set of result files
- Writes into `results/`:
  - `train_predictions.png`, `test_predictions.png`, `prediction_plot.png`, `residuals.png`
  - `results_summary.txt` (model configuration and train/test metrics)

### Optional: plot data only

If you have a single dataset file (e.g. Parquet or CSV) and want to plot tenor time series and histograms, you can adapt and run `plot_data.py` (it currently expects a Parquet path; change `DATA_FILE` and the loader as needed).
