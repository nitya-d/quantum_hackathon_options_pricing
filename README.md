## Q-volution 2026 – Quandela Track (Option Pricing with QML)

This folder contains the Quantum Reservoir Computing (QRC) pipeline for the Q-volution 2026 Hackathon (Quandela: Option Pricing with QML).  
Data is provided as **train** and **test** files (CSV or Excel). The pipeline uses a MerLin-based photonic reservoir and a classical readout (Ridge regression by default, or MLP / LightGBM).

### Folder structure

- **`data/`**: Training and test data (pandas-friendly CSV or original Excel).
  - **`train.csv`** / `train.xlsx`: Training swaption prices (Date + Tenor/Maturity columns).
  - **`test_template.csv`** / `test_template.xlsx`: Test template (Date, Type, Tenor/Maturity); may have no labels.
  - **`sample_Simulated_Swaption_Price.csv`** / `.xlsx`: Sample swaption price data.
  - All CSVs are normalized: **Date** first (ISO `YYYY-MM-DD`, sorted), then **Tenor** columns in lexicographic order, **Type** last if present. To regenerate CSVs from Excel: `python scripts/format_data.py`.
- **`scripts/format_data.py`**: Converts Excel files in `data/` to normalized CSV.
- **`src/data_loader.py`**: Loads train and test CSV/Excel files, builds time-series windows `(X, y)`, applies log returns, z-score normalization, and PCA to fit within photonic mode limits.
- **`src/quantum_reservoir.py`**: MerLin-based photonic quantum reservoir with angle/phase encoding only (no amplitude encoding or state injection).
- **`src/model.py`**: Hybrid QML model: quantum reservoir + classical readout (Ridge, MLP, or LightGBM).
- **`main.py`**: CLI entry point: load data, train, evaluate (if test labels exist), and save plots or predictions.
- **`plot_data.py`**: Optional utility to visualize swaption time series from a single dataset file.
- **`results/`**: Output directory for test plots (`test_timeseries.png`, `test_scatter.png`) and optionally `test_predictions.csv`.
- **`plots/`**: Output directory for `plot_data.py` figures (if used).
- **`.venv/`**: Python virtual environment (create with the setup below).

### Python environment

From the project root:

```bash
cd q-volution2026_quandela
python -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install pandas pyarrow matplotlib openpyxl torch merlinquantum perceval-quandela scikit-learn
```

Optional for LightGBM backend: `pip install lightgbm`.

### Dataset format

Train and test files (CSV or Excel) should contain:

- **`Date`**: First column in the normalized CSVs; ISO `YYYY-MM-DD`, sorted ascending. Parsed with `pd.to_datetime` in the loader.
- **Tenor columns**: `Tenor : <T>; Maturity : <M>` (e.g. `Tenor : 10; Maturity : 10`), in lexicographic order in the CSVs. The loader can use a single target column or multiple for multivariate input (then PCA reduces to `n_modes`).
- **Type** (optional): Last column in test/sample files (e.g. test_template, sample_Simulated_Swaption_Price).

The test file may be a submission template (only dates, no price values). Then the pipeline trains on the train file and can write predictions to `results/test_predictions.csv` if test feature rows exist. To get pandas-friendly CSVs from the Excel sources, run from the project root: `python scripts/format_data.py`.

### Running the QRC pipeline

Train and evaluate (default: train from `data/train.csv`, test from `data/test_template.csv`):

```bash
cd q-volution2026_quandela
.venv/bin/python main.py
```

Example with custom paths and options:

```bash
.venv/bin/python main.py --train_path data/train.csv --test_path data/test.csv --lookback 8 --n_modes 8 --model_type ridge
```

**Main CLI options:**

| Option | Default | Description |
|--------|---------|--------------|
| `--train_path` | `data/train.csv` | Path to training data (CSV or Excel). |
| `--test_path` | `data/test_template.csv` | Path to test data (CSV or Excel). |
| `--date_column` | `Date` | Name of the date column. |
| `--target_column` | `Tenor : 10; Maturity : 10` | Target price column(s). |
| `--lookback` | `8` | Number of past time steps per window. |
| `--n_modes` | `8` | Photonic modes (≤ 20); also used for PCA dimension. |
| `--n_photons` | `4` | Photons (≤ 10). |
| `--model_type` | `lgbm` | Classical readout: `ridge`, `mlp`, or `lgbm`. |
| `--no_log_returns` | — | Use raw prices instead of log returns. |

**Behaviour:**

- **Train**: Loads the train file, builds windows, applies log returns (unless `--no_log_returns`), z-score normalization, and PCA. Fits the quantum reservoir features with the chosen classical model.
- **Test**: If the test file has valid rows and the same columns, builds test windows and runs prediction. If test labels exist, prints metrics (MSE, RMSE, MAE) and saves time-series and scatter plots under `results/`. If the test file has no labels (template only), only predictions are produced and can be saved to `results/test_predictions.csv`.
- **Empty test**: If the test file yields no samples (e.g. all NaN), the script exits after training and reports “No test samples”.

### Pipeline internals

- **DataLoader** (`src/data_loader.py`): Reads train and test CSV/Excel; selects `date_column` and `target_column`(s); builds supervised windows; fits z-score and PCA on train only; returns `X_train`, `X_test`, `y_train`, and `y_test` (or `y_test=None` when test has no labels).
- **Quantum reservoir** (`src/quantum_reservoir.py`): MerLin `QuantumLayer` with angle encoding and fixed entangling layers; no amplitude encoding; mode expectations used as features.
- **Hybrid model** (`src/model.py`): Reservoir features are passed to Ridge (fixed alpha), a PyTorch MLP, or LightGBM (default) for regression.

### Optional: plot data only

If you have a single dataset file (e.g. Parquet or CSV) and want to plot tenor time series and histograms, you can adapt and run `plot_data.py` (it currently expects a Parquet path; change `DATA_FILE` and the loader as needed).
