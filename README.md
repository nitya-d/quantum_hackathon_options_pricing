## Q-volution 2026 – Quandela Dataset Playground

This folder contains local experiments around the Quandela / Aqora swaption dataset used for the Q-volution 2026 challenge.
The data is stored as a single Parquet file and two main code paths are provided:
- a small plotting script to visualize term-structure slices
- a full MerLin-based Quantum Reservoir Computing (QRC) pipeline built on PyTorch

### Folder structure

- **`data/019c9f39-e697-7fa1-9725-d93bdd138124.parquet`**: Local copy of the Quandela swaption dataset downloaded from Aqora.
- **`src/data_loader.py`**: Parquet-based loader that builds time-series windows `(X, y)` for a chosen swaption column.
- **`src/quantum_reservoir.py`**: MerLin-based photonic quantum reservoir with angle/phase encoding (no amplitude encoding).
- **`src/model.py`**: PyTorch `HybridQMLModel` combining the quantum reservoir with an MLP regressor.
- **`main.py`**: CLI entry point that wires data loading, reservoir, and training loop.
- **`plot_data.py`**: Utility script that reads the Parquet file and generates basic plots.
- **`plots/`**: Output directory where generated figures are saved (created automatically).
- **`.venv/`**: Local Python virtual environment (created automatically when running the setup below).

### Python environment

Create and initialize a local virtual environment in this folder:

```bash
cd q-volution2026_quandela
python -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install pandas pyarrow matplotlib torch merlinquantum perceval-quandela
```

If you already executed a similar command once, the environment and dependencies should already be available.

### Dataset format

The Parquet file contains:

- A **`Date`** column (string in `DD/MM/YYYY` format, e.g. `01/01/2050`).
- Many numeric columns of the form  
  `Tenor : <T>; Maturity : <M>`  
  where `<T>` is the tenor (e.g. `1`, `5`, `10`, `30`) and `<M>` is the maturity in years (e.g. `1`, `5`, `10`, `30`).

The script `plot_data.py`:

- Converts `Date` to a proper `datetime` object and sorts rows by date.
- Selects a subset of tenor/maturity combinations for visualization.

### How to generate plots

From inside `q-volution2026_quandela`:

```bash
cd q-volution2026_quandela
.venv/bin/python plot_data.py
```

This will:

- Read `data/019c9f39-e697-7fa1-9725-d93bdd138124.parquet`.
- Create the folder `plots/` if it does not exist.
- Save two PNG figures:
  - `plots/tenor10_time_series.png`: time series for `Tenor : 10` at selected maturities (`1`, `5`, `10`, `30` years).
  - `plots/tenor10_maturity10_hist.png`: histogram of one selected tenor/maturity column (by default prioritizing `Tenor : 10; Maturity : 10` if present).

You can edit `plot_data.py` to change:

- Which tenor/maturity combinations are plotted in the time series (`target_columns`).
- Which single column is used for the histogram (`candidate_columns`).

### Quantum Reservoir Computing pipeline

To train and evaluate the full QRC model on the swaption dataset:

```bash
cd q-volution2026_quandela
.venv/bin/python main.py --epochs 50 --batch_size 64
```

Key CLI options:

- `--target_column`: which `Tenor : T; Maturity : M` column to use as the price series.
- `--lookback`: number of past time steps encoded by the reservoir.
- `--n_modes`, `--n_photons`: photonic circuit size, subject to hard limits (≤ 20 modes, ≤ 10 photons).
- `--no_log_returns`: disable log-return transformation and work directly with prices.

Under the hood:

- `src/data_loader.py` loads `data/019c9f39-...parquet`, selects the target column, and builds `lookback`-sized windows with z-score normalization (train-only statistics).
- `src/quantum_reservoir.py` constructs a MerLin `QuantumLayer` using **angle/phase encoding only** (no amplitude encoding or state injection) and extracts per-mode expectation features.
- `src/model.py` defines a PyTorch MLP that takes these quantum features as input and predicts future option prices.


