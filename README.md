## Q-volution 2026 – Quandela Dataset Playground

This folder contains local experiments around the Quandela / Aqora swaption dataset used for the Q-volution 2026 challenge.
The data is stored as a single Parquet file and a small plotting script is provided to quickly visualize some term-structure slices.

### Folder structure

- **`019c9f39-e697-7fa1-9725-d93bdd138124.parquet`**: Local copy of the Quandela swaption dataset downloaded from Aqora.
- **`plot_data.py`**: Utility script that reads the Parquet file and generates basic plots.
- **`plots/`**: Output directory where generated figures are saved (created automatically).
- **`.venv/`**: Local Python virtual environment (created automatically when running the setup command below).

### Python environment

Create and initialize a local virtual environment in this folder:

```bash
cd q-volution2026_quandela
python -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install pandas pyarrow matplotlib
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

- Read `019c9f39-e697-7fa1-9725-d93bdd138124.parquet`.
- Create the folder `plots/` if it does not exist.
- Save two PNG figures:
  - `plots/tenor10_time_series.png`: time series for `Tenor : 10` at selected maturities (`1`, `5`, `10`, `30` years).
  - `plots/tenor10_maturity10_hist.png`: histogram of one selected tenor/maturity column (by default prioritizing `Tenor : 10; Maturity : 10` if present).

You can edit `plot_data.py` to change:

- Which tenor/maturity combinations are plotted in the time series (`target_columns`).
- Which single column is used for the histogram (`candidate_columns`).

### Notes

- This README only documents local data handling and plotting utilities.
- Integration with the main QML pipeline (e.g. scripts in `qiskit-fall-fest2025_paris-saclay_quandela`) can reuse the same Parquet file or load it via `pandas.read_parquet` as needed.

