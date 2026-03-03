"""
Format all datasets in data/ into pandas-friendly CSV files.

- Reads Excel files (train.xlsx, test.xlsx, test_template.xlsx, sample_Simulated_Swaption_Price.xlsx).
- Normalizes: Date first (as datetime), then Tenor/Maturity columns in stable order, Type last if present.
- Saves as CSV with consistent dtypes and date format (ISO).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# File name -> (path, sheet name or 0)
DATASETS = {
    "train.xlsx": ("train.xlsx", 0),
    "test.xlsx": ("test.xlsx", 0),
    "test_template.xlsx": ("test_template.xlsx", 0),
    "sample_Simulated_Swaption_Price.xlsx": ("sample_Simulated_Swaption_Price.xlsx", "Swaption price data sample"),
}


def get_date_col(df: pd.DataFrame) -> str:
    """Find date column (Date or similar)."""
    for c in ["Date", "date", "DATE"]:
        if c in df.columns:
            return c
    raise ValueError(f"No date column in {list(df.columns)}")


def get_tenor_columns(df: pd.DataFrame) -> list[str]:
    """Return Tenor : *; Maturity : * columns in stable order."""
    return [c for c in df.columns if c.strip().startswith("Tenor :") and "; Maturity :" in c]


def get_other_columns(df: pd.DataFrame, date_col: str, tenor_cols: list[str]) -> list[str]:
    """Other columns (e.g. Type) in original order, excluding date and tenor."""
    used = {date_col} | set(tenor_cols)
    return [c for c in df.columns if c not in used]


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize to: Date (first, datetime), tenor columns (sorted), then others."""
    date_col = get_date_col(df)
    tenor_cols = get_tenor_columns(df)
    other_cols = get_other_columns(df, date_col, tenor_cols)

    # Parse date
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], dayfirst=True, errors="coerce")

    # Column order: Date, tenor columns (lexicographic), then Type/others
    tenor_cols_sorted = sorted(tenor_cols)
    order = [date_col] + tenor_cols_sorted + other_cols
    out = out[[c for c in order if c in out.columns]]

    # Sort by date (drop NaT if any)
    out = out.dropna(subset=[date_col])
    out = out.sort_values(date_col).reset_index(drop=True)

    # Ensure numeric columns are float
    for c in tenor_cols_sorted:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for out_name, (src_name, sheet) in DATASETS.items():
        src = DATA_DIR / src_name
        if not src.exists():
            print(f"Skip (not found): {src}")
            continue

        df = pd.read_excel(src, sheet_name=sheet, engine="openpyxl")
        df = normalize(df)

        out_path = DATA_DIR / Path(out_name).with_suffix(".csv")
        df.to_csv(out_path, index=False, date_format="%Y-%m-%d")
        print(f"Saved: {out_path}  shape={df.shape}  columns={len(df.columns)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
