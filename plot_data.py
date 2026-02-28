import os

import matplotlib.pyplot as plt
import pandas as pd


DATA_FILE = "019c9f39-e697-7fa1-9725-d93bdd138124.parquet"
OUTPUT_DIR = "plots"


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    df = pd.read_parquet(DATA_FILE)
    if "Date" in df.columns:
        # Dataset uses day/month/year format, e.g. 01/01/2050
        df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
        df = df.sort_values("Date")
    return df


def plot_tenor_time_series(df: pd.DataFrame) -> None:
    target_columns = [
        "Tenor : 10; Maturity : 1",
        "Tenor : 10; Maturity : 5",
        "Tenor : 10; Maturity : 10",
        "Tenor : 10; Maturity : 30",
    ]
    available_columns = [c for c in target_columns if c in df.columns]
    if not available_columns or "Date" not in df.columns:
        return

    plt.figure(figsize=(10, 6))
    for col in available_columns:
        plt.plot(df["Date"], df[col], label=col)

    plt.xlabel("Date")
    plt.ylabel("Rate")
    plt.title("Time series for Tenor 10 across selected maturities")
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "tenor10_time_series.png")
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_single_maturity_histogram(df: pd.DataFrame) -> None:
    candidate_columns = [
        "Tenor : 10; Maturity : 10",
        "Tenor : 10; Maturity : 5",
        "Tenor : 5; Maturity : 10",
    ]
    column = next((c for c in candidate_columns if c in df.columns), None)
    if column is None:
        return

    plt.figure(figsize=(8, 5))
    df[column].hist(bins=40)

    plt.xlabel("Rate")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {column}")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "tenor10_maturity10_hist.png")
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    ensure_output_dir()
    df = load_dataset()
    plot_tenor_time_series(df)
    plot_single_maturity_histogram(df)


if __name__ == "__main__":
    main()

