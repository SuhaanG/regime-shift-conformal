"""
Verification script: plots realized volatility with detected regime-shift
labels overlaid, so the penalty parameter can be visually sanity-checked
against known historical events before being used as ground truth.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

TICKERS = ["VUG", "ARKK", "SPY"]
DATA_DIR = "data/raw"
OUTPUT_DIR = "data/verification_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_labeled_series(ticker: str):
    path = os.path.join(DATA_DIR, f"{ticker}_labeled.csv")
    df = pd.read_csv(path, parse_dates=["Date"])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["Date"], df["realized_vol"], label="Realized Volatility", color="steelblue", linewidth=0.8)

    shift_dates = df.loc[df["regime_shift_label"] == 1, "Date"]
    for date in shift_dates:
        ax.axvline(date, color="red", linestyle="--", alpha=0.5, linewidth=0.8)

    ax.set_title(f"{ticker}: Realized Volatility with Detected Regime Shifts (n={len(shift_dates)})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Realized Volatility")
    ax.legend(loc="upper right")

    output_path = os.path.join(OUTPUT_DIR, f"{ticker}_verification.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved verification plot: {output_path}")


if __name__ == "__main__":
    for ticker in TICKERS:
        plot_labeled_series(ticker)