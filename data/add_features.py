"""
Adds engineered technical features to already-labeled CSVs, without
re-running the download/labeling pipeline (which is already validated
against known historical events at penalty=25).

Rationale: raw OHLCV + realized_vol alone appears to carry weak signal
for regime-shift classification (validation metrics near chance level
in the pooled training run). Adds features known to carry more signal
for volatility-regime detection specifically:

- log_return: daily log return
- return_zscore: rolling z-score of return (how unusual today's move is
  relative to recent history)
- volume_zscore: rolling z-score of volume (unusual volume activity)
- vol_ratio: short-window realized vol / long-window realized vol
  (a direct proxy for "is volatility currently expanding or
  contracting relative to its recent baseline" — this is arguably the
  single most relevant engineered feature for this task)
- momentum: cumulative return over a fixed lookback window
"""

import pandas as pd
import numpy as np
import glob
import os

SHORT_VOL_WINDOW = 10
LONG_VOL_WINDOW = 60
ZSCORE_WINDOW = 20
MOMENTUM_WINDOW = 20


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    roll_mean_ret = df["log_return"].rolling(ZSCORE_WINDOW).mean()
    roll_std_ret = df["log_return"].rolling(ZSCORE_WINDOW).std()
    df["return_zscore"] = (df["log_return"] - roll_mean_ret) / (roll_std_ret + 1e-8)

    roll_mean_vol = df["Volume"].rolling(ZSCORE_WINDOW).mean()
    roll_std_vol = df["Volume"].rolling(ZSCORE_WINDOW).std()
    df["volume_zscore"] = (df["Volume"] - roll_mean_vol) / (roll_std_vol + 1e-8)

    short_vol = df["log_return"].rolling(SHORT_VOL_WINDOW).std()
    long_vol = df["log_return"].rolling(LONG_VOL_WINDOW).std()
    df["vol_ratio"] = short_vol / (long_vol + 1e-8)

    df["momentum"] = df["Close"].pct_change(MOMENTUM_WINDOW)

    return df


def process_file(csv_path: str):
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = add_engineered_features(df)
    df = df.dropna().reset_index(drop=True)  # drop rows with NaN from rolling windows
    df.to_csv(csv_path, index=False)
    print(f"  Updated {csv_path} ({len(df)} rows after dropping NaN warmup period)")


if __name__ == "__main__":
    primary_files = glob.glob("data/raw/*_labeled.csv")
    calibration_files = glob.glob("data/calibration_raw/*_labeled.csv")
    all_files = primary_files + calibration_files

    print(f"Adding engineered features to {len(all_files)} files...")
    for path in all_files:
        process_file(path)
    print("Done.")