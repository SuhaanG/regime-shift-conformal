"""
Data pipeline: downloads OHLCV data for the growth-equity universe and
labels regime shifts using PELT-based change point detection on rolling
realized volatility, as a Python-native substitute for the Bai-Perron
structural break test (see methodology section for justification).
"""

import pandas as pd
import numpy as np
import yfinance as yf
import ruptures as rpt
import os

# ---- Configuration ----
TICKERS = ["VUG", "ARKK", "SPY"]  # primary evaluation assets
START_DATE = "2015-01-01"
END_DATE = "2026-01-01"
ROLLING_WINDOW = 30  # days, for realized volatility calculation
OUTPUT_DIR = "data/raw"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV data for a single ticker."""
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker}. Check ticker symbol and date range.")
    df = df.reset_index()
    df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]  # flatten multiindex if present
    return df


def compute_realized_volatility(df: pd.DataFrame, window: int) -> pd.Series:
    """Rolling realized volatility from log returns."""
    log_returns = np.log(df["Close"] / df["Close"].shift(1))
    realized_vol = log_returns.rolling(window=window).std()
    return realized_vol


def label_regime_shifts(volatility_series: pd.Series, penalty: float = 25.0) -> np.ndarray:
    """
    Apply PELT change point detection to a volatility series.
    Returns a binary array (1 = regime shift date, 0 = stable) aligned
    to the input series' index.
    """
    clean_series = volatility_series.dropna()
    signal = clean_series.values.reshape(-1, 1)

    algo = rpt.Pelt(model="rbf").fit(signal)
    breakpoints = algo.predict(pen=penalty)  # returns end indices of each segment

    labels = np.zeros(len(clean_series), dtype=int)
    # breakpoints[-1] is the length of the series itself; exclude it
    for bp in breakpoints[:-1]:
        labels[bp] = 1

    # re-align to original (pre-dropna) index, filling leading NaNs with 0
    full_labels = pd.Series(0, index=volatility_series.index)
    full_labels.loc[clean_series.index] = labels
    return full_labels.values


def process_ticker(ticker: str) -> pd.DataFrame:
    print(f"Processing {ticker}...")
    df = download_ohlcv(ticker, START_DATE, END_DATE)
    df["realized_vol"] = compute_realized_volatility(df, ROLLING_WINDOW)
    df["regime_shift_label"] = label_regime_shifts(df["realized_vol"])
    output_path = os.path.join(OUTPUT_DIR, f"{ticker}_labeled.csv")
    df.to_csv(output_path, index=False)
    print(f"  Saved to {output_path} ({df['regime_shift_label'].sum()} labeled shifts detected)")
    return df


if __name__ == "__main__":
    for ticker in TICKERS:
        process_ticker(ticker)
