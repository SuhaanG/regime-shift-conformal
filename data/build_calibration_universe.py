"""
Builds a broader growth-equity universe for conformal calibration purposes,
separate from the primary evaluation assets (VUG, ARKK, SPY). Applies the
same PELT-based labeling procedure (penalty=25, validated against known
historical events) across a larger basket to accumulate a statistically
adequate number of labeled regime-shift instances for calibration.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import ruptures as rpt
import os
import time

# Representative large-cap growth constituents (illustrative subset;
# expand/adjust based on availability and your final universe definition)
CALIBRATION_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    "AVGO", "ADBE", "CRM", "NFLX", "AMD", "QCOM", "COST", "PEP"
]

START_DATE = "2015-01-01"
END_DATE = "2026-01-01"
ROLLING_WINDOW = 30
PENALTY = 25.0
OUTPUT_DIR = "data/calibration_raw"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker}.")
    df = df.reset_index()
    df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]
    return df


def compute_realized_volatility(df: pd.DataFrame, window: int) -> pd.Series:
    log_returns = np.log(df["Close"] / df["Close"].shift(1))
    return log_returns.rolling(window=window).std()


def label_regime_shifts(volatility_series: pd.Series, penalty: float) -> np.ndarray:
    clean_series = volatility_series.dropna()
    signal = clean_series.values.reshape(-1, 1)
    algo = rpt.Pelt(model="rbf").fit(signal)
    breakpoints = algo.predict(pen=penalty)

    labels = np.zeros(len(clean_series), dtype=int)
    for bp in breakpoints[:-1]:
        labels[bp] = 1

    full_labels = pd.Series(0, index=volatility_series.index)
    full_labels.loc[clean_series.index] = labels
    return full_labels.values


def process_ticker(ticker: str) -> int:
    print(f"Processing {ticker}...")
    try:
        df = download_ohlcv(ticker, START_DATE, END_DATE)
        df["realized_vol"] = compute_realized_volatility(df, ROLLING_WINDOW)
        df["regime_shift_label"] = label_regime_shifts(df["realized_vol"], PENALTY)
        output_path = os.path.join(OUTPUT_DIR, f"{ticker}_labeled.csv")
        df.to_csv(output_path, index=False)
        n_shifts = int(df["regime_shift_label"].sum())
        print(f"  Saved {output_path} ({n_shifts} shifts)")
        return n_shifts
    except Exception as e:
        print(f"  FAILED for {ticker}: {e}")
        return 0


if __name__ == "__main__":
    total_shifts = 0
    for ticker in CALIBRATION_TICKERS:
        total_shifts += process_ticker(ticker)
        time.sleep(1)  # avoid rate-limiting from rapid sequential requests
    print(f"\nTotal labeled shift events across calibration universe: {total_shifts}")