"""
GARCH baseline for regime-shift detection.

GARCH models the conditional variance of returns directly, making it a
more natural fit for volatility-regime detection than ARIMA (which
models the return series' mean process). Adapted the same way as the
ARIMA baseline: fit on a rolling window, forecast forward volatility,
and flag a shift if realized volatility during the horizon deviates
from the GARCH forecast beyond a threshold.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings("ignore")

WINDOW_SIZE = 60
HORIZON = 10


def load_series(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.dropna(subset=["log_return", "regime_shift_label"]).reset_index(drop=True)
    return df


def evaluate_garch_on_ticker(df: pd.DataFrame, threshold_z: float = 2.0):
    """
    For each valid window, fit a GARCH(1,1) model on the trailing
    window_size returns (scaled by 100, standard practice for numerical
    stability in the `arch` package), forecast horizon-day-ahead
    conditional volatility, and flag a shift if realized volatility
    during the horizon exceeds the forecasted volatility by more than
    threshold_z standard deviations.
    """
    returns = df["log_return"].values * 100  # scaling per arch package convention
    true_labels = df["regime_shift_label"].values
    n = len(df)

    preds, labels = [], []

    for end_idx in range(WINDOW_SIZE, n - HORIZON):
        window = returns[end_idx - WINDOW_SIZE:end_idx]
        future_actual = returns[end_idx + 1: end_idx + 1 + HORIZON]
        future_labels = true_labels[end_idx + 1: end_idx + 1 + HORIZON]
        binary_label = int(future_labels.any())

        try:
            model = arch_model(window, vol="Garch", p=1, q=1, rescale=False)
            fit = model.fit(disp="off")
            forecast = fit.forecast(horizon=HORIZON, reindex=False)
            forecasted_vol = np.sqrt(forecast.variance.values[-1])  # forecasted daily std dev per horizon day

            realized_vol = np.std(future_actual)
            avg_forecasted_vol = np.mean(forecasted_vol)

            # flag shift if realized volatility significantly exceeds forecast
            shift_flag = int(realized_vol > threshold_z * avg_forecasted_vol)
        except Exception:
            shift_flag = 0  # if GARCH fails to converge, default to "no shift"

        preds.append(shift_flag)
        labels.append(binary_label)

    return np.array(preds), np.array(labels)


def run_on_all_tickers(csv_paths: list):
    all_preds, all_labels = [], []
    for path in csv_paths:
        print(f"Processing {path}...")
        df = load_series(path)
        if len(df) <= WINDOW_SIZE + HORIZON:
            print(f"  Skipping {path}: insufficient length")
            continue
        preds, labels = evaluate_garch_on_ticker(df)
        all_preds.extend(preds)
        all_labels.extend(labels)

    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"\nGARCH baseline — precision: {precision:.3f}, recall: {recall:.3f}, F1: {f1:.3f}")
    return {"precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    primary_files = glob.glob("data/raw/*_labeled.csv")
    calibration_files = glob.glob("data/calibration_raw/*_labeled.csv")
    all_files = primary_files + calibration_files
    run_on_all_tickers(all_files)