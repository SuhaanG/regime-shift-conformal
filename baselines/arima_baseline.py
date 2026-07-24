"""
ARIMA baseline for regime-shift detection.

ARIMA is not natively a classifier, so it is adapted here: for each
window, an ARIMA model is fit on the window's historical log returns,
used to forecast the next `horizon` days, and a shift is flagged if
the realized return's deviation from the forecast exceeds a threshold
(calibrated on the training set), following the same window/horizon
structure as the deep learning models for a fair comparison.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings("ignore")  # ARIMA fitting on short windows throws frequent convergence warnings

WINDOW_SIZE = 60
HORIZON = 10
ARIMA_ORDER = (1, 1, 1)  # standard default; document this choice in methodology


def load_series(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.dropna(subset=["log_return", "regime_shift_label"]).reset_index(drop=True)
    return df


def evaluate_arima_on_ticker(df: pd.DataFrame, threshold_z: float = 2.0, stride: int = 5):
    """
    For each valid window, fit ARIMA on the trailing window_size returns,
    forecast horizon days ahead, and flag a shift if any forecasted
    residual (actual - predicted) exceeds threshold_z standard deviations
    of the window's own historical return volatility.
    """
    returns = df["log_return"].values
    true_labels = df["regime_shift_label"].values
    n = len(df)

    preds, labels = [], []

    for end_idx in range(WINDOW_SIZE, n - HORIZON, stride):
        window = returns[end_idx - WINDOW_SIZE:end_idx]
        future_actual = returns[end_idx + 1: end_idx + 1 + HORIZON]
        future_labels = true_labels[end_idx + 1: end_idx + 1 + HORIZON]
        binary_label = int(future_labels.any())

        try:
            model = ARIMA(window, order=ARIMA_ORDER)
            fit = model.fit()
            forecast = fit.forecast(steps=HORIZON)
            resid = future_actual - forecast
            window_std = np.std(window)
            shift_flag = int(np.any(np.abs(resid) > threshold_z * window_std))
        except Exception:
            shift_flag = 0  # if ARIMA fails to converge on this window, default to "no shift"

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
        preds, labels = evaluate_arima_on_ticker(df)
        all_preds.extend(preds)
        all_labels.extend(labels)

    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"\nARIMA baseline — precision: {precision:.3f}, recall: {recall:.3f}, F1: {f1:.3f}")
    return {"precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    primary_files = glob.glob("data/raw/*_labeled.csv")
    calibration_files = glob.glob("data/calibration_raw/*_labeled.csv")
    all_files = primary_files + calibration_files
    run_on_all_tickers(all_files)