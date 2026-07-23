"""
Chronological train/validation/test split for time series data, with
normalization statistics computed on the training portion only to avoid
lookahead bias (fixes the issue flagged in dataset.py).
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume", "realized_vol",
    "log_return", "return_zscore", "volume_zscore", "vol_ratio", "momentum",
]


def _load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.dropna(subset=["realized_vol"]).reset_index(drop=True)
    return df


def _compute_lead_times(labels: np.ndarray) -> np.ndarray:
    n = len(labels)
    lead_times = np.full(n, np.inf)
    next_shift_idx = np.inf
    for i in range(n - 1, -1, -1):
        if labels[i] == 1:
            next_shift_idx = i
        lead_times[i] = next_shift_idx - i if next_shift_idx != np.inf else np.inf
    return lead_times


def chronological_split(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15):
    """Splits a dataframe chronologically (no shuffling) into
    train/val/test. Returns three dataframes."""
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def fit_normalization_stats(train_df: pd.DataFrame) -> dict:
    """Computes mean/std for each feature using ONLY the training split."""
    stats = {}
    for col in FEATURE_COLUMNS:
        stats[col] = (train_df[col].mean(), train_df[col].std())
    return stats


def apply_normalization(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Applies precomputed train-set normalization stats to any split."""
    df = df.copy()
    for col in FEATURE_COLUMNS:
        mean, std = stats[col]
        df[col] = (df[col] - mean) / (std + 1e-8)
    return df


class RegimeShiftSplitDataset(Dataset):
    """
    Same windowing logic as RegimeShiftWindowDataset, but operates on an
    already-split, already-normalized dataframe rather than a raw CSV.
    """

    def __init__(self, df: pd.DataFrame, window_size: int = 60, horizon: int = 10):
        self.window_size = window_size
        self.horizon = horizon

        self.features = df[FEATURE_COLUMNS].values.astype(np.float32)
        self.raw_labels = df["regime_shift_label"].values.astype(np.int64)
        self.lead_times = _compute_lead_times(self.raw_labels)

        self.valid_indices = list(range(window_size, len(df) - horizon)) if len(df) > window_size + horizon else []

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        end_idx = self.valid_indices[idx]
        start_idx = end_idx - self.window_size

        window = self.features[start_idx:end_idx]
        future_window = self.raw_labels[end_idx + 1: end_idx + 1 + self.horizon]
        binary_label = int(future_window.any())

        lead_time = self.lead_times[end_idx]
        lead_time = lead_time if np.isfinite(lead_time) else float(self.horizon * 10)

        return (
            torch.tensor(window, dtype=torch.float32),
            torch.tensor(binary_label, dtype=torch.float32),
            torch.tensor(lead_time, dtype=torch.float32),
        )


def build_splits_for_ticker(csv_path: str, window_size: int = 60, horizon: int = 10):
    """Full pipeline: load -> chronological split -> fit norm on train ->
    apply to all splits -> build windowed datasets for each."""
    df = _load_and_clean(csv_path)
    train_df, val_df, test_df = chronological_split(df)

    stats = fit_normalization_stats(train_df)

    train_df = apply_normalization(train_df, stats)
    val_df = apply_normalization(val_df, stats)
    test_df = apply_normalization(test_df, stats)

    train_ds = RegimeShiftSplitDataset(train_df, window_size, horizon)
    val_ds = RegimeShiftSplitDataset(val_df, window_size, horizon)
    test_ds = RegimeShiftSplitDataset(test_df, window_size, horizon)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    train_ds, val_ds, test_ds = build_splits_for_ticker("data/raw/VUG_labeled.csv")
    print(f"Train: {len(train_ds)} windows")
    print(f"Val:   {len(val_ds)} windows")
    print(f"Test:  {len(test_ds)} windows")

    train_positives = sum(train_ds[i][1].item() for i in range(len(train_ds)))
    val_positives = sum(val_ds[i][1].item() for i in range(len(val_ds)))
    test_positives = sum(test_ds[i][1].item() for i in range(len(test_ds)))
    print(f"Positive labels — train: {train_positives}, val: {val_positives}, test: {test_positives}")