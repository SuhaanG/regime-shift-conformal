"""
PyTorch Dataset for regime-shift detection.

Converts a labeled OHLCV CSV (Date, OHLC, Volume, realized_vol,
regime_shift_label) into fixed-length rolling windows. Each window is
labeled 1 if a regime shift occurs within `horizon` days after the
window's final observation, else 0. Also stores the lead time (days
until the next shift, if any) for later use in conformal calibration.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume", "realized_vol"]


def _load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.dropna(subset=["realized_vol"]).reset_index(drop=True)
    return df


def _normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalization per feature, computed on this ticker's own history.
    Note: for a real train/test split, normalization statistics should be
    computed on the training portion only and applied to test data, to
    avoid lookahead bias. This function normalizes over the full series
    for simplicity; revisit before final experiments."""
    df = df.copy()
    for col in FEATURE_COLUMNS:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / (std + 1e-8)
    return df


def _compute_lead_times(labels: np.ndarray) -> np.ndarray:
    """For each index, compute the number of days until the next
    regime shift (label == 1). Returns np.inf if no future shift exists."""
    n = len(labels)
    lead_times = np.full(n, np.inf)
    next_shift_idx = np.inf
    for i in range(n - 1, -1, -1):
        if labels[i] == 1:
            next_shift_idx = i
        lead_times[i] = next_shift_idx - i if next_shift_idx != np.inf else np.inf
    return lead_times


class RegimeShiftWindowDataset(Dataset):
    """
    Produces (window_features, binary_label, lead_time) tuples.

    window_features: (window_size, n_features) tensor
    binary_label: 1 if a regime shift occurs within `horizon` days after
                  the window's last day, else 0
    lead_time: days from the window's last day until the next shift
               (inf if none within the remaining series)
    """

    def __init__(self, csv_path: str, window_size: int = 60, horizon: int = 10):
        self.window_size = window_size
        self.horizon = horizon

        df = _load_and_clean(csv_path)
        df = _normalize_features(df)

        self.features = df[FEATURE_COLUMNS].values.astype(np.float32)
        self.raw_labels = df["regime_shift_label"].values.astype(np.int64)
        self.lead_times = _compute_lead_times(self.raw_labels)

        # valid starting indices: need window_size history + horizon lookahead
        self.valid_indices = list(range(window_size, len(df) - horizon))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        end_idx = self.valid_indices[idx]  # last index of the input window
        start_idx = end_idx - self.window_size

        window = self.features[start_idx:end_idx]  # (window_size, n_features)

        # binary label: does a shift occur in (end_idx, end_idx + horizon]?
        future_window = self.raw_labels[end_idx + 1: end_idx + 1 + self.horizon]
        binary_label = int(future_window.any())

        lead_time = self.lead_times[end_idx]
        lead_time = lead_time if np.isfinite(lead_time) else float(self.horizon * 10)  # cap placeholder

        return (
            torch.tensor(window, dtype=torch.float32),
            torch.tensor(binary_label, dtype=torch.float32),
            torch.tensor(lead_time, dtype=torch.float32),
        )


def build_combined_dataset(csv_paths: list[str], window_size: int = 60, horizon: int = 10) -> ConcatDataset:
    """Combines multiple tickers' datasets into a single dataset for
    calibration or training across the broader universe."""
    datasets = [RegimeShiftWindowDataset(path, window_size, horizon) for path in csv_paths]
    return ConcatDataset(datasets)


if __name__ == "__main__":
    # Smoke test on VUG
    ds = RegimeShiftWindowDataset("data/raw/VUG_labeled.csv", window_size=60, horizon=10)
    print(f"Dataset size: {len(ds)}")
    window, label, lead_time = ds[0]
    print(f"Window shape: {window.shape}, label: {label.item()}, lead_time: {lead_time.item()}")
    print(f"Positive labels in dataset: {sum(ds[i][1].item() for i in range(len(ds)))}")