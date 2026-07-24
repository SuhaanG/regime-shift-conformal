"""
Quick sanity check: does a simple linear regression on the same
features do meaningfully better or worse than the neural lead-time
regressor? Helps distinguish "the neural architecture isn't capturing
the signal" from "the data has a low ceiling for this task regardless
of model complexity."
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression

from calibration.train_lead_time_model import build_pooled_splits, ALL_CSVS, PLACEHOLDER_LEAD_TIME


def flatten_dataset(dataset):
    """Flattens each (window_size, n_features) window into a single
    feature vector for a simple linear model."""
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    X, y = [], []
    for windows, _, lead_times in loader:
        flat_windows = windows.reshape(windows.shape[0], -1).numpy()
        X.append(flat_windows)
        y.append(lead_times.numpy())
    X = np.concatenate(X)
    y = np.concatenate(y)
    valid_mask = y < PLACEHOLDER_LEAD_TIME
    return X[valid_mask], y[valid_mask]


if __name__ == "__main__":
    train_ds, val_ds, test_ds = build_pooled_splits(ALL_CSVS)

    X_train, y_train = flatten_dataset(train_ds)
    X_val, y_val = flatten_dataset(val_ds)

    print(f"Training on {len(y_train)} samples, {X_train.shape[1]} flattened features")

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    mae = np.mean(np.abs(preds - y_val))
    rmse = np.sqrt(np.mean((preds - y_val) ** 2))

    print(f"\nLinear regression baseline:")
    print(f"  Val MAE:  {mae:.3f}")
    print(f"  Val RMSE: {rmse:.3f}")
    print(f"\nFor comparison: neural regressor MAE=22.014, naive median MAE=23.101")