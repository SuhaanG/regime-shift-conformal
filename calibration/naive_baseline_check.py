"""
Sanity check: compares the trained lead-time regressor's MAE against a
trivial baseline that always predicts the mean (or median) training-set
lead time, ignoring the input window entirely. If the trained model
doesn't meaningfully beat this naive baseline, it hasn't learned real
signal from the input features.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from torch.utils.data import DataLoader

from calibration.train_lead_time_model import build_pooled_splits, ALL_CSVS, PLACEHOLDER_LEAD_TIME


def get_valid_targets(dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    all_targets = []
    for _, _, lead_times in loader:
        all_targets.extend(lead_times.numpy())
    all_targets = np.array(all_targets)
    return all_targets[all_targets < PLACEHOLDER_LEAD_TIME]


def evaluate_naive_baseline(train_targets, eval_targets, method="mean"):
    naive_prediction = np.mean(train_targets) if method == "mean" else np.median(train_targets)
    mae = np.mean(np.abs(eval_targets - naive_prediction))
    rmse = np.sqrt(np.mean((eval_targets - naive_prediction) ** 2))
    return naive_prediction, mae, rmse


if __name__ == "__main__":
    train_ds, val_ds, test_ds = build_pooled_splits(ALL_CSVS)

    train_targets = get_valid_targets(train_ds)
    val_targets = get_valid_targets(val_ds)

    print(f"Training set: {len(train_targets)} valid lead-time targets")
    print(f"Validation set: {len(val_targets)} valid lead-time targets")

    for method in ["mean", "median"]:
        naive_pred, mae, rmse = evaluate_naive_baseline(train_targets, val_targets, method=method)
        print(f"\nNaive baseline ({method}={naive_pred:.2f} days):")
        print(f"  Val MAE:  {mae:.3f}")
        print(f"  Val RMSE: {rmse:.3f}")

    print(f"\nFor comparison, trained model's best val_MAE was: 22.767 (epoch 2)")