"""
Trains the LeadTimeRegressor on the pooled multi-ticker dataset,
predicting continuous days-until-next-shift rather than a binary label.

Uses the masked MSE loss defined in lead_time_model.py, which excludes
windows where no genuine future shift exists within the observed series
(the placeholder "large value" case), since those aren't real
regression targets — they're a censoring artifact of finite series
length, conceptually similar to right-censoring in survival analysis.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import copy
import torch
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np

from calibration.lead_time_model import LeadTimeRegressor, masked_mse_loss
from data.splits import build_splits_for_ticker

PRIMARY_CSVS = [
    "data/raw/VUG_labeled.csv",
    "data/raw/ARKK_labeled.csv",
    "data/raw/SPY_labeled.csv",
]
CALIBRATION_CSVS = glob.glob("data/calibration_raw/*_labeled.csv")
ALL_CSVS = PRIMARY_CSVS + CALIBRATION_CSVS

HORIZON = 10
PLACEHOLDER_LEAD_TIME = float(HORIZON * 10)  # matches the placeholder value set in data/splits.py


def build_pooled_splits(csv_paths: list, window_size: int = 60, horizon: int = HORIZON):
    train_sets, val_sets, test_sets = [], [], []
    for path in csv_paths:
        try:
            train_ds, val_ds, test_ds = build_splits_for_ticker(path, window_size, horizon)
            if len(train_ds) > 0 and len(val_ds) > 0 and len(test_ds) > 0:
                train_sets.append(train_ds)
                val_sets.append(val_ds)
                test_sets.append(test_ds)
            else:
                print(f"  Skipping {path}: insufficient data for one or more splits")
        except Exception as e:
            print(f"  Skipping {path}: {e}")

    return ConcatDataset(train_sets), ConcatDataset(val_sets), ConcatDataset(test_sets)


def evaluate(model, dataloader, device) -> dict:
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for windows, _, lead_times in dataloader:
            windows, lead_times = windows.to(device), lead_times.to(device)
            preds = model(windows)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(lead_times.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    valid_mask = all_targets < PLACEHOLDER_LEAD_TIME
    n_valid = valid_mask.sum()

    if n_valid == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "n_valid": 0}

    valid_preds = all_preds[valid_mask]
    valid_targets = all_targets[valid_mask]

    mae = np.mean(np.abs(valid_preds - valid_targets))
    rmse = np.sqrt(np.mean((valid_preds - valid_targets) ** 2))

    return {"mae": mae, "rmse": rmse, "n_valid": int(n_valid)}


def train(window_size: int = 60, horizon: int = HORIZON, batch_size: int = 32, n_epochs: int = 150, lr: float = 3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Pooling {len(ALL_CSVS)} tickers for lead-time regression training")

    missing_primary = [p for p in PRIMARY_CSVS if not os.path.exists(p)]
    if missing_primary:
        raise FileNotFoundError(f"Primary evaluation assets missing, aborting: {missing_primary}")

    train_ds, val_ds, test_ds = build_pooled_splits(ALL_CSVS, window_size, horizon)
    print(f"Pooled train: {len(train_ds)} windows, val: {len(val_ds)}, test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    n_features = train_ds[0][0].shape[1]
    model = LeadTimeRegressor(input_channels=n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_mae = float("inf")
    best_model_state = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches_with_valid_targets = 0

        for windows, _, lead_times in train_loader:
            windows, lead_times = windows.to(device), lead_times.to(device)

            optimizer.zero_grad()
            preds = model(windows)
            loss = masked_mse_loss(preds, lead_times, PLACEHOLDER_LEAD_TIME)

            if loss.item() > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item()
                n_batches_with_valid_targets += 1

        avg_train_loss = running_loss / max(n_batches_with_valid_targets, 1)
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch}/{n_epochs} — "
            f"train_loss (masked MSE): {avg_train_loss:.4f} | "
            f"val_MAE: {val_metrics['mae']:.3f} | "
            f"val_RMSE: {val_metrics['rmse']:.3f} | "
            f"n_valid_val_windows: {val_metrics['n_valid']}"
        )

        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  -> New best model (MAE={best_mae:.3f}), checkpointed")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), "lead_time_model_checkpoint.pt")
        print(f"\nLoaded best checkpoint (val_MAE={best_mae:.3f}), saved to lead_time_model_checkpoint.pt")

    return model


if __name__ == "__main__":
    train()