"""
Runs the main CNN-LSTM-Attention training pipeline across multiple
random seeds, reporting mean +/- std for F1 and PR-AUC rather than a
single run's number, to properly characterize run-to-run variance
(observed to range 0.098-0.133 F1 across unseeded runs in this project).
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import numpy as np
import torch

from models.train_pooled import (
    build_pooled_splits,
    compute_pos_weight,
    evaluate,
    ALL_CSVS,
    PRIMARY_CSVS,
)
from models.architecture import CNNLSTMAttention
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
import copy

SEEDS = [42, 123, 2024, 7, 999]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_single_seed(seed: int, window_size=60, horizon=10, batch_size=32, n_epochs=150, lr=3e-4):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    missing_primary = [p for p in PRIMARY_CSVS if not os.path.exists(p)]
    if missing_primary:
        raise FileNotFoundError(f"Primary evaluation assets missing: {missing_primary}")

    train_ds, val_ds, test_ds = build_pooled_splits(ALL_CSVS, window_size, horizon)

    train_labels = [train_ds[i][1].item() for i in range(len(train_ds))]
    n_pos = sum(train_labels)
    class_sample_count = [len(train_labels) - n_pos, n_pos]
    weights = [1.0 / class_sample_count[int(label)] for label in train_labels]

    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    pos_weight = compute_pos_weight(train_ds)
    n_features = train_ds[0][0].shape[1]
    model = CNNLSTMAttention(input_channels=n_features).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_f1 = 0.0
    best_pr_auc = 0.0

    for epoch in range(1, n_epochs + 1):
        model.train()
        for windows, labels, _ in train_loader:
            windows, labels = windows.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(windows)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_metrics = evaluate(model, val_loader, device)
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_pr_auc = val_metrics["pr_auc"]

    print(f"Seed {seed}: best_f1={best_f1:.3f}, pr_auc_at_best={best_pr_auc:.3f}")
    return best_f1, best_pr_auc


if __name__ == "__main__":
    f1_results, pr_auc_results = [], []

    for seed in SEEDS:
        print(f"\n=== Training with seed {seed} ===")
        f1, pr_auc = train_single_seed(seed)
        f1_results.append(f1)
        pr_auc_results.append(pr_auc)

    f1_arr = np.array(f1_results)
    pr_auc_arr = np.array(pr_auc_results)

    print(f"\n=== Multi-seed summary ({len(SEEDS)} seeds) ===")
    print(f"F1:     mean={f1_arr.mean():.3f}, std={f1_arr.std():.3f}, min={f1_arr.min():.3f}, max={f1_arr.max():.3f}")
    print(f"PR-AUC: mean={pr_auc_arr.mean():.3f}, std={pr_auc_arr.std():.3f}, min={pr_auc_arr.min():.3f}, max={pr_auc_arr.max():.3f}")