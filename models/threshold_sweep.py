"""
Sweeps the decision threshold on the validation set's predicted
probabilities to find the threshold that maximizes F1, rather than
assuming the default 0.5 cutoff is appropriate for this ~5% base-rate
classification problem.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

from models.architecture import CNNLSTMAttention
from models.train_pooled import build_pooled_splits, ALL_CSVS


def get_probs_and_labels(model, dataloader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for windows, labels, _ in dataloader:
            windows = windows.to(device)
            logits = model(windows)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)


def sweep_thresholds(probs, labels, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.05)

    results = []
    for t in thresholds:
        preds = (probs > t).astype(int)
        p = precision_score(labels, preds, zero_division=0)
        r = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        results.append({"threshold": t, "precision": p, "recall": r, "f1": f1})

    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_ds, _ = build_pooled_splits(ALL_CSVS)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    n_features = val_ds[0][0].shape[1]
    model = CNNLSTMAttention(input_channels=n_features).to(device)
    model.load_state_dict(torch.load("best_model_checkpoint.pt", map_location=device))

    probs, labels = get_probs_and_labels(model, val_loader, device)
    results = sweep_thresholds(probs, labels)

    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    best = max(results, key=lambda r: r["f1"])
    for r in results:
        marker = "  <-- best" if r == best else ""
        print(f"{r['threshold']:>10.2f} {r['precision']:>10.3f} {r['recall']:>10.3f} {r['f1']:>10.3f}{marker}")