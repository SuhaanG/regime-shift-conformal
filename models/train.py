"""
Training loop for the CNN-LSTM-Attention regime-shift detector.

Uses class-weighted binary cross-entropy to address the positive-class
imbalance identified in the windowed dataset, and tracks precision,
recall, and F1 rather than raw accuracy, given the class imbalance
makes accuracy an uninformative metric here.

This script is meant for a short local smoke test (few epochs, CPU) to
confirm the training loop runs correctly end-to-end before moving full
training runs to Colab.
"""
from torch.utils.data import WeightedRandomSampler
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

from models.architecture import CNNLSTMAttention
from data.splits import build_splits_for_ticker


def compute_pos_weight(dataset) -> float:
    """Computes the positive class weight for BCEWithLogitsLoss based on
    the training set's actual class imbalance (n_negative / n_positive)."""
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        raise ValueError("No positive examples in training set; cannot compute class weight.")
    return n_neg / n_pos


def evaluate(model, dataloader, device) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for windows, labels, _ in dataloader:
            windows, labels = windows.to(device), labels.to(device)
            logits = model(windows)
            loss = criterion(logits, labels)
            total_loss += loss.item() * windows.size(0)

            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"  [diagnostic] mean predicted prob: {torch.sigmoid(logits).mean().item():.4f}" if 'logits' in dir() else "")
    
    return {"loss": avg_loss, "precision": precision, "recall": recall, "f1": f1}


def train(
    ticker_csv: str = "data/raw/VUG_labeled.csv",
    window_size: int = 60,
    horizon: int = 10,
    batch_size: int = 32,
    n_epochs: int = 30,
    lr: float = 3e-4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, val_ds, test_ds = build_splits_for_ticker(ticker_csv, window_size, horizon)
    train_labels = [train_ds[i][1].item() for i in range(len(train_ds))]
    class_sample_count = [len(train_labels) - sum(train_labels), sum(train_labels)]  # [n_neg, n_pos]
    weights = [1.0 / class_sample_count[int(label)] for label in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    pos_weight = compute_pos_weight(train_ds)
    print(f"Positive class weight: {pos_weight:.2f}")

    n_features = train_ds[0][0].shape[1]
    model = CNNLSTMAttention(input_channels=n_features).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0
        for windows, labels, _ in train_loader:
            windows, labels = windows.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(windows)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * windows.size(0)

        train_loss = running_loss / len(train_ds)
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch}/{n_epochs} — "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_metrics['loss']:.4f} | "
            f"val_precision: {val_metrics['precision']:.3f} | "
            f"val_recall: {val_metrics['recall']:.3f} | "
            f"val_f1: {val_metrics['f1']:.3f}"
        )

    return model


if __name__ == "__main__":
    train()
