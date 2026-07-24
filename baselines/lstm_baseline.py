"""
Standalone LSTM baseline for regime-shift detection — no CNN
feature extraction, no attention mechanism. Isolates the contribution
of the recurrent component alone, as part of the ablation matrix.

Uses the same pooled multi-ticker dataset, windowing, class-weighted
sampling, and checkpointing/PR-AUC evaluation as the full
CNN-LSTM-Attention model, so results are directly comparable.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

from data.splits import build_splits_for_ticker

PRIMARY_CSVS = [
    "data/raw/VUG_labeled.csv",
    "data/raw/ARKK_labeled.csv",
    "data/raw/SPY_labeled.csv",
]
CALIBRATION_CSVS = glob.glob("data/calibration_raw/*_labeled.csv")
ALL_CSVS = PRIMARY_CSVS + CALIBRATION_CSVS


class StandaloneLSTM(nn.Module):
    """LSTM -> final hidden state -> classification head. No CNN, no attention."""

    def __init__(self, input_channels: int, lstm_hidden_dim: int = 64, lstm_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_channels)
        lstm_out, (h_n, _) = self.lstm(x)
        final_hidden = h_n[-1]  # last layer's final hidden state, (batch, lstm_hidden_dim)
        logit = self.classifier(final_hidden).squeeze(-1)
        return logit


def build_pooled_splits(csv_paths: list, window_size: int = 60, horizon: int = 10):
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


def compute_pos_weight(dataset) -> float:
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        raise ValueError("No positive examples in training set.")
    return n_neg / n_pos


def evaluate(model, dataloader, device) -> dict:
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for windows, labels, _ in dataloader:
            windows, labels = windows.to(device), labels.to(device)
            logits = model(windows)
            loss = criterion(logits, labels)
            total_loss += loss.item() * windows.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    pr_auc = average_precision_score(all_labels, all_probs)

    return {"loss": avg_loss, "precision": precision, "recall": recall, "f1": f1, "pr_auc": pr_auc}


def train(window_size: int = 60, horizon: int = 10, batch_size: int = 32, n_epochs: int = 150, lr: float = 3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Pooling {len(ALL_CSVS)} tickers for training (Standalone LSTM baseline)")

    missing_primary = [p for p in PRIMARY_CSVS if not os.path.exists(p)]
    if missing_primary:
        raise FileNotFoundError(f"Primary evaluation assets missing, aborting: {missing_primary}")

    train_ds, val_ds, test_ds = build_pooled_splits(ALL_CSVS, window_size, horizon)
    print(f"Pooled train: {len(train_ds)} windows, val: {len(val_ds)}, test: {len(test_ds)}")

    train_labels = [train_ds[i][1].item() for i in range(len(train_ds))]
    n_pos = sum(train_labels)
    class_sample_count = [len(train_labels) - n_pos, n_pos]
    weights = [1.0 / class_sample_count[int(label)] for label in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    pos_weight = compute_pos_weight(train_ds)
    print(f"Positive class weight: {pos_weight:.2f}")

    n_features = train_ds[0][0].shape[1]
    model = StandaloneLSTM(input_channels=n_features).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_f1 = 0.0
    best_model_state = None

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
            f"val_f1: {val_metrics['f1']:.3f} | "
            f"val_pr_auc: {val_metrics['pr_auc']:.3f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  -> New best model (F1={best_f1:.3f}), checkpointed")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), "lstm_baseline_checkpoint.pt")
        print(f"\nLoaded best checkpoint (val_f1={best_f1:.3f}), saved to lstm_baseline_checkpoint.pt")

    return model


if __name__ == "__main__":
    train()