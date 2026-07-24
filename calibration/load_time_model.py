"""
Lead-time regression model for conformal calibration.

Unlike the classification models (which predict whether a shift occurs
within a fixed horizon), this model predicts a continuous target: the
number of days until the next regime shift. This continuous prediction
is what gets conformalized — the classifier tells you IF a shift is
coming, this model estimates WHEN, and conformal calibration wraps a
statistically valid interval around that WHEN estimate.

Reuses the same CNN-LSTM-Attention backbone as the main architecture,
but with a regression head (single continuous output) instead of a
classification head, and trained with a masked loss that only counts
windows where a future shift genuinely exists within the observed
series (avoiding the placeholder "infinity" lead-time values used as
a fallback in the dataset).
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn

from models.architecture import CNNFeatureExtractor, AdditiveAttention


class LeadTimeRegressor(nn.Module):
    """
    CNN -> LSTM -> Attention -> single continuous output (predicted
    lead time in days until the next regime shift).
    """

    def __init__(
        self,
        input_channels: int,
        cnn_channels: int = 64,
        lstm_hidden_dim: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.cnn = CNNFeatureExtractor(input_channels, cnn_channels)
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.attention = AdditiveAttention(lstm_hidden_dim)
        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_channels)
        x = x.permute(0, 2, 1)
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.permute(0, 2, 1)

        lstm_out, _ = self.lstm(cnn_out)
        context, _ = self.attention(lstm_out)

        predicted_lead_time = self.regressor(context).squeeze(-1)
        # softplus ensures the model can only predict non-negative lead times,
        # which is a meaningful physical constraint (can't have a negative
        # number of days until an event)
        return torch.nn.functional.softplus(predicted_lead_time)


def masked_mse_loss(predictions: torch.Tensor, targets: torch.Tensor, max_valid_lead_time: float) -> torch.Tensor:
    """
    Computes MSE loss only over windows with a genuine, observed future
    shift (i.e., excluding the placeholder large value used when no
    future shift exists within the remaining series length).
    """
    valid_mask = targets < max_valid_lead_time
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=predictions.device)
    return nn.functional.mse_loss(predictions[valid_mask], targets[valid_mask])


if __name__ == "__main__":
    # Smoke test with random data
    batch_size, seq_len, n_features = 8, 60, 11
    dummy_input = torch.randn(batch_size, seq_len, n_features)

    model = LeadTimeRegressor(input_channels=n_features)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # expected: (8,)
    print(f"Output values (should all be >= 0): {output}")