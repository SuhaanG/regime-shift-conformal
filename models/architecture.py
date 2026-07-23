"""
CNN-LSTM-Attention architecture for regime shift detection.

Pipeline: 1D convolutional layers extract local features from raw
OHLCV-derived input sequences, an LSTM processes the resulting sequence
to capture long-horizon dependencies, and an attention layer produces a
weighted context vector emphasizing the most informative time steps
before final classification.
"""

import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    """1D convolutional feature extractor over the temporal axis."""

    def __init__(self, input_channels: int, cnn_channels: int = 64, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=cnn_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.conv2 = nn.Conv1d(
            in_channels=cnn_channels,
            out_channels=cnn_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x expected shape: (batch, input_channels, seq_len)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        return x  # (batch, cnn_channels, seq_len)


class AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention over LSTM hidden states."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn_weights = nn.Linear(hidden_dim, hidden_dim)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_outputs):
        # lstm_outputs shape: (batch, seq_len, hidden_dim)
        scores = self.context_vector(torch.tanh(self.attn_weights(lstm_outputs)))  # (batch, seq_len, 1)
        attn_weights = torch.softmax(scores, dim=1)  # normalize over seq_len
        weighted_context = torch.sum(attn_weights * lstm_outputs, dim=1)  # (batch, hidden_dim)
        return weighted_context, attn_weights.squeeze(-1)  # also return weights for interpretability


class CNNLSTMAttention(nn.Module):
    """
    Full architecture: CNN feature extraction -> LSTM sequence modeling
    -> Attention-weighted context -> classification head.

    Outputs a single logit per input sequence, representing the
    probability of a regime shift occurring within the prediction horizon.
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
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x, return_attention=False):
        # x expected shape: (batch, seq_len, input_channels)
        x = x.permute(0, 2, 1)  # -> (batch, input_channels, seq_len) for Conv1d
        cnn_out = self.cnn(x)  # (batch, cnn_channels, seq_len)
        cnn_out = cnn_out.permute(0, 2, 1)  # -> (batch, seq_len, cnn_channels) for LSTM

        lstm_out, _ = self.lstm(cnn_out)  # (batch, seq_len, lstm_hidden_dim)
        context, attn_weights = self.attention(lstm_out)

        logit = self.classifier(context).squeeze(-1)  # (batch,)

        if return_attention:
            return logit, attn_weights
        return logit


if __name__ == "__main__":
    # Quick smoke test with random data to confirm shapes work end-to-end
    batch_size, seq_len, n_features = 8, 60, 5  # e.g. 60-day window, 5 features (OHLCV)
    dummy_input = torch.randn(batch_size, seq_len, n_features)

    model = CNNLSTMAttention(input_channels=n_features)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # expected: (8,)

    output_with_attn, attn = model(dummy_input, return_attention=True)
    print(f"Attention weights shape: {attn.shape}")  # expected: (8, 60)