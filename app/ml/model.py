"""CNN-BiLSTM + MLP hybrid model for PII pattern classification.

Architecture:
    Structure Pattern → Char Embedding → 1D-CNN → BiLSTM → 128-dim
    Context Features  → MLP → 64-dim
    Combined 192-dim  → Classifier → Label probabilities
"""
from __future__ import annotations

import torch
import torch.nn as nn


# Character vocabulary for structure patterns
CHAR_TO_IDX = {
    "<PAD>": 0, "<UNK>": 1,
    "N": 2, "A": 3, "a": 4,
    "-": 5, ".": 6, " ": 7, "/": 8, "@": 9,
    "(": 10, ")": 11, "#": 12, "*": 13,
    ":": 14, "+": 15, "_": 16, ",": 17,
    "'": 18, "\\": 19, "&": 20, "!": 21,
    ";": 22, "=": 23, "[": 24, "]": 25,
}

MAX_PATTERN_LEN = 64


def encode_pattern(pattern: str) -> list[int]:
    """Convert structure pattern string to integer indices."""
    ids = []
    for ch in pattern[:MAX_PATTERN_LEN]:
        ids.append(CHAR_TO_IDX.get(ch, CHAR_TO_IDX["<UNK>"]))
    # Pad to fixed length
    while len(ids) < MAX_PATTERN_LEN:
        ids.append(CHAR_TO_IDX["<PAD>"])
    return ids


class PIIPatternModel(nn.Module):
    """Hybrid CNN-BiLSTM + MLP for PII pattern classification.

    Args:
        num_labels: Number of PII label classes.
        num_keywords: Size of keyword vocabulary (multi-hot input).
        num_sources: Number of detection source types.
        char_vocab_size: Character vocabulary size for structure patterns.
        char_embed_dim: Character embedding dimension.
        cnn_filters: Number of CNN filters per kernel size.
        lstm_hidden: LSTM hidden dimension (per direction).
        mlp_hidden: MLP hidden dimension for context features.
    """

    def __init__(
        self,
        num_labels: int,
        num_keywords: int = 400,
        num_sources: int = 7,
        char_vocab_size: int = len(CHAR_TO_IDX),
        char_embed_dim: int = 32,
        cnn_filters: int = 32,
        lstm_hidden: int = 64,
        mlp_hidden: int = 64,
    ):
        super().__init__()
        self.num_labels = num_labels

        # ── Pattern Encoder (CNN + BiLSTM) ──
        self.char_embed = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)

        # 1D-CNN: 3 kernel sizes to capture 2-char, 3-char, 4-char sub-patterns
        self.convs = nn.ModuleList([
            nn.Conv1d(char_embed_dim, cnn_filters, kernel_size=k, padding=k // 2)
            for k in [2, 3, 4]
        ])
        self.conv_bn = nn.BatchNorm1d(cnn_filters * 3)
        self.conv_dropout = nn.Dropout(0.2)

        # BiLSTM on top of CNN features
        self.lstm = nn.LSTM(
            input_size=cnn_filters * 3,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0,
        )
        self.pattern_dim = lstm_hidden * 2  # bidirectional

        # ── Context Encoder (MLP) ──
        # Input: keywords (multi-hot) + co-labels (multi-hot) + source (one-hot) + score + length
        context_input_dim = num_keywords + num_labels + num_sources + 2
        self.context_mlp = nn.Sequential(
            nn.Linear(context_input_dim, mlp_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(mlp_hidden),
            nn.Dropout(0.3),
        )
        self.context_dim = mlp_hidden

        # ── Classifier Head ──
        combined_dim = self.pattern_dim + self.context_dim
        classifier_hidden = 96
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(classifier_hidden, num_labels),
        )

    def forward(self, char_ids: torch.Tensor, context_features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            char_ids: (batch, MAX_PATTERN_LEN) int tensor of character indices.
            context_features: (batch, context_input_dim) float tensor.

        Returns:
            (batch, num_labels) logits.
        """
        # ── Pattern branch ──
        # (batch, seq_len) → (batch, seq_len, embed_dim)
        char_emb = self.char_embed(char_ids)
        # Conv1d expects (batch, channels, seq_len)
        x = char_emb.transpose(1, 2)

        # Apply each CNN kernel and keep full sequence length
        conv_outs = [torch.relu(conv(x)) for conv in self.convs]
        # Concat along channel dim: (batch, cnn_filters*3, seq_len)
        x = torch.cat(conv_outs, dim=1)
        x = self.conv_bn(x)
        x = self.conv_dropout(x)

        # Back to (batch, seq_len, features) for LSTM
        x = x.transpose(1, 2)
        lstm_out, (h_n, _) = self.lstm(x)
        # Take final hidden states from both directions
        # h_n shape: (num_layers*2, batch, lstm_hidden)
        pattern_vec = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (batch, lstm_hidden*2)

        # ── Context branch ──
        context_vec = self.context_mlp(context_features)  # (batch, mlp_hidden)

        # ── Combine and classify ──
        combined = torch.cat([pattern_vec, context_vec], dim=-1)
        logits = self.classifier(combined)
        return logits
