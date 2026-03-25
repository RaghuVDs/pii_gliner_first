"""CNN-BiLSTM + Attention + MLP hybrid model for PII pattern classification.

Architecture:
    Structure Pattern → Char Embed (48) → 1D-CNN (3 kernels) → 2-layer BiLSTM (96)
                      → Attention Pooling → 192-dim
    Context Features  → 2-layer MLP → 96-dim
    Combined 288-dim  → 2-layer Classifier → Label probabilities

All tensors flow on CUDA when available. ~200K parameters.
"""
from __future__ import annotations

import math
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
    while len(ids) < MAX_PATTERN_LEN:
        ids.append(CHAR_TO_IDX["<PAD>"])
    return ids


class AttentionPooling(nn.Module):
    """Learned attention pooling over LSTM sequence outputs.

    Instead of just taking the final hidden state (which loses info about
    early parts of the pattern), attention learns WHICH positions matter.
    E.g., for "NNN-NN-NNNN" it learns that dash positions and segment
    lengths are the most discriminative features.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )

    def forward(self, lstm_output: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_dim)
            pad_mask: (batch, seq_len) — True for real tokens, False for padding
        Returns:
            (batch, hidden_dim) attention-pooled vector
        """
        scores = self.attention(lstm_output).squeeze(-1)  # (batch, seq_len)
        if pad_mask is not None:
            scores = scores.masked_fill(~pad_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(1)  # (batch, 1, seq_len)
        pooled = torch.bmm(weights, lstm_output).squeeze(1)   # (batch, hidden_dim)
        return pooled


class PIIPatternModel(nn.Module):
    """Production-grade CNN-BiLSTM + Attention + MLP for PII pattern classification.

    Improvements over v1:
      - 2-layer BiLSTM with inter-layer dropout
      - Attention pooling (learns which pattern positions matter)
      - Pad masking (attention ignores padding)
      - Deeper context MLP (2 layers)
      - Deeper classifier (2 layers)
      - He/Kaiming initialization for all linear layers
      - Larger embedding and hidden dims
    """

    def __init__(
        self,
        num_labels: int,
        num_keywords: int = 400,
        num_sources: int = 7,
        char_vocab_size: int = len(CHAR_TO_IDX),
        char_embed_dim: int = 48,
        cnn_filters: int = 48,
        lstm_hidden: int = 96,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.25,
        mlp_hidden: int = 128,
        classifier_hidden: int = 160,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_labels = num_labels

        # ── Pattern Encoder ──────────────────────────────────────────
        self.char_embed = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)

        # 1D-CNN: 3 odd kernel sizes for same-padding
        self.cnn_kernels = [3, 5, 7]
        self.convs = nn.ModuleList([
            nn.Conv1d(char_embed_dim, cnn_filters, kernel_size=k, padding=k // 2)
            for k in self.cnn_kernels
        ])
        cnn_out_dim = cnn_filters * len(self.cnn_kernels)
        self.conv_norm = nn.LayerNorm(cnn_out_dim)
        self.conv_dropout = nn.Dropout(dropout * 0.5)

        # 2-layer BiLSTM
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
        )
        self.lstm_norm = nn.LayerNorm(lstm_hidden * 2)
        pattern_dim = lstm_hidden * 2

        # Attention pooling
        self.attention = AttentionPooling(pattern_dim)

        # ── Context Encoder (2-layer MLP) ────────────────────────────
        context_input_dim = num_keywords + num_labels + num_sources + 2
        self.context_mlp = nn.Sequential(
            nn.Linear(context_input_dim, mlp_hidden),
            nn.GELU(),
            nn.LayerNorm(mlp_hidden),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.GELU(),
            nn.LayerNorm(mlp_hidden // 2),
            nn.Dropout(dropout * 0.5),
        )
        context_dim = mlp_hidden // 2

        # ── Classifier Head (2-layer) ────────────────────────────────
        combined_dim = pattern_dim + context_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, classifier_hidden),
            nn.GELU(),
            nn.LayerNorm(classifier_hidden),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_labels),
        )

        # ── Weight initialization ────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """Kaiming/He initialization for linear/conv, orthogonal for LSTM."""
        for name, param in self.named_parameters():
            if "lstm" in name:
                if "weight_ih" in name:
                    nn.init.kaiming_normal_(param, nonlinearity="relu")
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
                    # Set forget gate bias to 1.0 for better gradient flow
                    hidden_size = param.size(0) // 4
                    param.data[hidden_size:2 * hidden_size].fill_(1.0)
            elif "weight" in name and param.dim() >= 2:
                nn.init.kaiming_normal_(param, nonlinearity="relu")
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, char_ids: torch.Tensor, context_features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            char_ids: (batch, MAX_PATTERN_LEN) int tensor of character indices.
            context_features: (batch, context_input_dim) float tensor.

        Returns:
            (batch, num_labels) logits.
        """
        # Build pad mask: True where char_id != 0 (PAD)
        pad_mask = char_ids != 0  # (batch, seq_len)

        # ── Pattern branch ──
        char_emb = self.char_embed(char_ids)        # (batch, seq_len, embed_dim)
        x = char_emb.transpose(1, 2)                # (batch, embed_dim, seq_len)

        conv_outs = [nn.functional.gelu(conv(x)) for conv in self.convs]
        x = torch.cat(conv_outs, dim=1)             # (batch, cnn_out, seq_len)
        x = x.transpose(1, 2)                       # (batch, seq_len, cnn_out)
        x = self.conv_norm(x)
        x = self.conv_dropout(x)

        lstm_out, _ = self.lstm(x)                   # (batch, seq_len, lstm_hidden*2)
        lstm_out = self.lstm_norm(lstm_out)
        pattern_vec = self.attention(lstm_out, pad_mask)  # (batch, lstm_hidden*2)

        # ── Context branch ──
        context_vec = self.context_mlp(context_features)

        # ── Combine and classify ──
        combined = torch.cat([pattern_vec, context_vec], dim=-1)
        return self.classifier(combined)
