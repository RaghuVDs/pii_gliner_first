"""Conditional Random Field (CRF) layer for sequence labeling.

Learns transition constraints between PII labels so that the model can
jointly decode all detections in a document rather than classifying each
detection independently.

The CRF enforces that label sequences are globally consistent — e.g.,
nearby detections of SSN and ROUTING_NUMBER are more likely when they
co-occur in financial contexts.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CRFLayer(nn.Module):
    """Linear-chain CRF for sequence labeling.

    Provides:
      - forward():  Computes negative log-likelihood loss for training
      - decode():   Viterbi decoding for inference (best label sequence)
    """

    def __init__(self, num_labels: int):
        super().__init__()
        self.num_labels = num_labels

        # Transition matrix: transitions[i][j] = score for transitioning from label i to label j
        self.transitions = nn.Parameter(torch.randn(num_labels, num_labels))

        # Start and end transition scores
        self.start_transitions = nn.Parameter(torch.randn(num_labels))
        self.end_transitions = nn.Parameter(torch.randn(num_labels))

        self._init_transitions()

    def _init_transitions(self):
        """Initialize transitions with small random values."""
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(
        self,
        emissions: torch.Tensor,
        labels: torch.LongTensor,
        mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        """Compute the negative log-likelihood loss.

        Args:
            emissions: (batch, seq_len, num_labels) — emission scores from the encoder.
            labels: (batch, seq_len) — gold label indices.
            mask: (batch, seq_len) — True for real positions, False for padding.

        Returns:
            Scalar loss (negative log-likelihood, averaged over batch).
        """
        if mask is None:
            mask = torch.ones_like(labels, dtype=torch.bool)

        log_numerator = self._compute_score(emissions, labels, mask)
        log_denominator = self._compute_log_partition(emissions, mask)

        # NLL = log(Z) - score(y)
        nll = log_denominator - log_numerator
        return nll.mean()

    def decode(
        self,
        emissions: torch.Tensor,
        mask: torch.BoolTensor | None = None,
    ) -> list[list[int]]:
        """Viterbi decoding — find the best label sequence for each batch item.

        Args:
            emissions: (batch, seq_len, num_labels) — emission scores.
            mask: (batch, seq_len) — True for real positions.

        Returns:
            List of label index sequences (one per batch item).
        """
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)

        batch_size, seq_len, num_labels = emissions.shape

        # Viterbi forward pass
        # viterbi[t] = best score ending at each label at time t
        viterbi = self.start_transitions + emissions[:, 0]  # (batch, num_labels)
        backpointers: list[torch.Tensor] = []

        for t in range(1, seq_len):
            # (batch, num_labels, 1) + (num_labels, num_labels) → (batch, num_labels, num_labels)
            scores = viterbi.unsqueeze(2) + self.transitions.unsqueeze(0)
            # Best previous label for each current label
            best_scores, best_labels = scores.max(dim=1)  # (batch, num_labels)

            # Add emission for current position
            viterbi_next = best_scores + emissions[:, t]

            # Apply mask: if position is padding, keep previous viterbi scores
            viterbi = torch.where(mask[:, t].unsqueeze(1), viterbi_next, viterbi)
            backpointers.append(best_labels)

        # Add end transition scores
        viterbi += self.end_transitions

        # Backtrack to find best paths
        best_paths = []
        for b in range(batch_size):
            # Find the sequence length for this item
            seq_length = mask[b].sum().item()

            # Best final label
            best_last_label = viterbi[b].argmax().item()
            path = [best_last_label]

            # Backtrack
            for t in range(int(seq_length) - 2, -1, -1):
                best_last_label = backpointers[t][b][best_last_label].item()
                path.append(best_last_label)

            path.reverse()
            best_paths.append(path[:int(seq_length)])

        return best_paths

    def _compute_score(
        self,
        emissions: torch.Tensor,
        labels: torch.LongTensor,
        mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Compute the score of the gold label sequence.

        Returns: (batch,) scores.
        """
        batch_size, seq_len, _ = emissions.shape

        # Start transition + first emission
        score = self.start_transitions[labels[:, 0]]  # (batch,)
        score += emissions[:, 0].gather(1, labels[:, 0].unsqueeze(1)).squeeze(1)

        for t in range(1, seq_len):
            # Transition score from labels[t-1] to labels[t]
            trans = self.transitions[labels[:, t - 1], labels[:, t]]
            # Emission score for labels[t]
            emit = emissions[:, t].gather(1, labels[:, t].unsqueeze(1)).squeeze(1)

            # Only add for non-padding positions
            score += (trans + emit) * mask[:, t].float()

        # End transition for the last real position
        # Find the last real label for each batch item
        seq_lengths = mask.sum(dim=1) - 1  # (batch,)
        last_labels = labels.gather(1, seq_lengths.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_labels]

        return score

    def _compute_log_partition(
        self,
        emissions: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Compute log-partition function (log Z) via the forward algorithm.

        Returns: (batch,) log-partition values.
        """
        batch_size, seq_len, num_labels = emissions.shape

        # alpha[i] = log-sum-exp of all paths ending at label i at current time
        alpha = self.start_transitions + emissions[:, 0]  # (batch, num_labels)

        for t in range(1, seq_len):
            # alpha_t[j] = logsumexp_i(alpha[i] + transitions[i][j]) + emissions[t][j]
            # (batch, num_labels, 1) + (num_labels, num_labels)
            scores = alpha.unsqueeze(2) + self.transitions.unsqueeze(0)
            alpha_next = torch.logsumexp(scores, dim=1) + emissions[:, t]

            # Apply mask
            alpha = torch.where(mask[:, t].unsqueeze(1), alpha_next, alpha)

        # Add end transitions and compute final log-partition
        alpha += self.end_transitions
        return torch.logsumexp(alpha, dim=1)  # (batch,)
