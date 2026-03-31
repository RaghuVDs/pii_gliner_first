"""Prototypical few-shot classifier for rapid PII type adaptation.

When a human promotes a rule via manual_promote_rule(), this module extracts
a 288-dim feature vector from the LSTM encoder and stores it as a "prototype"
for that label. On future runs, unknown candidates are compared to stored
prototypes via cosine similarity — if close enough, they're classified
immediately without waiting for 3+ sightings.

All stored data is PII-safe: only feature vectors, labels, and metadata.
"""
from __future__ import annotations

import os
import logging
from typing import Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger("pii_engine.few_shot")

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None

ML_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
PROTOTYPES_PATH = os.path.join(ML_SAVE_DIR, "prototypes.yaml")

DEFAULT_SIMILARITY_THRESHOLD = 0.80
DEFAULT_MIN_PROTOTYPES = 1  # Minimum prototypes needed to classify


class PrototypicalFewShotClassifier:
    """Classifies unknown PII candidates by comparing to stored prototypes."""

    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        min_prototypes: int = DEFAULT_MIN_PROTOTYPES,
    ):
        self.similarity_threshold = similarity_threshold
        self.min_prototypes = min_prototypes

        # label -> list of feature vectors (stored as lists for yaml serialization)
        self.prototypes: Dict[str, List[List[float]]] = {}
        self._load()

    def _load(self):
        """Load prototypes from disk."""
        if not os.path.exists(PROTOTYPES_PATH):
            return
        try:
            with open(PROTOTYPES_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and "prototypes" in data:
                self.prototypes = data["prototypes"]
                total = sum(len(v) for v in self.prototypes.values())
                logger.info(f"[FEW-SHOT] Loaded {total} prototypes across {len(self.prototypes)} labels")
        except Exception as e:
            logger.warning(f"[FEW-SHOT] Failed to load prototypes: {e}")

    def save(self):
        """Save prototypes to disk."""
        data = {
            "prototypes": self.prototypes,
            "total_prototypes": sum(len(v) for v in self.prototypes.values()),
            "labels": list(self.prototypes.keys()),
        }
        os.makedirs(os.path.dirname(PROTOTYPES_PATH), exist_ok=True)
        tmp = PROTOTYPES_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        os.replace(tmp, PROTOTYPES_PATH)

    def add_prototype(self, label: str, feature_vector: List[float]):
        """Add a prototype feature vector for a label.

        Called when a human promotes a rule — the LSTM encoder extracts the
        288-dim feature vector for the pattern+context, and it's stored here.

        Args:
            label: PII taxonomy label (e.g., "BIOMETRIC_TEMPLATE_ID").
            feature_vector: 288-dim combined feature vector from PIIPatternModel.get_features().
        """
        if label not in self.prototypes:
            self.prototypes[label] = []

        # Keep at most 10 prototypes per label (prevent unbounded growth)
        if len(self.prototypes[label]) >= 10:
            self.prototypes[label].pop(0)  # Remove oldest

        self.prototypes[label].append(feature_vector)
        self.save()
        logger.info(f"[FEW-SHOT] Added prototype for '{label}' ({len(self.prototypes[label])} total)")

    def classify(self, feature_vector: List[float]) -> Optional[Tuple[str, float]]:
        """Classify an unknown candidate by comparing to stored prototypes.

        Args:
            feature_vector: 288-dim feature vector of the unknown candidate.

        Returns:
            (label, similarity) if a match is found above threshold, else None.
        """
        if not self.prototypes or torch is None:
            return None

        query = torch.tensor(feature_vector, dtype=torch.float32)

        best_label = None
        best_similarity = 0.0

        for label, proto_list in self.prototypes.items():
            if len(proto_list) < self.min_prototypes:
                continue

            # Compute centroid of prototypes for this label
            proto_tensor = torch.tensor(proto_list, dtype=torch.float32)
            centroid = proto_tensor.mean(dim=0)

            # Cosine similarity
            similarity = F.cosine_similarity(query.unsqueeze(0), centroid.unsqueeze(0)).item()

            if similarity > best_similarity:
                best_similarity = similarity
                best_label = label

        if best_label is not None and best_similarity >= self.similarity_threshold:
            return (best_label, round(best_similarity, 4))

        return None

    def is_ready(self) -> bool:
        """Whether there are any prototypes available."""
        return any(len(v) >= self.min_prototypes for v in self.prototypes.values())

    def known_labels(self) -> List[str]:
        """Labels with enough prototypes to classify."""
        return [
            label for label, protos in self.prototypes.items()
            if len(protos) >= self.min_prototypes
        ]

    def stats(self) -> Dict:
        """Return stats about stored prototypes."""
        return {
            "total_labels": len(self.prototypes),
            "total_prototypes": sum(len(v) for v in self.prototypes.values()),
            "labels": {label: len(protos) for label, protos in self.prototypes.items()},
        }
