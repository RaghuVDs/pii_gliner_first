"""Active learning manager for intelligent PII rule promotion.

Replaces passive 3-sighting accumulation with smart selection that ranks
unknown candidates by informativeness, presenting the most valuable ones
for human review first.

Strategies:
  1. Uncertainty sampling: LSTM top-1 vs top-2 prediction gap
  2. Detector disagreement: Count sources that disagree about a span
  3. Diversity: Avoid redundant queries via keyword signature dedup

All stored data is PII-safe: structure patterns, masked neighborhoods,
keywords, labels — no actual PII values.
"""
from __future__ import annotations

import os
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml

from app.models import Detection

logger = logging.getLogger("pii_engine.active_learning")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
QUEUE_PATH = os.path.join(CONFIG_DIR, "active_queue.yaml")


def _load_yaml(path: str) -> Any:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data else {}


def _save_yaml(path: str, data: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    os.replace(tmp, path)


class ActiveLearningManager:
    """Ranks unknown PII candidates by informativeness for human review."""

    def __init__(self):
        self._queue: Dict[str, Dict] = {}
        self._load_queue()

    def _load_queue(self):
        """Load persistent active learning queue from disk."""
        data = _load_yaml(QUEUE_PATH)
        self._queue = data.get("queue", {})

    def _save_queue(self):
        """Save queue to disk."""
        data = {
            "queue": self._queue,
            "total_items": len(self._queue),
            "last_updated": datetime.now().isoformat(),
        }
        _save_yaml(QUEUE_PATH, data)

    def score_detections(
        self,
        detections: List[Detection],
        lstm_prediction_gaps: Optional[Dict[int, float]] = None,
    ) -> None:
        """Score detections by informativeness and add high-value ones to queue.

        Args:
            detections: All detections from the pipeline.
            lstm_prediction_gaps: Optional dict mapping detection index →
                (top1_score - top2_score) from LSTM predictions. Lower gap = more uncertain.
        """
        from app.adaptive_learning import _value_to_structure

        if not lstm_prediction_gaps:
            lstm_prediction_gaps = {}

        for i, d in enumerate(detections):
            # Only score unknowns and low-confidence detections
            if d.label not in {"UNKNOWN_PII", "UNKNOWN_IDENTIFIER", "UNKNOWN_SECRET", "CODE", "REFERENCE_IDENTIFIER"}:
                if d.score > 0.60:
                    continue  # High-confidence known labels don't need review

            informativeness = self._compute_informativeness(d, i, detections, lstm_prediction_gaps)

            if informativeness < 0.3:
                continue  # Not informative enough to bother reviewing

            # Create PII-safe queue entry keyed by structure+label
            structure = _value_to_structure(d.text)
            key = f"{structure}|{d.label}|{d.source}"

            if key in self._queue:
                # Update existing entry with higher informativeness
                entry = self._queue[key]
                entry["seen_count"] = entry.get("seen_count", 0) + 1
                entry["informativeness"] = max(entry.get("informativeness", 0), informativeness)
                entry["last_seen"] = datetime.now().isoformat()
            else:
                self._queue[key] = {
                    "structure": structure,
                    "current_label": d.label,
                    "source": d.source,
                    "score": round(d.score, 4),
                    "informativeness": round(informativeness, 4),
                    "seen_count": 1,
                    "first_seen": datetime.now().isoformat(),
                    "last_seen": datetime.now().isoformat(),
                }

    def _compute_informativeness(
        self,
        detection: Detection,
        det_idx: int,
        all_detections: List[Detection],
        lstm_gaps: Dict[int, float],
    ) -> float:
        """Compute informativeness score for a detection.

        Higher score = more valuable for human review.

        Combines:
          1. Uncertainty (from LSTM prediction gap or low confidence)
          2. Detector disagreement (multiple sources disagree on label)
          3. Novelty (is this a new pattern we haven't seen?)
        """
        score = 0.0

        # 1. Uncertainty: low confidence or close LSTM predictions
        if detection.score < 0.50:
            score += 0.4 * (1.0 - detection.score / 0.50)

        # LSTM prediction gap (lower = more uncertain)
        if det_idx in lstm_gaps:
            gap = lstm_gaps[det_idx]
            if gap < 0.3:
                score += 0.3 * (1.0 - gap / 0.3)

        # 2. Detector disagreement: check if other detections overlap with different labels
        disagreement = 0
        for other in all_detections:
            if other is detection:
                continue
            # Check for overlapping span
            overlap = min(detection.end, other.end) - max(detection.start, other.start)
            span_len = detection.end - detection.start
            if span_len > 0 and overlap > span_len * 0.5:
                if other.label != detection.label:
                    disagreement += 1
        if disagreement > 0:
            score += min(0.3, 0.15 * disagreement)

        # 3. Novelty: unknown labels get a base boost
        if detection.label in {"UNKNOWN_PII", "UNKNOWN_IDENTIFIER", "UNKNOWN_SECRET"}:
            score += 0.2

        return min(1.0, score)

    def get_review_queue(self, top_k: int = 10) -> List[Dict]:
        """Get the top-K most informative candidates for human review.

        Returns:
            List of queue entries sorted by informativeness (highest first).
        """
        entries = [
            {**v, "key": k}
            for k, v in self._queue.items()
            if not v.get("reviewed", False)
        ]
        entries.sort(key=lambda x: (-x["informativeness"], -x["seen_count"]))
        return entries[:top_k]

    def submit_review(self, key: str, correct_label: str) -> bool:
        """Submit a human review for a queued candidate.

        Args:
            key: Queue entry key (from get_review_queue output).
            correct_label: The correct PII taxonomy label.

        Returns:
            True if the review was recorded.
        """
        if key not in self._queue:
            return False

        entry = self._queue[key]
        entry["reviewed"] = True
        entry["correct_label"] = correct_label
        entry["reviewed_at"] = datetime.now().isoformat()
        self._save_queue()

        logger.info(f"[ACTIVE-LEARNING] Review submitted: {key} → {correct_label}")
        return True

    def get_reviewed(self) -> List[Dict]:
        """Get all reviewed entries (for feeding into training)."""
        return [
            {**v, "key": k}
            for k, v in self._queue.items()
            if v.get("reviewed", False)
        ]

    def clear_reviewed(self):
        """Remove reviewed entries from the queue (after they've been used for training)."""
        to_remove = [k for k, v in self._queue.items() if v.get("reviewed", False)]
        for k in to_remove:
            del self._queue[k]
        self._save_queue()

    def flush(self):
        """Save queue to disk."""
        self._save_queue()

    def stats(self) -> Dict[str, Any]:
        """Get active learning statistics."""
        total = len(self._queue)
        reviewed = sum(1 for v in self._queue.values() if v.get("reviewed", False))
        pending = total - reviewed
        avg_informativeness = (
            sum(v.get("informativeness", 0) for v in self._queue.values()) / total
            if total > 0 else 0
        )
        return {
            "total_items": total,
            "pending_review": pending,
            "reviewed": reviewed,
            "avg_informativeness": round(avg_informativeness, 4),
        }
