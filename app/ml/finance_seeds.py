"""Seed data placeholder — all training data comes from GLiNER detections.

No hardcoded patterns. The model learns entirely from real detection runs:
- High-confidence GLiNER/regex detections become supervised training examples
- The model improves organically as more transcripts are processed
- No manual seed maintenance needed

The FINANCE_SEEDS list is empty — kept for API compatibility.
"""
from __future__ import annotations

FINANCE_SEEDS: list[dict] = []
SEED_COUNT = 0
