"""
Adaptive PII Learning — PII-Safe Self-Learning Context Rules

COMPLIANCE: This module NEVER stores actual PII values on disk.
Instead it stores:
  - Structure patterns (e.g., "NNN-NN-NNNN" instead of "621-73-4489")
  - PII label names, confidence scores, detection sources
  - Neighborhood text with all PII spans masked as <LABEL> tags
  - Context keywords extracted from sanitized neighborhoods
  - Co-occurring label statistics

Refactored for SaaS: Updated imports from app.engine.models.
"""
from __future__ import annotations

import os
import re
import threading
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml

from app.engine.models import Detection

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "..", "..", "data")
PENDING_PATH = os.path.join(CONFIG_DIR, "pending_rules.yaml")
CONTEXT_RULES_PATH = os.path.join(CONFIG_DIR, "context_rules.yaml")
STATS_PATH = os.path.join(CONFIG_DIR, "detection_stats.yaml")

# Words too common to be useful as context keywords
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "need", "dare",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "and", "but", "or", "if", "while", "because", "until", "although",
    "this", "that", "these", "those", "i", "me", "my", "we", "our", "you",
    "your", "he", "him", "his", "she", "her", "it", "its", "they", "them",
    "their", "what", "which", "who", "whom", "ok", "yes", "no", "just",
    "also", "about", "up", "down", "any", "like", "right", "well", "now",
    "agent", "customer", "sure", "please", "thank", "thanks", "hi", "hello",
    "let", "get", "got", "go", "going", "know", "see", "look", "make",
}


# ── PII-Safe Utilities ─────────────────────────────────────────────────


def _value_to_structure(value: str) -> str:
    """Convert a PII value to a structure pattern — NEVER stores the actual value."""
    out = []
    for ch in value:
        if ch.isdigit():
            out.append("N")
        elif ch.isupper():
            out.append("A")
        elif ch.islower():
            out.append("a")
        else:
            out.append(ch)
    return "".join(out)


def _mask_pii_in_text(text: str, detections: List[Detection]) -> str:
    """Replace all detected PII spans in text with <LABEL> placeholders."""
    if not detections:
        return text
    sorted_dets = sorted(detections, key=lambda d: d.start, reverse=True)
    masked = text
    for d in sorted_dets:
        if 0 <= d.start < d.end <= len(masked):
            masked = masked[:d.start] + f"<{d.label}>" + masked[d.end:]
    return masked


def _extract_safe_neighborhood(
    text: str,
    start: int,
    end: int,
    all_detections: List[Detection],
    pad: int = 150,
) -> str:
    """Extract neighborhood text around a span with all PII masked out."""
    win_start = max(0, start - pad)
    win_end = min(len(text), end + pad)
    window_text = text[win_start:win_end]

    window_dets = []
    for d in all_detections:
        if d.start < win_end and d.end > win_start:
            adj_start = max(0, d.start - win_start)
            adj_end = min(len(window_text), d.end - win_start)
            if adj_start < adj_end:
                window_dets.append(Detection(
                    label=d.label, text=d.text,
                    start=adj_start, end=adj_end,
                    score=d.score, source=d.source, meta=d.meta,
                ))

    return _mask_pii_in_text(window_text, window_dets)


# ── YAML Helpers ────────────────────────────────────────────────────────


def _load_yaml(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _save_yaml(path: str, data: Dict, indent: int = 2) -> None:
    tmp = path + ".tmp"

    class IndentedDumper(yaml.SafeDumper):
        pass

    def _list_representer(dumper, data):
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=False)

    IndentedDumper.add_representer(list, _list_representer)

    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(
            data, f, Dumper=IndentedDumper,
            default_flow_style=False, allow_unicode=True,
            sort_keys=False, indent=indent,
        )
    os.replace(tmp, path)


def _extract_keywords(neighborhood: str, value: str, top_n: int = 8) -> List[str]:
    """Extract the most informative keywords from a neighborhood string."""
    text = neighborhood.lower()
    value_lower = value.lower().strip()
    tokens = re.findall(r"\b[a-z][a-z\-]{2,}\b", text)
    value_parts = set(value_lower.split())
    meaningful = [
        t for t in tokens
        if t not in _STOPWORDS and t not in value_parts
        and len(t) >= 3 and not t.isdigit()
    ]
    counts = Counter(meaningful)
    return [word for word, _ in counts.most_common(top_n)]


# ── Unknown PII Accumulator (PII-Safe) ─────────────────────────────────


class UnknownPIIAccumulator:
    """Accumulates unclassified PII detections with PII-SAFE context."""

    def __init__(self, pending_path: str = PENDING_PATH):
        self.pending_path = pending_path
        self._lock = threading.Lock()
        self._buffer: List[Dict] = []

    def record(self, candidates: List[Dict], all_detections: List[Detection], full_text: str) -> None:
        sanitized = []
        for c in candidates:
            value = c.get("value", "").strip()
            start = c.get("start", 0)
            end = c.get("end", 0)
            if not value:
                continue
            safe_neighborhood = _extract_safe_neighborhood(full_text, start, end, all_detections, pad=150)
            sanitized.append({
                "structure_pattern": _value_to_structure(value),
                "value_length": len(value),
                "original_label": c.get("original_label", "UNKNOWN_PII"),
                "safe_neighborhood": safe_neighborhood,
                "score": c.get("score", 0.0),
                "source": c.get("source", "unknown"),
                "co_occurring_labels": sorted(set(
                    d.label for d in all_detections
                    if abs(d.start - start) < 500 and d.label != c.get("original_label")
                )),
            })
        with self._lock:
            self._buffer.extend(sanitized)

    def flush(self) -> int:
        with self._lock:
            to_process = self._buffer[:]
            self._buffer.clear()
        if not to_process:
            return 0
        pending = _load_yaml(self.pending_path)
        new_groups = 0
        for candidate in to_process:
            safe_neighborhood = candidate.get("safe_neighborhood", "")
            structure = candidate.get("structure_pattern", "")
            original_label = candidate.get("original_label", "UNKNOWN_PII")
            if not safe_neighborhood:
                continue
            keywords = _extract_keywords(safe_neighborhood, structure)
            if not keywords:
                continue
            group_key = "_".join(sorted(keywords[:4])).upper()
            if not group_key:
                continue
            suggested_label = _suggest_label(keywords, safe_neighborhood)
            if group_key in pending:
                entry = pending[group_key]
                entry["seen_count"] = entry.get("seen_count", 0) + 1
                entry["last_seen"] = datetime.now().isoformat()
                examples = entry.get("example_contexts", [])
                example_snippet = f"[{structure}] | ...{safe_neighborhood[:150]}..."
                if example_snippet not in examples:
                    examples.append(example_snippet)
                    entry["example_contexts"] = examples[-5:]
                existing_kw = set(entry.get("suggested_keywords", []))
                existing_kw.update(keywords)
                entry["suggested_keywords"] = sorted(existing_kw)[:12]
                structures = entry.get("structure_patterns", [])
                if structure and structure not in structures:
                    structures.append(structure)
                    entry["structure_patterns"] = structures[-10:]
                co_labels = set(entry.get("co_occurring_labels", []))
                co_labels.update(candidate.get("co_occurring_labels", []))
                entry["co_occurring_labels"] = sorted(co_labels)[:15]
                if suggested_label and not entry.get("suggested_label"):
                    entry["suggested_label"] = suggested_label
            else:
                pending[group_key] = {
                    "suggested_label": suggested_label,
                    "seen_count": 1,
                    "first_seen": datetime.now().isoformat(),
                    "last_seen": datetime.now().isoformat(),
                    "original_labels": [original_label],
                    "example_contexts": [f"[{structure}] | ...{safe_neighborhood[:150]}..."],
                    "suggested_keywords": keywords,
                    "structure_patterns": [structure] if structure else [],
                    "co_occurring_labels": candidate.get("co_occurring_labels", []),
                    "avg_score": candidate.get("score", 0.0),
                    "promoted": False,
                }
                new_groups += 1
        _save_yaml(self.pending_path, pending)
        return new_groups

    def get_pending(self) -> Dict:
        return _load_yaml(self.pending_path)

    def get_stats(self) -> Dict[str, Any]:
        pending = self.get_pending()
        if not pending:
            return {"total_groups": 0, "ready_to_promote": 0, "total_sightings": 0}
        total_sightings = sum(e.get("seen_count", 0) for e in pending.values())
        ready = sum(1 for e in pending.values()
                    if e.get("seen_count", 0) >= 3 and not e.get("promoted", False))
        return {"total_groups": len(pending), "ready_to_promote": ready, "total_sightings": total_sightings}


# ── Detection Stats Tracker (PII-Safe) ─────────────────────────────────


class DetectionStatsTracker:
    def __init__(self, stats_path: str = STATS_PATH):
        self.stats_path = stats_path
        self._lock = threading.Lock()

    def record_run(self, detections: List[Detection]) -> None:
        if not detections:
            return
        with self._lock:
            stats = _load_yaml(self.stats_path)
            stats.setdefault("total_runs", 0)
            stats["total_runs"] += 1
            stats.setdefault("label_counts", {})
            stats.setdefault("source_counts", {})
            stats.setdefault("label_source_counts", {})
            stats["last_updated"] = datetime.now().isoformat()
            for d in detections:
                stats["label_counts"][d.label] = stats["label_counts"].get(d.label, 0) + 1
                stats["source_counts"][d.source] = stats["source_counts"].get(d.source, 0) + 1
                combo_key = f"{d.label}|{d.source}"
                stats["label_source_counts"][combo_key] = stats["label_source_counts"].get(combo_key, 0) + 1
            _save_yaml(self.stats_path, stats)

    def get_stats(self) -> Dict[str, Any]:
        return _load_yaml(self.stats_path)


# ── Label Suggestion Heuristic ──────────────────────────────────────────


def _suggest_label(keywords: List[str], neighborhood: str) -> str:
    n = neighborhood.lower()
    kw_set = set(keywords)
    label_hints = [
        ({"account", "number"}, "ACCOUNT_NUMBER"),
        ({"member", "id"}, "MEMBER_ID"),
        ({"policy", "number"}, "INSURANCE_POLICY"),
        ({"policy"}, "INSURANCE_POLICY"),
        ({"license", "number"}, "LICENSE_NUMBER"),
        ({"certificate"}, "CERTIFICATE_NUMBER"),
        ({"badge"}, "EMPLOYEE_ID"),
        ({"claim"}, "CLAIM_NUMBER"),
        ({"case", "number"}, "CASE_NUMBER"),
        ({"ticket"}, "TICKET_NUMBER"),
        ({"reference"}, "REFERENCE_IDENTIFIER"),
        ({"biometric"}, "BIOMETRIC_INFORMATION"),
        ({"fingerprint"}, "BIOMETRIC_INFORMATION"),
        ({"loan"}, "LOAN_ACCOUNT_NUMBER"),
        ({"mortgage"}, "MORTGAGE_LOAN_INFO"),
        ({"insurance"}, "INSURANCE_POLICY"),
        ({"medical"}, "MEDICAL_RECORD_NUMBER"),
        ({"patient"}, "MEDICAL_RECORD_NUMBER"),
        ({"prescription"}, "PRESCRIPTION_NUMBER"),
        ({"student"}, "STUDENT_ID"),
        ({"vehicle"}, "VIN"),
        ({"passport"}, "PASSPORT_NUMBER"),
        ({"driver"}, "DRIVERS_LICENSE_NUMBER"),
        ({"routing"}, "ROUTING_NUMBER"),
        ({"security", "question"}, "SECURITY_QUESTION"),
        ({"security", "answer"}, "SECURITY_ANSWER"),
        ({"pin"}, "PIN"),
        ({"password"}, "PASSWORD"),
    ]
    for hint_keywords, label in label_hints:
        if hint_keywords.issubset(kw_set) or all(k in n for k in hint_keywords):
            return label
    if len(keywords) >= 2:
        return "_".join(keywords[:3]).upper()
    return "UNKNOWN_PII"


# ── Promotion Pipeline ──────────────────────────────────────────────────


def promote_pending_rules(
    threshold: int = 3,
    pending_path: str = PENDING_PATH,
    context_rules_path: str = CONTEXT_RULES_PATH,
    auto_confirm: bool = False,
) -> List[Dict[str, Any]]:
    pending = _load_yaml(pending_path)
    context_rules = _load_yaml(context_rules_path)
    candidates = []
    for group_key, entry in pending.items():
        if entry.get("promoted", False):
            continue
        if entry.get("seen_count", 0) < threshold:
            continue
        label = entry.get("suggested_label", group_key)
        keywords = entry.get("suggested_keywords", [])
        if not keywords:
            continue
        candidates.append({
            "group_key": group_key, "label": label, "keywords": keywords,
            "seen_count": entry.get("seen_count", 0),
            "example_contexts": entry.get("example_contexts", []),
            "structure_patterns": entry.get("structure_patterns", []),
            "co_occurring_labels": entry.get("co_occurring_labels", []),
        })
    if not auto_confirm:
        return candidates
    promoted = []
    for c in candidates:
        label = c["label"]
        keywords = c["keywords"]
        if label in context_rules:
            existing = set(context_rules[label])
            new_kw = [kw for kw in keywords if kw not in existing]
            context_rules[label].extend(new_kw)
        else:
            context_rules[label] = keywords
        pending[c["group_key"]]["promoted"] = True
        pending[c["group_key"]]["promoted_at"] = datetime.now().isoformat()
        pending[c["group_key"]]["promoted_as"] = label
        promoted.append(c)
    if promoted:
        _save_yaml(context_rules_path, context_rules)
        _save_yaml(pending_path, pending)
    return promoted


def manual_promote(
    group_key: str, label: str, keywords: Optional[List[str]] = None,
    pending_path: str = PENDING_PATH, context_rules_path: str = CONTEXT_RULES_PATH,
) -> bool:
    pending = _load_yaml(pending_path)
    context_rules = _load_yaml(context_rules_path)
    if group_key not in pending:
        return False
    entry = pending[group_key]
    kw = keywords or entry.get("suggested_keywords", [])
    if not kw:
        return False
    if label in context_rules:
        existing = set(context_rules[label])
        new_kw = [k for k in kw if k not in existing]
        context_rules[label].extend(new_kw)
    else:
        context_rules[label] = kw
    entry["promoted"] = True
    entry["promoted_at"] = datetime.now().isoformat()
    entry["promoted_as"] = label
    _save_yaml(context_rules_path, context_rules)
    _save_yaml(pending_path, pending)
    return True


def review_pending_rules(pending_path: str = PENDING_PATH) -> List[Dict[str, Any]]:
    pending = _load_yaml(pending_path)
    entries = []
    for group_key, entry in pending.items():
        entries.append({
            "group_key": group_key,
            "suggested_label": entry.get("suggested_label", "UNKNOWN_PII"),
            "seen_count": entry.get("seen_count", 0),
            "promoted": entry.get("promoted", False),
            "suggested_keywords": entry.get("suggested_keywords", []),
            "example_contexts": entry.get("example_contexts", []),
            "structure_patterns": entry.get("structure_patterns", []),
            "co_occurring_labels": entry.get("co_occurring_labels", []),
            "first_seen": entry.get("first_seen"),
            "last_seen": entry.get("last_seen"),
        })
    return sorted(entries, key=lambda x: x["seen_count"], reverse=True)
