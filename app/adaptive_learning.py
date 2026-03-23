"""
Adaptive PII Learning — Strategy 3: Self-Learning Context Rules

This module provides:
1. UnknownPIIAccumulator: Records unclassified PII detections with their context,
   extracts keyword patterns, and accumulates them in pending_rules.yaml.
2. promote_pending_rules(): Moves high-confidence pending rules (seen N+ times)
   into context_rules.yaml so the engine picks them up automatically.
3. review_pending_rules(): Returns the current state of pending rules for human review.
"""
from __future__ import annotations

import os
import re
import threading
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
PENDING_PATH = os.path.join(CONFIG_DIR, "pending_rules.yaml")
CONTEXT_RULES_PATH = os.path.join(CONFIG_DIR, "context_rules.yaml")

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


def _load_yaml(path: str) -> Dict:
    """Load a YAML file, returning empty dict if missing or empty."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _save_yaml(path: str, data: Dict, indent: int = 2) -> None:
    """Atomically write a YAML file with proper indentation."""
    tmp = path + ".tmp"

    class IndentedDumper(yaml.SafeDumper):
        pass

    def _list_representer(dumper, data):
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=False)

    IndentedDumper.add_representer(list, _list_representer)

    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(
            data, f,
            Dumper=IndentedDumper,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            indent=indent,
        )
    os.replace(tmp, path)


def _extract_keywords(neighborhood: str, value: str, top_n: int = 8) -> List[str]:
    """Extract the most informative keywords from a neighborhood string.

    Filters out stopwords, the detected value itself, pure numbers,
    and very short tokens. Returns the top_n most frequent meaningful words.
    """
    # Normalize
    text = neighborhood.lower()
    value_lower = value.lower().strip()

    # Tokenize: keep words and hyphenated compounds
    tokens = re.findall(r"\b[a-z][a-z\-]{2,}\b", text)

    # Filter
    value_parts = set(value_lower.split())
    meaningful = [
        t for t in tokens
        if t not in _STOPWORDS
        and t not in value_parts
        and len(t) >= 3
        and not t.isdigit()
    ]

    # Count and return top_n
    counts = Counter(meaningful)
    return [word for word, _ in counts.most_common(top_n)]


class UnknownPIIAccumulator:
    """Accumulates unclassified PII detections and their context patterns.

    Thread-safe. Writes to pending_rules.yaml on each flush.

    Usage:
        accumulator = UnknownPIIAccumulator()
        accumulator.record(candidates)  # List of unknown candidate dicts
        accumulator.flush()             # Write to disk
    """

    def __init__(self, pending_path: str = PENDING_PATH):
        self.pending_path = pending_path
        self._lock = threading.Lock()
        self._buffer: List[Dict] = []

    def record(self, candidates: List[Dict]) -> None:
        """Buffer unknown PII candidates for later flush to disk.

        Each candidate dict should have:
            - value: str (the detected text)
            - original_label: str (e.g., UNKNOWN_PII)
            - neighborhood: str (surrounding text context)
            - start: int
            - end: int
            - score: float
            - source: str
        """
        with self._lock:
            self._buffer.extend(candidates)

    def flush(self) -> int:
        """Write buffered candidates to pending_rules.yaml.

        Groups candidates by extracted keyword signature, updates seen_count,
        and stores example contexts for human review.

        Returns the number of new candidate groups written.
        """
        with self._lock:
            to_process = self._buffer[:]
            self._buffer.clear()

        if not to_process:
            return 0

        pending = _load_yaml(self.pending_path)
        new_groups = 0

        for candidate in to_process:
            value = candidate.get("value", "").strip()
            neighborhood = candidate.get("neighborhood", "")
            original_label = candidate.get("original_label", "UNKNOWN_PII")

            if not value or not neighborhood:
                continue

            # Extract informative keywords from neighborhood
            keywords = _extract_keywords(neighborhood, value)
            if not keywords:
                continue

            # Create a stable group key from sorted top keywords
            # This groups similar contexts together
            group_key = "_".join(sorted(keywords[:4])).upper()
            if not group_key:
                continue

            # Generate a suggested label from the keywords
            suggested_label = _suggest_label(keywords, neighborhood)

            if group_key in pending:
                entry = pending[group_key]
                entry["seen_count"] = entry.get("seen_count", 0) + 1
                entry["last_seen"] = datetime.now().isoformat()

                # Add example context (keep max 5)
                examples = entry.get("example_contexts", [])
                example_snippet = f"{value} | ...{neighborhood[:120]}..."
                if example_snippet not in examples:
                    examples.append(example_snippet)
                    entry["example_contexts"] = examples[-5:]

                # Merge keywords
                existing_kw = set(entry.get("suggested_keywords", []))
                existing_kw.update(keywords)
                entry["suggested_keywords"] = sorted(existing_kw)[:12]

                # Update suggested label if we now have a better one
                if suggested_label and not entry.get("suggested_label"):
                    entry["suggested_label"] = suggested_label
            else:
                pending[group_key] = {
                    "suggested_label": suggested_label,
                    "seen_count": 1,
                    "first_seen": datetime.now().isoformat(),
                    "last_seen": datetime.now().isoformat(),
                    "original_labels": [original_label],
                    "example_contexts": [f"{value} | ...{neighborhood[:120]}..."],
                    "suggested_keywords": keywords,
                    "promoted": False,
                }
                new_groups += 1

        _save_yaml(self.pending_path, pending)
        return new_groups

    def get_pending(self) -> Dict:
        """Read and return current pending rules."""
        return _load_yaml(self.pending_path)

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics about pending rules."""
        pending = self.get_pending()
        if not pending:
            return {"total_groups": 0, "ready_to_promote": 0, "total_sightings": 0}

        total_sightings = sum(e.get("seen_count", 0) for e in pending.values())
        ready = sum(1 for e in pending.values()
                    if e.get("seen_count", 0) >= 3 and not e.get("promoted", False))
        return {
            "total_groups": len(pending),
            "ready_to_promote": ready,
            "total_sightings": total_sightings,
        }


def _suggest_label(keywords: List[str], neighborhood: str) -> str:
    """Attempt to generate a PII label name from extracted keywords.

    Heuristic: look for domain-specific patterns in the keywords/neighborhood
    that hint at the PII type.
    """
    n = neighborhood.lower()
    kw_set = set(keywords)

    # Pattern-based suggestions
    label_hints = [
        ({"account", "number"}, "ACCOUNT_NUMBER"),
        ({"member", "id"}, "MEMBER_ID"),
        ({"member", "number"}, "MEMBER_ID"),
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

    # Fallback: construct a label from top keywords
    if len(keywords) >= 2:
        return "_".join(keywords[:3]).upper()

    return "UNKNOWN_PII"


def promote_pending_rules(
    threshold: int = 3,
    pending_path: str = PENDING_PATH,
    context_rules_path: str = CONTEXT_RULES_PATH,
    auto_confirm: bool = False,
) -> List[Dict[str, Any]]:
    """Strategy 3, Step C: Promote high-confidence pending rules to context_rules.yaml.

    Scans pending_rules.yaml for entries with seen_count >= threshold that
    haven't been promoted yet. Adds their keywords to context_rules.yaml
    so the engine picks them up in subsequent runs.

    Args:
        threshold: Minimum seen_count required for promotion (default: 3).
        pending_path: Path to pending_rules.yaml.
        context_rules_path: Path to context_rules.yaml.
        auto_confirm: If True, promote without returning candidates first.
                      If False, just return promotion candidates for review.

    Returns:
        List of dicts describing promoted (or promotable) entries:
        [{"group_key": ..., "label": ..., "keywords": [...], "seen_count": N}]
    """
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
            "group_key": group_key,
            "label": label,
            "keywords": keywords,
            "seen_count": entry.get("seen_count", 0),
            "example_contexts": entry.get("example_contexts", []),
        })

    if not auto_confirm:
        return candidates

    # Auto-promote: merge into context_rules.yaml
    promoted = []
    for c in candidates:
        label = c["label"]
        keywords = c["keywords"]

        if label in context_rules:
            # Merge new keywords with existing ones (avoid duplicates)
            existing = set(context_rules[label])
            new_kw = [kw for kw in keywords if kw not in existing]
            context_rules[label].extend(new_kw)
        else:
            context_rules[label] = keywords

        # Mark as promoted in pending
        pending[c["group_key"]]["promoted"] = True
        pending[c["group_key"]]["promoted_at"] = datetime.now().isoformat()
        pending[c["group_key"]]["promoted_as"] = label
        promoted.append(c)

    # Save both files
    if promoted:
        _save_yaml(context_rules_path, context_rules)
        _save_yaml(pending_path, pending)

    return promoted


def manual_promote(
    group_key: str,
    label: str,
    keywords: Optional[List[str]] = None,
    pending_path: str = PENDING_PATH,
    context_rules_path: str = CONTEXT_RULES_PATH,
) -> bool:
    """Manually promote a specific pending rule with a user-specified label.

    Use this when the auto-suggested label isn't right — the human reviewer
    provides the correct PII label name.

    Args:
        group_key: The group key in pending_rules.yaml to promote.
        label: The correct PII label to assign.
        keywords: Optional override keywords. If None, uses suggested_keywords.
        pending_path: Path to pending_rules.yaml.
        context_rules_path: Path to context_rules.yaml.

    Returns:
        True if promoted, False if group_key not found.
    """
    pending = _load_yaml(pending_path)
    context_rules = _load_yaml(context_rules_path)

    if group_key not in pending:
        return False

    entry = pending[group_key]
    kw = keywords or entry.get("suggested_keywords", [])

    if not kw:
        return False

    # Merge into context_rules
    if label in context_rules:
        existing = set(context_rules[label])
        new_kw = [k for k in kw if k not in existing]
        context_rules[label].extend(new_kw)
    else:
        context_rules[label] = kw

    # Mark promoted
    entry["promoted"] = True
    entry["promoted_at"] = datetime.now().isoformat()
    entry["promoted_as"] = label

    _save_yaml(context_rules_path, context_rules)
    _save_yaml(pending_path, pending)
    return True


def review_pending_rules(pending_path: str = PENDING_PATH) -> List[Dict[str, Any]]:
    """Return all pending rules for human review, sorted by seen_count descending."""
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
            "first_seen": entry.get("first_seen"),
            "last_seen": entry.get("last_seen"),
        })
    return sorted(entries, key=lambda x: x["seen_count"], reverse=True)