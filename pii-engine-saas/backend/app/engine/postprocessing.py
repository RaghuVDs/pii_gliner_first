from __future__ import annotations
from functools import lru_cache
import re
from typing import List, Dict
from collections import Counter

from app.engine.models import Detection
from app.engine.preprocessing import should_keep_detection


def _context_window(text: str, start: int, end: int, pad: int = 100) -> str:
    s = max(0, start - pad)
    e = min(len(text), end + pad)
    return text[s:e]

@lru_cache(maxsize=2048)
def check_grammar(val: str) -> tuple[bool, bool]:
    """Lightweight grammar gate: uppercase first letter = proper noun, digit token = number."""
    tokens = val.split()
    has_propn = any(t[0].isupper() for t in tokens if t)
    has_num = any(t.isdigit() or t.replace(".", "", 1).isdigit() for t in tokens)
    return has_propn, has_num

# Common words that may appear uppercase but are NOT person names
GRAMMAR_GATE_BLOCKLIST = {
    "customer", "account", "member", "cardmember", "agent", "manager",
    "supervisor", "representative", "associate", "merchant", "admin",
    "administrator", "operator", "analyst", "director", "coordinator",
    "specialist", "consultant", "advisor", "executive", "officer",
    "active", "pending", "premium", "platinum", "gold", "silver",
    "bronze", "standard", "basic", "approved", "declined", "cancelled",
    "verified", "unverified", "primary", "secondary", "default",
    "unknown", "anonymous", "system", "service", "support", "help",
    "test", "demo", "sample", "example", "null", "none", "n/a",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}

def apply_universal_dynamic_filters(text: str, detections: List[Detection]) -> List[Detection]:
    kept: List[Detection] = []

    val_counts = Counter([d.text.lower().strip() for d in detections if d.source == "gliner"])

    for d in detections:
        val = (d.text or "").strip()
        if not val:
            continue

        if d.source in ("regex", "field_label", "context", "pattern_lstm"):
            neighborhood = _context_window(text, d.start, d.end, pad=100)
            if should_keep_detection(d.label, val, neighborhood, source=d.source):
                kept.append(d)
            continue

        # Use suffix/segment matching instead of substring to avoid false hits
        # (e.g. "ID" substring in DISABILITY_STATUS)
        _NUMERIC_SUFFIXES = ("_ID", "_NUMBER", "_CODE", "_DATE", "_SCORE", "_PIN", "_LAST4")
        is_numeric_label = (
            d.label.endswith(_NUMERIC_SUFFIXES)
            or d.label in {"AGE", "FICO_SCORE", "PIN", "ONE_TIME_CODE", "CARD_SECURITY_CODE"}
        )
        has_digit = any(char.isdigit() for char in val)

        if is_numeric_label and not has_digit:
            continue

        # HIGH-CONFIDENCE GLiNER BYPASS: When GLiNER's raw score >= 0.65,
        # the model is confident enough to trust without grammar validation.
        high_confidence_gliner = (d.source == "gliner" and d.score >= 0.65)

        if not high_confidence_gliner:
            has_propn, has_num = check_grammar(val)
            is_acronym = val.isupper() and len(val) > 1

            is_person_label = "PERSON" in d.label or d.label == "MOTHERS_MAIDEN_NAME"
            if is_person_label:
                if not should_keep_detection(d.label, val, ""):
                    continue
            elif not (has_propn or has_num or has_digit or is_acronym):
                continue
        else:
            is_person_label = "PERSON" in d.label or d.label == "MOTHERS_MAIDEN_NAME"

        # Block common words that appear uppercase but aren't PII
        if val.lower().strip() in GRAMMAR_GATE_BLOCKLIST:
            continue

        # Reject very low-confidence GLiNER multi-word PERSON detections
        if is_person_label and d.source == "gliner" and " " in val:
            if d.score < 0.45:
                continue

        if not has_digit and val_counts[val.lower()] > 3:
            if "PERSON" not in d.label:
                continue

        neighborhood = _context_window(text, d.start, d.end, pad=100)
        if should_keep_detection(d.label, val, neighborhood, source=d.source):
            kept.append(d)

    return kept

def remove_false_positives(text: str, detections: List[Detection]) -> List[Detection]:
    return apply_universal_dynamic_filters(text, detections)


def split_person_names(text: str, detections: List[Detection]) -> List[Detection]:
    """
    Splits generic FULL_NAME detections into granular First, Middle, and Last name detections.
    """
    out: List[Detection] = []

    TITLES = {
        "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "dr", "dr.", "sir", "madam",
        "prof", "prof.", "officer", "detective", "sergeant", "sgt", "sgt.",
        "lieutenant", "lt", "lt.", "esq", "esq.", "attorney", "judge",
        # Stop words that regex IGNORECASE may accidentally capture as name parts
        "the", "a", "an", "is", "are", "was", "were", "for", "and", "but",
        "or", "at", "by", "in", "on", "to", "of", "with", "from",
    }

    for d in detections:
        if d.label != "PERSON_FULL_NAME":
            out.append(d)
            continue

        raw = (d.text or "").strip()
        # Strip surrounding quotes (straight and smart)
        raw = raw.strip("\"'\u201c\u201d\u2018\u2019")

        # Build (word, offset_in_raw) pairs by walking raw with regex
        # This avoids the raw.find() bug with duplicate substrings
        word_spans = [(m.group(), m.start()) for m in re.finditer(r"\S+", raw)]
        # Filter out titles/stop words, keeping their positions
        parts = [(w, pos) for w, pos in word_spans if w.lower() not in TITLES]

        if not parts:
            continue

        meta = dict(getattr(d, "meta", {}) or {})

        # 1. Single Word Remaining (After stripping titles)
        if len(parts) == 1:
            word, rel = parts[0]
            out.append(
                Detection(
                    label="PERSON_LAST_NAME",
                    start=d.start + rel,
                    end=d.start + rel + len(word),
                    text=word,
                    score=d.score,
                    source="derived",
                    meta=meta,
                )
            )
            continue

        # 2. First and Last Name (2 or more words)
        first_word, first_rel = parts[0]
        last_word, last_rel = parts[-1]

        out.append(
            Detection(
                label="PERSON_FIRST_NAME",
                start=d.start + first_rel,
                end=d.start + first_rel + len(first_word),
                text=first_word,
                score=max(d.score - 0.02, 0.0),
                source="derived",
                meta=meta,
            )
        )

        if last_rel != first_rel:
            out.append(
                Detection(
                    label="PERSON_LAST_NAME",
                    start=d.start + last_rel,
                    end=d.start + last_rel + len(last_word),
                    text=last_word,
                    score=max(d.score - 0.02, 0.0),
                    source="derived",
                    meta=meta,
                )
            )

        # 3. Middle Name (3 or more words)
        if len(parts) > 2:
            mid_word, mid_rel = parts[1]
            mid_last_word, mid_last_rel = parts[-2]
            middle_text = raw[mid_rel:mid_last_rel + len(mid_last_word)]
            out.append(
                Detection(
                    label="PERSON_MIDDLE_NAME",
                    start=d.start + mid_rel,
                    end=d.start + mid_rel + len(middle_text),
                    text=middle_text,
                    score=max(d.score - 0.02, 0.0),
                    source="derived",
                    meta=meta,
                )
            )

    return out



# Common English words that coincide with short person names -- require high confidence to propagate
_AMBIGUOUS_NAME_WORDS = {
    "may", "art", "mark", "bill", "will", "grant", "grace", "hope", "faith",
    "joy", "summer", "autumn", "dawn", "eve", "holly", "iris", "ivy", "jade",
    "pearl", "ruby", "sage", "rose", "lily", "violet", "penny", "pat", "bob",
    "rob", "sue", "al", "ed", "jo", "ray", "lee", "gene", "jean", "don",
    "frank", "cliff", "dale", "glen", "heath", "lance", "miles", "wade",
    "chase", "cole", "drew", "ford", "lane", "mason", "page", "reed", "troy",
    "cash", "chance", "judge", "major", "august", "spring", "brook", "storm",
}

def propagate_person_names(text: str, detections: List[Detection]) -> List[Detection]:
    """
    Second pass: find undetected re-mentions of already-detected person names.

    Catches patterns like "Dr. Patel", "Mr. Doe", "Hi Emily" where a name
    was already detected elsewhere but this specific mention was missed.
    """
    # Collect unique detected name values by label (only from high-fidelity sources)
    known_names: Dict[str, set] = {}
    name_scores: Dict[str, float] = {}
    covered_spans: List[tuple] = []
    for d in detections:
        if d.label in ("PERSON_FIRST_NAME", "PERSON_LAST_NAME", "PERSON_MIDDLE_NAME"):
            if d.source in ("gliner", "field_label", "context", "regex", "derived"):
                known_names.setdefault(d.label, set()).add(d.text.strip())
                name_scores[d.text.strip()] = max(name_scores.get(d.text.strip(), 0), d.score)
        covered_spans.append((d.start, d.end))

    if not known_names:
        return detections

    new_detections = list(detections)

    # Build combined set of all known name strings -> label
    # Minimum 3 chars to avoid matching common short words
    name_to_label: Dict[str, str] = {}
    for label, names in known_names.items():
        for name in names:
            if len(name) < 3:
                continue
            # Skip ambiguous words unless original detection was high confidence
            if name.lower() in _AMBIGUOUS_NAME_WORDS and name_scores.get(name, 0) < 0.70:
                continue
            name_to_label[name] = label

    # Scan text for each known name
    for name, label in name_to_label.items():
        # Use word-boundary regex to find all occurrences
        pattern = re.compile(r"(?<![A-Za-z])" + re.escape(name) + r"(?![A-Za-z])")
        for m in pattern.finditer(text):
            start, end = m.start(), m.end()

            # Skip if this span overlaps any existing detection
            already_covered = any(
                start < e and end > s
                for s, e in covered_spans
            )
            if already_covered:
                continue

            new_detections.append(
                Detection(
                    label=label,
                    text=name,
                    start=start,
                    end=end,
                    score=0.90,
                    source="propagated",
                    meta={"propagated_from": "known_name"},
                )
            )
            covered_spans.append((start, end))

    return new_detections


def add_instance_numbers(detections: List[Detection]) -> List[Detection]:
    """
    Appends the strictly formatted replacement tags (e.g. <PERSON_FIRST_NAME_1>).

    Same (label, text) pairs share the same instance number so that
    repeated mentions of the same entity (e.g. "Doe" appearing twice)
    are redacted with a consistent tag.
    """
    # Sort chronological for natural numbering
    ordered_detections = sorted(detections, key=lambda x: x.start)

    # Track: label -> next available index
    next_idx: Dict[str, int] = {}
    # Track: (label, normalized_text) -> assigned index
    seen: Dict[tuple, int] = {}
    # Count unique values per label to decide if _N suffix is needed
    unique_per_label: Dict[str, set] = {}
    for d in ordered_detections:
        key = (d.label, (d.text or "").strip().lower())
        unique_per_label.setdefault(d.label, set()).add(key)

    updated: List[Detection] = []
    for d in ordered_detections:
        norm_text = (d.text or "").strip().lower()
        key = (d.label, norm_text)

        if key in seen:
            idx = seen[key]
        else:
            idx = next_idx.get(d.label, 0) + 1
            next_idx[d.label] = idx
            seen[key] = idx

        meta = dict(getattr(d, "meta", {}) or {})
        if len(unique_per_label.get(d.label, set())) == 1 and sum(1 for det in ordered_detections if det.label == d.label) == 1:
            meta["instance_label"] = d.label
        else:
            meta["instance_label"] = f"{d.label}_{idx}"

        d_new = Detection(
            label=d.label,
            start=d.start,
            end=d.end,
            text=d.text,
            score=d.score,
            source=d.source,
            meta=meta,
        )
        d_new.replacement_tag = f"<{d.label}_{idx}>"
        updated.append(d_new)

    return updated
