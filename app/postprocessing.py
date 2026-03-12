from __future__ import annotations
from functools import lru_cache
import re
import spacy
from typing import List, Dict
from collections import Counter

from app.models import Detection
from app.preprocessing import should_keep_detection

# Lazy load the spaCy grammar engine so it doesn't slow down boot times
nlp = None

def get_nlp():
    global nlp
    if nlp is None:
        # MASSIVE SPEEDUP: Disable NER and Parser if only doing POS Tagging
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    return nlp

def _context_window(text: str, start: int, end: int, pad: int = 100) -> str:
    s = max(0, start - pad)
    e = min(len(text), end + pad)
    return text[s:e]

@lru_cache(maxsize=2048)
def check_grammar(val: str) -> tuple[bool, bool]:
    doc = get_nlp()(val)
    has_propn = any(token.pos_ == "PROPN" for token in doc)
    has_num = any(token.pos_ == "NUM" or token.like_num for token in doc)
    return has_propn, has_num

def apply_universal_dynamic_filters(text: str, detections: List[Detection]) -> List[Detection]:
    kept: List[Detection] = []
    
    val_counts = Counter([d.text.lower().strip() for d in detections if d.source == "gliner"])

    for d in detections:
        val = (d.text or "").strip()
        if not val:
            continue

        if d.source == "regex":
            neighborhood = _context_window(text, d.start, d.end, pad=100)
            if should_keep_detection(d.label, val, neighborhood):
                kept.append(d)
            continue

        is_numeric_label = any(keyword in d.label for keyword in ["ID", "NUMBER", "CODE", "DATE", "SCORE", "AGE", "ZIP"])
        has_digit = any(char.isdigit() for char in val)
        
        if is_numeric_label and not has_digit:
            continue

        has_propn, has_num = check_grammar(val)
        is_acronym = val.isupper() and len(val) > 1

        if not (has_propn or has_num or has_digit or is_acronym):
            continue

        if not has_digit and val_counts[val.lower()] > 3:
            if "PERSON" not in d.label:
                continue 

        neighborhood = _context_window(text, d.start, d.end, pad=100)
        if should_keep_detection(d.label, val, neighborhood):
            kept.append(d)

    return kept

def remove_false_positives(text: str, detections: List[Detection]) -> List[Detection]:
    return apply_universal_dynamic_filters(text, detections)


def split_person_names(text: str, detections: List[Detection]) -> List[Detection]:
    """
    Splits generic FULL_NAME detections into granular First, Middle, and Last name detections.
    """
    out: List[Detection] = []
    
    TITLES = {"mr", "mr.", "mrs", "mrs.", "ms", "ms.", "dr", "dr.", "sir", "madam"}

    for d in detections:
        if d.label != "PERSON_FULL_NAME":
            out.append(d)
            continue

        raw = (d.text or "").strip().strip("\"'“”‘’")
        parts = [p for p in re.split(r"\s+", raw) if p and p.lower() not in TITLES]

        if not parts:
            continue

        # 1. Single Word Remaining (After stripping titles)
        if len(parts) == 1:
            first = parts[0]
            rel_first = raw.find(first)
            if rel_first >= 0:
                out.append(
                    Detection(
                        label="PERSON_LAST_NAME", # Usually if it's "Mr. Doe", Doe is the last name!
                        start=d.start + rel_first,
                        end=d.start + rel_first + len(first),
                        text=first,
                        score=d.score,
                        source="derived",
                        meta={**getattr(d, "meta", {})},
                    )
                )
            continue

        # 2. First and Last Name (2 or more words)
        first = parts[0]
        last = parts[-1]

        rel_first = raw.find(first)
        rel_last = raw.rfind(last)

        if rel_first >= 0:
            out.append(
                Detection(
                    label="PERSON_FIRST_NAME",
                    start=d.start + rel_first,
                    end=d.start + rel_first + len(first),
                    text=first,
                    score=max(d.score - 0.02, 0.0),
                    source="derived",
                    meta={**getattr(d, "meta", {})},
                )
            )

        if rel_last >= 0 and last != first:
            out.append(
                Detection(
                    label="PERSON_LAST_NAME",
                    start=d.start + rel_last,
                    end=d.start + rel_last + len(last),
                    text=last,
                    score=max(d.score - 0.02, 0.0),
                    source="derived",
                    meta={**getattr(d, "meta", {})},
                )
            )

        # 3. Middle Name (3 or more words)
        if len(parts) > 2:
            middle = " ".join(parts[1:-1])
            rel_middle = raw.find(middle)
            
            if rel_middle >= 0:
                out.append(
                    Detection(
                        label="PERSON_MIDDLE_NAME",
                        start=d.start + rel_middle,
                        end=d.start + rel_middle + len(middle),
                        text=middle,
                        score=max(d.score - 0.02, 0.0),
                        source="derived",
                        meta={**getattr(d, "meta", {})},
                    )
                )

    return out


def add_instance_numbers(detections: List[Detection]) -> List[Detection]:
    """
    Appends the strictly formatted replacement tags (e.g. <PERSON_FIRST_NAME_1>)
    """
    counts: Dict[str, int] = {}
    totals: Dict[str, int] = {}

    # Sort chronological for natural numbering
    ordered_detections = sorted(detections, key=lambda x: x.start)

    for d in ordered_detections:
        totals[d.label] = totals.get(d.label, 0) + 1

    updated: List[Detection] = []
    for d in ordered_detections:
        counts[d.label] = counts.get(d.label, 0) + 1
        idx = counts[d.label]

        meta = dict(getattr(d, "meta", {}) or {})
        if totals[d.label] == 1:
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
        # Add strict dynamic tag (eg: <PERSON_FIRST_NAME_1>)
        d_new.replacement_tag = f"<{d.label}_{idx}>"
        updated.append(d_new)

    return updated