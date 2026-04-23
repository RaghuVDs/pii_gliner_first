"""Candidate-window pre-scan: find regions of a document worth running GLiNER on.

Why:
  GLiNER is the most expensive detector in the pipeline. Running it over an
  entire 80k-character conversation transcript wastes ~95% of compute on
  filler dialogue that contains no PII. By pre-scanning with cheap detectors
  (regex value patterns + field-label trigger phrases + context keywords)
  we can identify the small fraction of the text where PII actually clusters,
  then run GLiNER only on padded windows around those positions.

How:
  1. Compile all regex value patterns from regex_rules.yaml
  2. Compile all field-label trigger phrases from field_patterns.yaml
  3. Compile all context-keyword patterns from context_rules.yaml
  4. For each match, expand by ±pad_chars to give GLiNER enough surrounding
     context for the named-entity classifier to make confident predictions
  5. Merge overlapping windows
  6. If merged windows would cover more than `max_coverage` of the document
     (e.g. >50%), the document is PII-dense — return None and let GLiNER
     scan the full text. Pre-scan would only add overhead in that case.

Output:
  List[(start, end)] of original-text byte offsets, sorted, non-overlapping;
  OR None meaning "scan full text".

Nothing in this module is hardcoded to a particular taxonomy or label set.
The trigger material is built dynamically from the live config dicts the
engine already loads.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from app.models import Detection

logger = logging.getLogger("pii_engine.perf.windows")

# Labels whose regexes rely on letter case (e.g. proper-noun [A-Z] anchors).
# Mirrors RegexDetector._CASE_SENSITIVE_LABELS — kept in sync so the
# consolidated single-pass sweep produces identical detections.
_CASE_SENSITIVE_LABELS = {"PERSON_FULL_NAME"}


# A "hit" is just a position in the text. We don't care which detector
# produced it; the merging step treats all hits uniformly.
Hit = int


@dataclass
class WindowScanStats:
    """Diagnostics from one scan() call. Useful for tuning + debugging."""
    n_regex_hits: int = 0
    n_field_hits: int = 0
    n_context_hits: int = 0
    n_windows_before_merge: int = 0
    n_windows_after_merge: int = 0
    coverage: float = 0.0
    fell_through_to_full_scan: bool = False


class CandidateWindowScanner:
    """Cheap pre-scanner that finds regions of interest for GLiNER.

    Construction is fast; scan() is the hot path. The compiled regex set is
    built once at engine startup and reused for every row.

    The scanner is configured at construction with three pieces of material:
      - regex_rules: dict of {label: [regex_pattern_strings]} (value patterns)
      - field_patterns: dict of {label: [trigger_alias_strings]} (label triggers)
      - context_rules: dict of {label: [regex_pattern_strings]} (additional hints)

    `pad_chars` controls how far to expand around each hit. The default of
    300 is calibrated to give GLiNER's deberta-v3-large enough surrounding
    tokens to make confident PII predictions; smaller windows hurt recall.

    `max_coverage` is the fall-through threshold: if the merged windows
    would cover more than this fraction of the text, scan() returns None,
    signalling the caller to run GLiNER on the full text.
    """

    # High-precision phrase anchors. These are short fragments that, when
    # they appear in conversation, almost always introduce actual PII in the
    # next ~30 characters. We use these instead of bare field-label keywords
    # ("name", "phone", "email") which are far too common in dialogue and
    # cause the scanner to flag every sentence.
    _PHRASE_ANCHORS = [
        # Identity introductions
        r"\bmy name is\b", r"\bthis is\b", r"\bi am\s+(?:mr|mrs|ms|dr)\.?\b",
        r"\bthe customer's name\b", r"\bthe caller(?:'s)? name\b",
        # Contact info
        r"\bmy email( address)? is\b", r"\bemail me at\b", r"\bsend (?:it|that) to\b",
        r"\bmy phone( number)? is\b", r"\bcall(?:ing)? from\b", r"\breach me at\b",
        # Address
        r"\bi live at\b", r"\bmy address is\b", r"\bship(?:ping)? to\b",
        r"\bbilling address\b", r"\bmailing address\b",
        # Account / financial
        r"\bmy account( number)? is\b", r"\bthe account ending in\b", r"\bcard number\b",
        r"\bthe last (?:four|4)\b", r"\bcvv\b", r"\bexpiration date\b",
        # Identifiers
        r"\bmy ssn( is)?\b", r"\bsocial security( number)?( is)?\b",
        r"\bdate of birth( is)?\b", r"\bdob( is)?\b", r"\bborn on\b",
        r"\bdriver'?s license\b", r"\bpassport( number)?\b",
    ]

    def __init__(
        self,
        regex_rules: Optional[Dict[str, List[str]]] = None,
        field_patterns: Optional[Dict[str, List[str]]] = None,
        context_rules: Optional[Dict[str, List[str]]] = None,
        pad_chars: int = 200,
        max_coverage: float = 0.50,
    ):
        self.pad_chars = max(50, int(pad_chars))
        self.max_coverage = max(0.0, min(1.0, float(max_coverage)))
        # Each entry is (label, compiled_regex). Labels are kept so the same
        # sweep that finds candidate-window positions can also produce the
        # final Detection objects, eliminating a duplicate full-text regex
        # pass downstream in RegexDetector.
        self._compiled_value_regexes: List[Tuple[str, re.Pattern]] = []
        self._compiled_field_separator_regex: Optional[re.Pattern] = None
        self._compiled_phrase_anchors: List[re.Pattern] = []

        self._compile_value_regexes(regex_rules or {})
        self._compile_field_separator_regex(field_patterns or {})
        self._compile_phrase_anchors()
        # context_rules intentionally NOT used as triggers — too noisy.
        # The context_detector still uses them downstream for label promotion;
        # they're just unsuitable as window-discovery anchors.

        logger.info(
            f"[windows] scanner ready (high-precision): "
            f"{len(self._compiled_value_regexes)} value regexes, "
            f"{'1' if self._compiled_field_separator_regex else '0'} field-with-separator regex, "
            f"{len(self._compiled_phrase_anchors)} phrase anchors, "
            f"pad={self.pad_chars}, max_coverage={self.max_coverage:.0%}"
        )

    # ── Compilation (one-time at startup) ────────────────────────────────

    def _compile_value_regexes(self, regex_rules: Dict[str, List[str]]) -> None:
        """Compile every value-pattern regex from regex_rules.yaml.

        We keep them as separate compiled patterns rather than mashing into
        one mega-regex because some have inline lookbehinds/lookaheads that
        don't compose cleanly under alternation. Each entry is stored with
        its label so the same sweep that finds candidate windows can also
        emit final Detection objects, eliminating a duplicate full-text
        regex pass downstream in RegexDetector.

        Mirrors RegexDetector's case-sensitivity rule for PERSON_FULL_NAME so
        the consolidated sweep produces identical output to the old two-pass
        flow.
        """
        for label, patterns in regex_rules.items():
            if not isinstance(patterns, list):
                continue
            case_sensitive = label in _CASE_SENSITIVE_LABELS
            for p in patterns:
                if not isinstance(p, str) or not p.strip():
                    continue
                try:
                    if case_sensitive and "(?i)" not in p:
                        rx = re.compile(p)
                    else:
                        rx = re.compile(p, re.IGNORECASE)
                    self._compiled_value_regexes.append((label, rx))
                except re.error as e:
                    logger.debug(f"[windows] skipping malformed regex for {label}: {e}")

    def _compile_field_separator_regex(self, field_patterns: Dict[str, List[str]]) -> None:
        """Build a single regex matching `label_alias <sep> value` (NOT bare aliases).

        The bare-alias version of this scanner over-flagged in conversation
        because words like "name", "email", "phone" appear in plain dialogue.
        Requiring an explicit separator (`:`, `=`, `-`, `;`, em-dash) makes
        each match a near-certain PII signal: structured `Field: Value` form.

        We pull the same alias list from field_patterns.yaml the
        PatternFieldDetector uses, but wrap each one in `(alias)\\s*[:=\\-;\\u2014]`.
        """
        triggers: List[str] = []
        seen: set = set()
        for label, aliases in field_patterns.items():
            if not isinstance(aliases, list):
                continue
            for alias in aliases:
                if not isinstance(alias, str) or not alias.strip():
                    continue
                norm = alias.strip().lower()
                if norm in seen or len(norm) < 2:
                    continue
                seen.add(norm)
                escaped = re.escape(norm).replace(r"\ ", r"\s+")
                triggers.append(escaped)

        if not triggers:
            self._compiled_field_separator_regex = None
            return

        triggers.sort(key=len, reverse=True)
        # \b(alias)\b followed by optional space and one of the separator chars
        big = r"\b(?:" + "|".join(triggers) + r")\b\s*[:=\-;\u2014]"
        try:
            self._compiled_field_separator_regex = re.compile(big, re.IGNORECASE)
        except re.error as e:
            logger.warning(f"[windows] field-separator regex compile failed: {e}")
            self._compiled_field_separator_regex = None

    def _compile_phrase_anchors(self) -> None:
        """Compile the curated high-precision phrase anchors."""
        for p in self._PHRASE_ANCHORS:
            try:
                self._compiled_phrase_anchors.append(re.compile(p, re.IGNORECASE))
            except re.error as e:
                logger.debug(f"[windows] skipping malformed anchor: {p} ({e})")

    # ── Hot path: scan one row ───────────────────────────────────────────

    def scan(
        self, text: str
    ) -> Tuple[Optional[List[Tuple[int, int]]], List[Detection], WindowScanStats]:
        """Find candidate windows in `text` and return regex detections.

        Returns (windows, regex_detections, stats):
          - windows: sorted, non-overlapping (start, end) tuples in original
            text coordinates. Empty list means "no PII found anywhere — skip
            GLiNER entirely". None means "PII is too dense — fall through to
            a full-text GLiNER scan".
          - regex_detections: full Detection objects produced by the value
            regex sweep. The caller should add these to the detection list
            instead of running RegexDetector — we already did the work here.
          - stats: diagnostic counters for tuning and logging.
        """
        stats = WindowScanStats()
        regex_detections: List[Detection] = []
        if not text:
            return [], regex_detections, stats

        text_len = len(text)
        hits: List[Hit] = []

        # 1. Value-pattern regexes (find SSN/email/phone/CC literal values).
        #    These are the highest-precision triggers — every match is real PII.
        #    Each match also becomes a Detection so the engine can skip a
        #    second full-text sweep in RegexDetector.
        for label, rx in self._compiled_value_regexes:
            for m in rx.finditer(text):
                hits.append(m.start())
                stats.n_regex_hits += 1
                if m.lastindex:
                    text_val = m.group(1)
                    start_idx = m.start(1)
                    end_idx = m.end(1)
                else:
                    text_val = m.group(0)
                    start_idx = m.start()
                    end_idx = m.end()
                regex_detections.append(
                    Detection(
                        label=label,
                        start=start_idx,
                        end=end_idx,
                        text=text_val,
                        score=0.98,
                        source="regex",
                    )
                )

        # Early density bail: if value-regex hits ALONE already imply more
        # than max_coverage of the text would be padded windows, there's no
        # point running phrase anchors or field-separator passes — they can
        # only push coverage higher. Skip straight to full-scan fall-through.
        # Uses a cheap upper-bound estimate (hits * 2*pad) which over-counts
        # on overlapping hits but is correct for the "way over budget" case
        # we want to short-circuit.
        if hits:
            approx_coverage = (len(hits) * 2 * self.pad_chars) / max(1, text_len)
            if approx_coverage > self.max_coverage:
                stats.n_windows_before_merge = len(hits)
                stats.coverage = approx_coverage
                stats.fell_through_to_full_scan = True
                return None, regex_detections, stats

        # 2. Field-with-separator regex (matches `Email: alice@x.com` patterns
        #    but NOT bare "email" mentioned in dialogue).
        if self._compiled_field_separator_regex is not None:
            for m in self._compiled_field_separator_regex.finditer(text):
                hits.append(m.start())
                stats.n_field_hits += 1

        # 3. Curated phrase anchors ("my name is", "i live at", etc.)
        for rx in self._compiled_phrase_anchors:
            for m in rx.finditer(text):
                hits.append(m.start())
                stats.n_context_hits += 1

        if not hits:
            # No triggers found anywhere. Returning [] tells the caller to
            # skip GLiNER entirely. This is the best-case path for filler-heavy
            # rows that contain zero PII.
            return [], regex_detections, stats

        # 4. Expand each hit into a padded window and merge overlaps
        windows = self._merge_windows(hits, text_len)
        stats.n_windows_before_merge = len(hits)
        stats.n_windows_after_merge = len(windows)

        # 5. Coverage check — fall through to full scan if too dense
        coverage = sum(e - s for s, e in windows) / max(1, text_len)
        stats.coverage = coverage
        if coverage > self.max_coverage:
            stats.fell_through_to_full_scan = True
            return None, regex_detections, stats

        return windows, regex_detections, stats

    def regex_sweep_only(self, text: str) -> List[Detection]:
        """Run value-pattern regexes on full text. No windowing, no anchors.

        This is the model-first pipeline's regex stage: GLiNER runs first
        on full text to identify PII regions, then this method provides
        format-validated detections (SSN, email, CC, IBAN, etc.) that are
        higher-precision than GLiNER alone. No window computation is done.
        """
        regex_detections: List[Detection] = []
        if not text:
            return regex_detections
        for label, rx in self._compiled_value_regexes:
            for m in rx.finditer(text):
                if m.lastindex:
                    text_val = m.group(1)
                    start_idx = m.start(1)
                    end_idx = m.end(1)
                else:
                    text_val = m.group(0)
                    start_idx = m.start()
                    end_idx = m.end()
                regex_detections.append(
                    Detection(
                        label=label,
                        start=start_idx,
                        end=end_idx,
                        text=text_val,
                        score=0.98,
                        source="regex",
                    )
                )
        return regex_detections

    def _merge_windows(self, hits: List[Hit], text_len: int) -> List[Tuple[int, int]]:
        """Sort + expand + merge overlapping intervals."""
        if not hits:
            return []
        pad = self.pad_chars
        # Build (start, end) intervals, then sort by start
        intervals = sorted(
            (max(0, h - pad), min(text_len, h + pad)) for h in hits
        )
        merged: List[Tuple[int, int]] = []
        cur_s, cur_e = intervals[0]
        for s, e in intervals[1:]:
            if s <= cur_e:
                if e > cur_e:
                    cur_e = e
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        return merged
