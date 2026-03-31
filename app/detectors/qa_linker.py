"""Question-Answer PII linker for call transcripts.

In call transcripts, agents ASK for PII and customers ANSWER with it:
    Agent: "Can you verify your address?"
    Customer: "Rd. #29 Milwaukee Oregon 97267"

The agent's QUESTION tells us what the customer's ANSWER contains.
We don't need to recognize the format of the answer — we just need to
link the question to the response and label accordingly.

This works for ANY format, ANY PII type, ANY language — because it
understands the CONVERSATION STRUCTURE, not the data format.
"""
from __future__ import annotations

import re
import logging
from typing import List, Dict, Optional, Tuple

from app.models import Detection

logger = logging.getLogger("pii_engine.qa_linker")

# Question patterns that indicate the next response contains PII.
# Maps question keywords → PII label that the answer should receive.
# These aren't PII format patterns — they're QUESTION INTENT patterns.
QA_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Address questions
    (re.compile(r'(?i)(?:what(?:\'s| is)|verify|confirm|give me|provide)\s+'
                r'(?:your|the|his|her)\s+'
                r'(?:home |mailing |billing |street |business |physical )?'
                r'address'),
     "STREET_ADDRESS"),

    # Account/policy number questions
    (re.compile(r'(?i)(?:what(?:\'s| is)|verify|confirm|give me|provide)\s+'
                r'(?:your|the|his|her)\s+'
                r'(?:account|policy|member|reference|claim|case|ticket|order)\s*'
                r'(?:number|#|id|no\.?)?'),
     "UNKNOWN_PII"),

    # SSN questions
    (re.compile(r'(?i)(?:what(?:\'s| is)|verify|confirm|give me|provide)\s+'
                r'(?:your|the|his|her)\s+'
                r'(?:social|ssn|social security)'),
     "SSN"),

    # DOB questions
    (re.compile(r'(?i)(?:what(?:\'s| is)|verify|confirm|give me|provide)\s+'
                r'(?:your|the|his|her)\s+'
                r'(?:date of birth|dob|birth ?date|birthday)'),
     "DATE_OF_BIRTH"),

    # Phone questions
    (re.compile(r'(?i)(?:what(?:\'s| is)|verify|confirm|give me|provide)\s+'
                r'(?:your|the|his|her)\s+'
                r'(?:phone|cell|mobile|telephone|contact)\s*(?:number)?'),
     "PHONE_NUMBER"),

    # Email questions
    (re.compile(r'(?i)(?:what(?:\'s| is)|verify|confirm|give me|provide)\s+'
                r'(?:your|the|his|her)\s+'
                r'(?:email|e-mail)'),
     "EMAIL_ADDRESS"),

    # Name questions
    (re.compile(r'(?i)(?:what(?:\'s| is)|verify|confirm|give me|provide)\s+'
                r'(?:your|the|his|her)\s+'
                r'(?:full |legal )?name'),
     "PERSON_FULL_NAME"),

    # ID/License questions
    (re.compile(r'(?i)(?:what(?:\'s| is)|verify|confirm|give me|provide)\s+'
                r'(?:your|the|his|her)\s+'
                r'(?:driver\'?s? license|id|passport|license)\s*(?:number)?'),
     "DRIVERS_LICENSE_NUMBER"),

    # Generic "verify your X" — catch-all for any PII type
    (re.compile(r'(?i)(?:can (?:you |I )?|please )?'
                r'(?:verify|confirm|provide|give me)\s+'
                r'(?:your|the)\s+'
                r'([a-zA-Z][a-zA-Z\s]{2,30}?)(?:\?|$|\.|\s*[-—])'),
     "_DYNAMIC"),  # Label extracted from the question itself
]

# Speaker turn patterns for transcript parsing
_TURN_RE = re.compile(
    r'(?:^|\n)\s*'
    r'(Agent|Customer|Caller|Representative|Rep|Advisor|Banker|CSR)\s*'
    r'[-:]\s*',
    re.IGNORECASE | re.MULTILINE,
)


class QALinker:
    """Links agent questions to customer answers in call transcripts."""

    def detect(
        self, text: str, existing_detections: List[Detection] = None
    ) -> List[Detection]:
        """Detect PII by linking agent questions to customer responses.

        Parses the transcript into turns, identifies agent questions about PII,
        and labels the customer's response accordingly.
        """
        covered = self._build_covered_set(existing_detections or [])
        turns = self._parse_turns(text)
        detections: List[Detection] = []

        for i, (speaker, turn_text, turn_start, turn_end) in enumerate(turns):
            # Only process agent turns as questions
            if speaker.lower() not in ("agent", "representative", "rep", "advisor", "banker", "csr"):
                continue

            # Check if this turn contains a PII question
            for pattern, label in QA_PATTERNS:
                match = pattern.search(turn_text)
                if not match:
                    continue

                # Handle dynamic label extraction
                actual_label = label
                if label == "_DYNAMIC":
                    # Extract the PII type from the question itself
                    if match.lastindex and match.lastindex >= 1:
                        field_name = match.group(1).strip().lower()
                        actual_label = "UNKNOWN_PII"
                    else:
                        continue

                # Find the NEXT customer turn (the answer)
                next_customer = self._find_next_customer_turn(turns, i)
                if next_customer is None:
                    continue

                _, answer_text, answer_start, answer_end = next_customer

                # The entire customer response (or first meaningful segment) is the PII
                # Extract the first clause/value from the answer
                pii_value, pii_start, pii_end = self._extract_answer_value(
                    answer_text, answer_start
                )

                if not pii_value or len(pii_value) < 3:
                    continue

                # Skip if already covered
                if self._is_covered(pii_start, pii_end, covered):
                    continue

                meta = {"qa_question": turn_text[:80].strip(), "qa_label_hint": actual_label}

                detections.append(Detection(
                    label=actual_label,
                    text=pii_value,
                    start=pii_start,
                    end=pii_end,
                    score=0.60,
                    source="qa_linker",
                    meta=meta,
                ))
                break  # Only one match per agent turn

        return detections

    def _parse_turns(self, text: str) -> List[Tuple[str, str, int, int]]:
        """Parse transcript into (speaker, text, start, end) turns."""
        turns = []
        matches = list(_TURN_RE.finditer(text))

        for i, m in enumerate(matches):
            speaker = m.group(1)
            turn_start = m.end()  # After "Agent: " / "Customer: "
            turn_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            turn_text = text[turn_start:turn_end].strip()
            turns.append((speaker, turn_text, turn_start, turn_end))

        return turns

    def _find_next_customer_turn(
        self, turns: List[Tuple], agent_idx: int
    ) -> Optional[Tuple]:
        """Find the next customer/caller turn after an agent turn."""
        for j in range(agent_idx + 1, len(turns)):
            speaker = turns[j][0]
            if speaker.lower() in ("customer", "caller"):
                return turns[j]
            # If another agent turn comes first, the question wasn't answered
            if speaker.lower() in ("agent", "representative", "rep", "advisor", "banker", "csr"):
                return None
        return None

    def _extract_answer_value(
        self, answer_text: str, answer_start: int
    ) -> Tuple[str, int, int]:
        """Extract the PII value from a customer's answer.

        Takes the first sentence/clause of the answer, which usually
        contains the direct answer to the agent's question.
        """
        # Split on sentence boundaries
        first_clause = re.split(r'[.;!?\n]', answer_text)[0].strip()

        # Remove conversational filler
        filler = re.compile(r'^(?:sure|yes|yeah|ok|okay|of course|absolutely|right|um|uh|so|well)[,.]?\s*', re.IGNORECASE)
        first_clause = filler.sub('', first_clause).strip()

        # Remove "it's" / "that's" prefix
        pronoun = re.compile(r'^(?:it\'?s|that\'?s|here\'?s|they\'?re)\s+', re.IGNORECASE)
        first_clause = pronoun.sub('', first_clause).strip()

        if not first_clause:
            return ("", 0, 0)

        # Find the actual position in the original text
        idx = answer_text.find(first_clause)
        if idx == -1:
            idx = 0
        pii_start = answer_start + idx
        pii_end = pii_start + len(first_clause)

        return (first_clause, pii_start, pii_end)

    @staticmethod
    def _build_covered_set(detections: List[Detection]) -> set:
        covered = set()
        for d in detections:
            covered.update(range(d.start, d.end))
        return covered

    @staticmethod
    def _is_covered(start: int, end: int, covered: set) -> bool:
        span_len = end - start
        if span_len <= 0:
            return True
        overlap = len(set(range(start, end)) & covered)
        return overlap > span_len * 0.5
