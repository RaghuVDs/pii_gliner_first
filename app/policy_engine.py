from typing import List
from app.models import Detection

class MaskingPolicyEngine:
    def __init__(self, masking_rules: dict):
        # We ignore partial rules now. Strict placeholders only.
        pass

    def redact(self, text: str, detections: List[Detection]) -> str:
        # Sort in reverse so replacing text doesn't mess up the indices of earlier entities
        sorted_detections = sorted(detections, key=lambda x: x.start, reverse=True)
        redacted_text = text

        for d in sorted_detections:
            # Strictly use the dynamic tag (eg: <PERSON_FIRST_NAME_1>)
            replacement = getattr(d, 'replacement_tag', f"<{d.label}_1>")
            redacted_text = redacted_text[:d.start] + replacement + redacted_text[d.end:]

        return redacted_text