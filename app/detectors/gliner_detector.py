from __future__ import annotations
from typing import List, Optional, Tuple, Dict
from app.models import Detection
import yaml
import os

try:
    import torch
    from gliner import GLiNER
except ImportError:
    GLiNER = None

class GLiNERDetector:
    def __init__(
        self,
        model_name: str = "knowledgator/gliner-pii-large-v1.0",
        threshold: float = 0.35,
        enabled: bool = True,
        taxonomy_path: str = "app/config/pii_taxonomy.yaml"
    ):
        self.enabled = enabled and GLiNER is not None
        self.model_name = model_name
        self.threshold = threshold
        self.model = None

        if self.enabled:
            try:
                # Explicitly allocate CPU threads to prevent thrashing
                cpu_cores = os.cpu_count() or 4
                torch.set_num_threads(cpu_cores)
                
                print(f"Loading GLiNER on CPU using {cpu_cores} threads...")
                self.model = GLiNER.from_pretrained(model_name)
                
            except Exception as e:
                self.enabled = False
                print(f"GLiNER Load Error: {e}")

        self.alias_to_amex_label: Dict[str, str] = {}
        self.gliner_prompt_labels: List[str] = []
        self.label_thresholds: Dict[str, float] = {}
        self._load_taxonomy(taxonomy_path)

    def _load_taxonomy(self, path: str):
        with open(path, 'r') as f:
            taxonomy = yaml.safe_load(f)
            
        for group in taxonomy.values():
            for amex_label, data in group.items():
                custom_threshold = data.get("threshold", self.threshold)
                for alias in data.get("gliner_aliases", []):
                    self.gliner_prompt_labels.append(alias)
                    self.alias_to_amex_label[alias] = amex_label
                    self.label_thresholds[alias] = custom_threshold
                    
        self.gliner_prompt_labels = list(set(self.gliner_prompt_labels))

    def detect(self, text: str) -> List[Detection]:
        if not self.enabled or not self.model or not text.strip():
            return []

        detections: List[Detection] = []
        chunks = self._sliding_window_chunker(text, window_size=1500, overlap=150)
        
        if not chunks:
            return []

        # Hands all text to PyTorch at once so it can optimize the math across cores
        chunk_texts = [c[0] for c in chunks]
        chunk_starts = [c[1] for c in chunks]

        batch_preds = self.model.batch_predict_entities(
            chunk_texts,
            self.gliner_prompt_labels,
            threshold=self.threshold,
        )

        for preds, chunk_start in zip(batch_preds, chunk_starts):
            for pred in preds:
                found_alias = pred["label"]
                score = float(pred.get("score", 0.0))
                
                if score < self.label_thresholds.get(found_alias, self.threshold):
                    continue

                strict_amex_label = self.alias_to_amex_label.get(found_alias, "UNKNOWN_PII")
                start = chunk_start + int(pred["start"])
                end = chunk_start + int(pred["end"])
                value = text[start:end]

                if not value.strip(): continue

                detections.append(
                    Detection(
                        label=strict_amex_label,
                        text=value,
                        start=start,
                        end=end,
                        score=score,
                        source="gliner",
                        meta={"gliner_alias": found_alias}
                    )
                )
        return detections

    def _sliding_window_chunker(self, text: str, window_size: int, overlap: int) -> List[Tuple[str, int]]:
        chunks = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + window_size, text_length)
            if end < text_length:
                last_space = text.rfind(' ', start, end)
                if last_space != -1: end = last_space
            chunks.append((text[start:end], start))
            start = end - overlap
            if start <= chunks[-1][1]: start = end
        return chunks