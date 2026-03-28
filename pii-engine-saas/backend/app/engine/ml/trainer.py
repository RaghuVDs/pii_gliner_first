"""Production self-training loop for the PII pattern classifier.

Training discipline:
  - Stratified train/val split (85/15) for real accuracy measurement
  - AdamW optimizer with OneCycleLR scheduler (cosine annealing + warmup)
  - Gradient clipping (max_norm=1.0) to prevent LSTM gradient explosion
  - Label smoothing (0.1) to prevent overconfident predictions
  - Early stopping (patience=20) based on validation loss
  - Best model checkpoint (lowest val loss, not last epoch)
  - Mixed precision training (torch.cuda.amp) for CUDA speed
  - pin_memory=True for CUDA DataLoader
  - Data deduplication to prevent memorizing repeated examples
  - Adaptive hyperparameters based on dataset size

All data flows on CUDA when available. Zero PII stored.
"""
from __future__ import annotations

import os
import re
import hashlib
import logging
import threading
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("pii_engine.lstm")

import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from app.engine.models import Detection
from app.engine.adaptive_learning import _value_to_structure, _extract_safe_neighborhood
from app.engine.ml.model import PIIPatternModel, encode_pattern, MAX_PATTERN_LEN

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
ML_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
TRAINING_DATA_PATH = os.path.join(CONFIG_DIR, "training_data.yaml")
MODEL_PATH = os.path.join(ML_SAVE_DIR, "pattern_model.pt")
VOCAB_PATH = os.path.join(ML_SAVE_DIR, "vocab.yaml")
METRICS_LOG_PATH = os.path.join(ML_SAVE_DIR, "metrics_log.yaml")

# ── Hyperparameters ──────────────────────────────────────────────────
MIN_EXAMPLES_TO_TRAIN = 30
RETRAIN_INTERVAL = 100
COLLECTION_SCORE_THRESHOLD = 0.40
PSEUDO_LABEL_THRESHOLD = 0.75
VALIDATION_SPLIT = 0.15
EARLY_STOPPING_PATIENCE = 20
LABEL_SMOOTHING = 0.1
MAX_GRAD_NORM = 1.0


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


def _dedup_key(ex: Dict) -> str:
    """Create a dedup key from structure + label + top keywords."""
    struct = ex.get("structure", "")
    label = ex.get("label", "")
    kws = tuple(sorted(ex.get("keywords", [])[:3]))
    raw = f"{struct}|{label}|{kws}"
    return hashlib.md5(raw.encode()).hexdigest()


def _deduplicate_examples(examples: List[Dict]) -> List[Dict]:
    """Remove exact-duplicate training examples while keeping diverse contexts."""
    seen = set()
    unique = []
    for ex in examples:
        key = _dedup_key(ex)
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    return unique


def _stratified_split(
    examples: List[Dict], val_ratio: float = 0.15
) -> Tuple[List[Dict], List[Dict]]:
    """Stratified train/val split ensuring each label appears in both sets."""
    by_label: Dict[str, List[Dict]] = defaultdict(list)
    for ex in examples:
        by_label[ex.get("label", "UNKNOWN_PII")].append(ex)

    train_set, val_set = [], []
    for label, exs in by_label.items():
        n_val = max(1, int(len(exs) * val_ratio))
        # Ensure at least 1 in train
        if len(exs) <= 2:
            train_set.extend(exs)
            val_set.extend(exs)  # duplicate into val for tiny classes
        else:
            val_set.extend(exs[:n_val])
            train_set.extend(exs[n_val:])

    return train_set, val_set


# ── Stopwords for keyword extraction ────────────────────────────────
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "and", "but", "or", "if", "not", "no", "yes", "this", "that",
    "it", "its", "you", "your", "my", "we", "our", "they", "them",
    "i", "me", "he", "she", "his", "her", "so", "than", "too",
    "just", "also", "about", "up", "down", "like", "well", "now",
    "ok", "sure", "please", "thank", "thanks", "hi", "hello",
    "let", "get", "got", "go", "know", "see", "right",
})


def _compute_per_label_metrics(
    all_preds: List[int], all_labels: List[int], idx_to_label: Dict[int, str],
) -> Dict[str, Any]:
    """Compute precision, recall, F1 per label + macro/weighted averages + confusion matrix.

    Returns a dict with:
      - per_label: {label: {precision, recall, f1, support}}
      - macro_precision, macro_recall, macro_f1
      - weighted_precision, weighted_recall, weighted_f1
      - confusion: {true_label: {predicted_label: count}}
    """
    # Count TP, FP, FN per label index
    label_indices = sorted(set(all_labels + all_preds))
    tp: Dict[int, int] = defaultdict(int)
    fp: Dict[int, int] = defaultdict(int)
    fn: Dict[int, int] = defaultdict(int)
    confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for pred, true in zip(all_preds, all_labels):
        true_name = idx_to_label.get(true, f"IDX_{true}")
        pred_name = idx_to_label.get(pred, f"IDX_{pred}")
        confusion[true_name][pred_name] += 1
        if pred == true:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1

    per_label = {}
    total_support = 0
    macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
    w_p, w_r, w_f1 = 0.0, 0.0, 0.0

    for idx in label_indices:
        label_name = idx_to_label.get(idx, f"IDX_{idx}")
        support = tp[idx] + fn[idx]
        total_support += support

        precision = tp[idx] / max(tp[idx] + fp[idx], 1)
        recall = tp[idx] / max(tp[idx] + fn[idx], 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        per_label[label_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        macro_p += precision
        macro_r += recall
        macro_f1 += f1
        w_p += precision * support
        w_r += recall * support
        w_f1 += f1 * support

    n_labels = max(len(label_indices), 1)
    total_support = max(total_support, 1)

    # Convert confusion to regular dict for YAML serialization
    confusion_dict = {k: dict(v) for k, v in confusion.items()}

    return {
        "per_label": per_label,
        "macro_precision": round(macro_p / n_labels, 4),
        "macro_recall": round(macro_r / n_labels, 4),
        "macro_f1": round(macro_f1 / n_labels, 4),
        "weighted_precision": round(w_p / total_support, 4),
        "weighted_recall": round(w_r / total_support, 4),
        "weighted_f1": round(w_f1 / total_support, 4),
        "confusion_matrix": confusion_dict,
    }


def _sample_history(values: List[float], max_points: int = 25) -> List[float]:
    """Sample a training history list to at most max_points for compact logging.

    Keeps: first 3, every N-th in middle, last 3.
    """
    if len(values) <= max_points:
        return values
    head = values[:3]
    tail = values[-3:]
    middle_pool = values[3:-3]
    step = max(1, len(middle_pool) // (max_points - 6))
    middle = middle_pool[::step]
    return head + middle + tail


def _save_metrics_log(metrics_entry: Dict[str, Any]) -> None:
    """Append a metrics entry to the persistent metrics log file.

    Each entry represents one training run with full metrics.
    """
    log = _load_yaml(METRICS_LOG_PATH)
    if not isinstance(log, dict):
        log = {}
    runs = log.get("runs", [])
    runs.append(metrics_entry)
    log["runs"] = runs
    log["total_retrains"] = len(runs)
    log["last_retrain"] = metrics_entry.get("timestamp", datetime.now().isoformat())
    _save_yaml(METRICS_LOG_PATH, log)


class PIIPatternDataset(Dataset):
    """PyTorch dataset for PII-safe training examples.
    All tensors returned as CPU tensors — DataLoader with pin_memory handles transfer.
    """

    def __init__(
        self,
        examples: List[Dict],
        label_to_idx: Dict[str, int],
        keyword_to_idx: Dict[str, int],
        source_to_idx: Dict[str, int],
    ):
        self.examples = examples
        self.label_to_idx = label_to_idx
        self.keyword_to_idx = keyword_to_idx
        self.source_to_idx = source_to_idx
        self.num_keywords = len(keyword_to_idx)
        self.num_labels = len(label_to_idx)
        self.num_sources = len(source_to_idx)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ex = self.examples[idx]

        # Encode structure pattern
        char_ids = torch.tensor(encode_pattern(ex.get("structure", "")), dtype=torch.long)

        # Build context feature vector
        context = torch.zeros(self.num_keywords + self.num_labels + self.num_sources + 2)

        # Keywords multi-hot
        for kw in ex.get("keywords", []):
            if kw in self.keyword_to_idx:
                context[self.keyword_to_idx[kw]] = 1.0

        # Co-occurring labels multi-hot
        off_labels = self.num_keywords
        for cl in ex.get("co_labels", []):
            if cl in self.label_to_idx:
                context[off_labels + self.label_to_idx[cl]] = 1.0

        # Source one-hot
        off_source = self.num_keywords + self.num_labels
        source = ex.get("source", "unknown")
        if source in self.source_to_idx:
            context[off_source + self.source_to_idx[source]] = 1.0

        # Score and normalized length
        context[-2] = ex.get("score", 0.0)
        context[-1] = min(ex.get("length", 0) / 100.0, 1.0)

        # Label index
        label = ex.get("label", "UNKNOWN_PII")
        label_idx = self.label_to_idx.get(label, self.label_to_idx.get("UNKNOWN_PII", 0))

        return char_ids, context, torch.tensor(label_idx, dtype=torch.long)


class SelfTrainer:
    """Production self-trainer with proper ML discipline.

    Training features:
      - Stratified train/val split
      - AdamW + OneCycleLR (cosine warmup)
      - Gradient clipping, label smoothing, early stopping
      - Mixed precision (AMP) on CUDA
      - Best model checkpoint
      - Data deduplication
    """

    def __init__(
        self,
        training_data_path: str = TRAINING_DATA_PATH,
        model_path: str = MODEL_PATH,
        vocab_path: str = VOCAB_PATH,
    ):
        self.training_data_path = training_data_path
        self.model_path = model_path
        self.vocab_path = vocab_path
        self._lock = threading.Lock()
        self._buffer: List[Dict] = []
        self._examples_since_last_train = 0

        # Model and vocab
        self.model: Optional[PIIPatternModel] = None
        self.label_to_idx: Dict[str, int] = {}
        self.idx_to_label: Dict[int, str] = {}
        self.keyword_to_idx: Dict[str, int] = {}
        self.source_to_idx: Dict[str, int] = {
            "gliner": 0, "regex": 1, "field_label": 2,
            "context": 3, "derived": 4, "propagated": 5,
            "pattern_lstm": 6, "unknown": 7,
        }

        # CUDA device selection
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")

        # Mixed precision scaler (CUDA only)
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        self._load_model()

    # ── Data Collection ─────────────────────────────────────────────

    def collect_from_detections(self, detections: List[Detection], full_text: str) -> int:
        """Collect PII-safe training examples from high-confidence detections."""
        if not detections:
            return 0

        new_examples = []
        for d in detections:
            if d.score < COLLECTION_SCORE_THRESHOLD:
                continue
            val = (d.text or "").strip()
            if not val or len(val) < 2:
                continue

            safe_neighborhood = _extract_safe_neighborhood(
                full_text, d.start, d.end, detections, pad=150
            )
            keywords = self._extract_keywords(safe_neighborhood)
            co_labels = sorted(set(
                o.label for o in detections
                if o is not d and abs(o.start - d.start) < 500 and o.label != d.label
            ))

            new_examples.append({
                "structure": _value_to_structure(val),
                "label": d.label,
                "length": len(val),
                "keywords": keywords[:12],
                "co_labels": co_labels[:10],
                "source": d.source,
                "score": round(d.score, 4),
                "confidence_type": "supervised",
                "timestamp": datetime.now().isoformat(),
            })

        if new_examples:
            with self._lock:
                self._buffer.extend(new_examples)
        return len(new_examples)

    def flush_to_disk(self) -> int:
        """Write buffered examples to training_data.yaml."""
        with self._lock:
            to_write = self._buffer[:]
            self._buffer.clear()

        if not to_write:
            return 0

        data = _load_yaml(self.training_data_path)
        if not isinstance(data, dict):
            data = {}
        existing = data.get("examples", [])
        existing.extend(to_write)

        # Deduplicate on flush to keep file clean
        existing = _deduplicate_examples(existing)

        data["examples"] = existing
        data["total_count"] = len(existing)
        data["last_updated"] = datetime.now().isoformat()
        _save_yaml(self.training_data_path, data)

        self._examples_since_last_train += len(to_write)
        return len(to_write)

    def collect_from_promoted(self, promoted_rules: List[Dict]) -> int:
        """Ingest promoted rules as human-verified training examples (highest confidence).

        Called by engine.promote_rules(auto=True) to feed confirmed patterns
        into the LSTM training pipeline. These are the highest-quality examples
        because a human (or auto-threshold) verified them.

        Args:
            promoted_rules: List of promoted rule dicts, each with:
                - label: str (the confirmed PII label)
                - keywords: List[str]
                - structure_patterns: List[str] (value structures seen)
                - co_occurring_labels: List[str]
                - seen_count: int

        Returns:
            Number of training examples added.
        """
        new_examples = []
        for rule in promoted_rules:
            label = rule.get("label", "UNKNOWN_PII")
            keywords = rule.get("keywords", [])
            co_labels = rule.get("co_occurring_labels", [])
            structures = rule.get("structure_patterns", [])

            if not structures:
                # If no structures recorded, create a placeholder from keywords
                structures = [""]

            for struct in structures:
                new_examples.append({
                    "structure": struct,
                    "label": label,
                    "length": len(struct),
                    "keywords": keywords[:12],
                    "co_labels": co_labels[:10],
                    "source": "context",
                    "score": 0.95,
                    "confidence_type": "human_verified",
                    "timestamp": datetime.now().isoformat(),
                })

        if new_examples:
            with self._lock:
                self._buffer.extend(new_examples)

        return len(new_examples)

    def _extract_keywords(self, text: str, top_n: int = 12) -> List[str]:
        """Extract context keywords from sanitized text."""
        tokens = re.findall(r"\b[a-z][a-z\-]{2,}\b", text.lower())
        meaningful = [t for t in tokens if t not in _STOPWORDS and len(t) >= 3]
        return [w for w, _ in Counter(meaningful).most_common(top_n)]

    # ── Vocabulary ──────────────────────────────────────────────────

    def _build_vocab(self, examples: List[Dict]) -> None:
        """Build label and keyword vocabularies from training data."""
        labels = sorted(set(ex.get("label", "UNKNOWN_PII") for ex in examples))
        if "UNKNOWN_PII" not in labels:
            labels.append("UNKNOWN_PII")
        self.label_to_idx = {label: i for i, label in enumerate(labels)}
        self.idx_to_label = {i: label for label, i in self.label_to_idx.items()}

        kw_counter: Counter = Counter()
        for ex in examples:
            kw_counter.update(ex.get("keywords", []))
        top_kw = [kw for kw, _ in kw_counter.most_common(400)]
        self.keyword_to_idx = {kw: i for i, kw in enumerate(top_kw)}

    def _save_vocab(self) -> None:
        os.makedirs(ML_SAVE_DIR, exist_ok=True)
        _save_yaml(self.vocab_path, {
            "label_to_idx": self.label_to_idx,
            "keyword_to_idx": self.keyword_to_idx,
            "source_to_idx": self.source_to_idx,
        })

    def _load_vocab(self) -> bool:
        if not os.path.exists(self.vocab_path):
            return False
        vocab = _load_yaml(self.vocab_path)
        if not vocab:
            return False
        self.label_to_idx = vocab.get("label_to_idx", {})
        self.idx_to_label = {int(i): label for label, i in self.label_to_idx.items()}
        self.keyword_to_idx = vocab.get("keyword_to_idx", {})
        src = vocab.get("source_to_idx")
        if src:
            self.source_to_idx = src
        return bool(self.label_to_idx)

    # ── Training ────────────────────────────────────────────────────

    def should_retrain(self) -> bool:
        if self._examples_since_last_train < RETRAIN_INTERVAL:
            return False
        data = _load_yaml(self.training_data_path)
        total = data.get("total_count", 0) if isinstance(data, dict) else 0
        return total >= MIN_EXAMPLES_TO_TRAIN

    def retrain(self, epochs: int | None = None) -> Dict[str, Any]:
        """Train the model with full ML discipline.

        Args:
            epochs: Max epochs. If None, auto-calculated from dataset size.

        Returns:
            Training stats dict.
        """
        data = _load_yaml(self.training_data_path)
        all_examples = data.get("examples", []) if isinstance(data, dict) else []
        all_examples = _deduplicate_examples(all_examples)

        if len(all_examples) < MIN_EXAMPLES_TO_TRAIN:
            return {
                "status": "skipped",
                "reason": f"Need {MIN_EXAMPLES_TO_TRAIN} examples, have {len(all_examples)}",
            }

        # Build vocabularies
        self._build_vocab(all_examples)
        self._save_vocab()

        num_labels = len(self.label_to_idx)
        num_keywords = len(self.keyword_to_idx)
        num_sources = len(self.source_to_idx)

        # Stratified train/val split
        train_examples, val_examples = _stratified_split(all_examples, VALIDATION_SPLIT)

        # Adaptive hyperparameters based on dataset size
        n = len(train_examples)
        if epochs is None:
            if n < 100:
                epochs = 200
            elif n < 500:
                epochs = 120
            else:
                epochs = 80

        batch_size = min(64, max(8, n // 6))
        lr = 3e-3 if n < 200 else 2e-3
        weight_decay = 1e-3 if n < 200 else 5e-4

        # Datasets and loaders
        train_ds = PIIPatternDataset(train_examples, self.label_to_idx, self.keyword_to_idx, self.source_to_idx)
        val_ds = PIIPatternDataset(val_examples, self.label_to_idx, self.keyword_to_idx, self.source_to_idx)

        pin = self.device.type == "cuda"
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            pin_memory=pin, num_workers=0, drop_last=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            pin_memory=pin, num_workers=0,
        )

        # Build model
        self.model = PIIPatternModel(
            num_labels=num_labels,
            num_keywords=num_keywords,
            num_sources=num_sources,
        ).to(self.device)

        # Optimizer: AdamW (decoupled weight decay)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999),
        )

        # Scheduler: OneCycleLR (warmup + cosine annealing, steps per batch)
        total_steps = len(train_loader) * epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, total_steps=max(total_steps, 1),
            pct_start=0.2, anneal_strategy="cos", div_factor=10, final_div_factor=100,
        )
        # Suppress false positive warning — OneCycleLR is designed to step per batch
        import warnings
        warnings.filterwarnings("ignore", "Detected call of .lr_scheduler.step.*")

        # Class-weighted loss with label smoothing
        label_counts = Counter(ex.get("label") for ex in train_examples)
        weights = torch.ones(num_labels, device=self.device)
        for label, count in label_counts.items():
            idx = self.label_to_idx.get(label)
            if idx is not None:
                weights[idx] = 1.0 / max(count, 1)
        weights = weights / weights.sum() * num_labels
        criterion = nn.CrossEntropyLoss(
            weight=weights, label_smoothing=LABEL_SMOOTHING,
        )

        # Training state
        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        for epoch in range(epochs):
            # ── Train ──
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for char_ids, context, labels in train_loader:
                char_ids = char_ids.to(self.device, non_blocking=True)
                context = context.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        logits = self.model(char_ids, context)
                        loss = criterion(logits, labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    logits = self.model(char_ids, context)
                    loss = criterion(logits, labels)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                    optimizer.step()

                scheduler.step()

                train_loss += loss.item() * labels.size(0)
                train_correct += (logits.argmax(-1) == labels).sum().item()
                train_total += labels.size(0)

            avg_train_loss = train_loss / max(train_total, 1)
            train_acc = train_correct / max(train_total, 1)

            # ── Validate ──
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for char_ids, context, labels in val_loader:
                    char_ids = char_ids.to(self.device, non_blocking=True)
                    context = context.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    if self.use_amp:
                        with torch.amp.autocast("cuda"):
                            logits = self.model(char_ids, context)
                            loss = criterion(logits, labels)
                    else:
                        logits = self.model(char_ids, context)
                        loss = criterion(logits, labels)

                    val_loss += loss.item() * labels.size(0)
                    val_correct += (logits.argmax(-1) == labels).sum().item()
                    val_total += labels.size(0)

            avg_val_loss = val_loss / max(val_total, 1)
            val_acc = val_correct / max(val_total, 1)

            history["train_loss"].append(round(avg_train_loss, 4))
            history["val_loss"].append(round(avg_val_loss, 4))
            history["train_acc"].append(round(train_acc, 4))
            history["val_acc"].append(round(val_acc, 4))

            # ── Early stopping on best val loss ──
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        self.model.eval()

        self._save_model()
        self._examples_since_last_train = 0

        stopped_epoch = len(history["train_loss"])
        best_epoch = history["val_loss"].index(round(best_val_loss, 4)) + 1

        # ── Per-label metrics on val set using best model ──
        all_val_preds: List[int] = []
        all_val_labels: List[int] = []
        with torch.no_grad():
            for char_ids, context, labels in val_loader:
                char_ids = char_ids.to(self.device, non_blocking=True)
                context = context.to(self.device, non_blocking=True)
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        logits = self.model(char_ids, context)
                else:
                    logits = self.model(char_ids, context)
                all_val_preds.extend(logits.argmax(-1).cpu().tolist())
                all_val_labels.extend(labels.tolist())

        label_metrics = _compute_per_label_metrics(all_val_preds, all_val_labels, self.idx_to_label)

        # ── Per-label metrics on train set ──
        all_train_preds: List[int] = []
        all_train_labels: List[int] = []
        with torch.no_grad():
            for char_ids, context, labels in train_loader:
                char_ids = char_ids.to(self.device, non_blocking=True)
                context = context.to(self.device, non_blocking=True)
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        logits = self.model(char_ids, context)
                else:
                    logits = self.model(char_ids, context)
                all_train_preds.extend(logits.argmax(-1).cpu().tolist())
                all_train_labels.extend(labels.tolist())

        train_metrics = _compute_per_label_metrics(all_train_preds, all_train_labels, self.idx_to_label)

        # ── Build result ──
        result = {
            "status": "trained",
            "timestamp": datetime.now().isoformat(),
            "device": str(self.device),
            "mixed_precision": self.use_amp,
            "num_examples": len(all_examples),
            "train_size": len(train_examples),
            "val_size": len(val_examples),
            "num_labels": num_labels,
            "num_keywords": num_keywords,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "label_smoothing": LABEL_SMOOTHING,
            "max_grad_norm": MAX_GRAD_NORM,
            "max_epochs": epochs,
            "stopped_epoch": stopped_epoch,
            "best_epoch": best_epoch,
            "early_stopped": stopped_epoch < epochs,
            "best_val_loss": round(best_val_loss, 4),
            # ── Aggregate accuracy ──
            "train_accuracy": train_metrics["weighted_recall"],
            "val_accuracy": label_metrics["weighted_recall"],
            # ── Per-label validation metrics ──
            "val_macro_precision": label_metrics["macro_precision"],
            "val_macro_recall": label_metrics["macro_recall"],
            "val_macro_f1": label_metrics["macro_f1"],
            "val_weighted_precision": label_metrics["weighted_precision"],
            "val_weighted_recall": label_metrics["weighted_recall"],
            "val_weighted_f1": label_metrics["weighted_f1"],
            "val_per_label": label_metrics["per_label"],
            "val_confusion_matrix": label_metrics["confusion_matrix"],
            # ── Per-label training metrics ──
            "train_macro_f1": train_metrics["macro_f1"],
            "train_weighted_f1": train_metrics["weighted_f1"],
            # ── Training curves (sampled: first 5, every 10th, last 5) ──
            "history_sample": {
                "train_loss": _sample_history(history["train_loss"]),
                "val_loss": _sample_history(history["val_loss"]),
                "train_acc": _sample_history(history["train_acc"]),
                "val_acc": _sample_history(history["val_acc"]),
            },
        }

        # ── Persist to metrics log ──
        _save_metrics_log(result)

        logger.info(
            f"[LSTM] Retrain complete — {len(all_examples)} examples, "
            f"{num_labels} labels, val_weighted_f1={result['val_weighted_f1']:.3f}, "
            f"best_epoch={best_epoch}/{stopped_epoch}, device={self.device}"
        )

        return result

    def _save_model(self) -> None:
        if self.model is None:
            return
        os.makedirs(ML_SAVE_DIR, exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)

    def _load_model(self) -> bool:
        if not os.path.exists(self.model_path) or not os.path.exists(self.vocab_path):
            return False
        if not self._load_vocab():
            return False
        try:
            self.model = PIIPatternModel(
                num_labels=len(self.label_to_idx),
                num_keywords=len(self.keyword_to_idx),
                num_sources=len(self.source_to_idx),
            )
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device, weights_only=True)
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info(
                f"[LSTM] Model loaded from disk — {len(self.label_to_idx)} labels, "
                f"device={self.device}, keywords={len(self.keyword_to_idx)}"
            )
            return True
        except Exception as e:
            logger.warning(f"[LSTM] Failed to load model: {e}")
            self.model = None
            return False

    # ── Inference ───────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self.model is not None and len(self.label_to_idx) > 0

    def predict(
        self,
        structure: str,
        length: int,
        keywords: List[str],
        co_labels: List[str],
        source: str = "unknown",
        score: float = 0.5,
    ) -> Tuple[str, float]:
        """Predict PII label from PII-safe features. Returns (label, confidence)."""
        if not self.is_ready():
            return ("UNKNOWN_PII", 0.0)

        self.model.eval()
        char_ids = torch.tensor([encode_pattern(structure)], dtype=torch.long, device=self.device)

        num_kw = len(self.keyword_to_idx)
        num_lb = len(self.label_to_idx)
        num_sr = len(self.source_to_idx)
        context = torch.zeros(1, num_kw + num_lb + num_sr + 2, device=self.device)

        for kw in keywords:
            if kw in self.keyword_to_idx:
                context[0, self.keyword_to_idx[kw]] = 1.0
        for cl in co_labels:
            if cl in self.label_to_idx:
                context[0, num_kw + self.label_to_idx[cl]] = 1.0
        if source in self.source_to_idx:
            context[0, num_kw + num_lb + self.source_to_idx[source]] = 1.0
        context[0, -2] = score
        context[0, -1] = min(length / 100.0, 1.0)

        with torch.no_grad():
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(char_ids, context)
            else:
                logits = self.model(char_ids, context)
            probs = torch.softmax(logits, dim=-1)
            confidence, pred_idx = probs.max(dim=-1)

        return (self.idx_to_label.get(pred_idx.item(), "UNKNOWN_PII"), round(confidence.item(), 4))

    def predict_for_detection(
        self, detection: Detection, all_detections: List[Detection], full_text: str,
    ) -> Tuple[str, float]:
        """Predict label for a specific detection."""
        if not self.is_ready():
            return ("UNKNOWN_PII", 0.0)
        val = (detection.text or "").strip()
        if not val:
            return ("UNKNOWN_PII", 0.0)

        safe_nb = _extract_safe_neighborhood(full_text, detection.start, detection.end, all_detections, pad=150)
        keywords = self._extract_keywords(safe_nb)
        co_labels = sorted(set(
            d.label for d in all_detections
            if d is not detection and abs(d.start - detection.start) < 500 and d.label != detection.label
        ))
        return self.predict(
            structure=_value_to_structure(val), length=len(val),
            keywords=keywords, co_labels=co_labels,
            source=detection.source, score=detection.score,
        )

    # ── Pseudo-Labeling ─────────────────────────────────────────────

    def pseudo_label(self, detections: List[Detection], full_text: str) -> List[Detection]:
        """Upgrade UNKNOWN_PII / low-confidence detections using model predictions."""
        if not self.is_ready():
            return detections

        updated = []
        new_training = []

        for d in detections:
            should_upgrade = (
                d.label == "UNKNOWN_PII"
                or (d.score < 0.55 and d.source == "gliner")       # GLiNER uncertain
                or (d.score < 0.50 and d.source == "context")      # Context-promoted but weak
            )
            if not should_upgrade:
                updated.append(d)
                continue

            predicted_label, confidence = self.predict_for_detection(d, detections, full_text)

            if confidence >= PSEUDO_LABEL_THRESHOLD and predicted_label != "UNKNOWN_PII":
                upgraded = Detection(
                    label=predicted_label, text=d.text, start=d.start, end=d.end,
                    score=confidence, source="pattern_lstm",
                    meta={**d.meta, "original_label": d.label, "original_source": d.source,
                          "original_score": d.score, "lstm_confidence": confidence},
                )
                updated.append(upgraded)
                logger.debug(
                    f"[LSTM] Upgraded: {d.label}→{predicted_label} "
                    f"(conf={confidence:.3f}, was {d.source}:{d.score:.2f})"
                )

                val = (d.text or "").strip()
                if val:
                    safe_nb = _extract_safe_neighborhood(full_text, d.start, d.end, detections, pad=150)
                    new_training.append({
                        "structure": _value_to_structure(val), "label": predicted_label,
                        "length": len(val), "keywords": self._extract_keywords(safe_nb)[:12],
                        "co_labels": sorted(set(
                            o.label for o in detections
                            if o is not d and abs(o.start - d.start) < 500 and o.label != predicted_label
                        ))[:10],
                        "source": d.source, "score": round(confidence, 4),
                        "confidence_type": "pseudo_labeled", "timestamp": datetime.now().isoformat(),
                    })
            else:
                updated.append(d)

        if new_training:
            with self._lock:
                self._buffer.extend(new_training)

        return updated

    # ── Stats ───────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        data = _load_yaml(self.training_data_path)
        examples = data.get("examples", []) if isinstance(data, dict) else []

        if not examples:
            return {
                "model_ready": self.is_ready(),
                "total_examples": 0, "unique_labels": 0,
                "status": "collecting_data",
                "need_for_training": MIN_EXAMPLES_TO_TRAIN,
                "device": str(self.device),
            }

        label_counts = Counter(ex.get("label") for ex in examples)
        conf_types = Counter(ex.get("confidence_type", "unknown") for ex in examples)

        return {
            "model_ready": self.is_ready(),
            "total_examples": len(examples),
            "unique_labels": len(label_counts),
            "top_labels": dict(label_counts.most_common(10)),
            "confidence_breakdown": dict(conf_types),
            "examples_since_last_train": self._examples_since_last_train,
            "retrain_interval": RETRAIN_INTERVAL,
            "min_examples": MIN_EXAMPLES_TO_TRAIN,
            "device": str(self.device),
            "mixed_precision": self.use_amp,
        }

    def known_labels(self) -> List[str]:
        """Return the list of PII labels the LSTM can currently predict.

        Empty if the model hasn't been trained yet.
        """
        if not self.is_ready():
            return []
        return sorted(self.label_to_idx.keys())

    def get_metrics_history(self) -> Dict[str, Any]:
        """Return the full metrics log from all training runs.

        Each run entry contains: per-label precision/recall/F1, confusion matrix,
        macro/weighted averages, training curves, hyperparameters, and timestamps.
        """
        return _load_yaml(METRICS_LOG_PATH)
