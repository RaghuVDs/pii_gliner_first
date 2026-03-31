"""Confidence calibration for PII detection scores.

Provides:
  - PlattScaler: Logistic calibration for GLiNER scores (learns a, b params)
  - TemperatureScaler: Single-parameter calibration for LSTM logits

Both produce calibrated probabilities where a 0.80 score means ~80% chance
of being a correct detection.
"""
from __future__ import annotations

import math
import os
import logging
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger("pii_engine.calibration")

ML_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
CALIBRATION_PATH = os.path.join(ML_SAVE_DIR, "calibration.yaml")


class PlattScaler:
    """Platt scaling for GLiNER confidence scores.

    Fits: calibrated = sigmoid(a * raw_score + b)
    where a, b are learned from (score, is_correct) pairs.
    """

    def __init__(self):
        self.a: float = 1.0   # slope (default = identity-ish)
        self.b: float = 0.0   # intercept
        self._fitted = False

    def fit(self, scores: List[float], labels: List[bool], lr: float = 0.01, epochs: int = 200):
        """Fit Platt scaling parameters from (score, is_correct) pairs.

        Args:
            scores: Raw confidence scores from GLiNER.
            labels: True if the detection was correct, False if false positive.
            lr: Learning rate for gradient descent.
            epochs: Number of optimization iterations.
        """
        if len(scores) < 20:
            logger.warning("[CALIBRATION] Too few examples for Platt scaling (need >= 20)")
            return

        a, b = 0.0, 0.0
        n = len(scores)

        for _ in range(epochs):
            grad_a, grad_b = 0.0, 0.0
            for s, y in zip(scores, labels):
                logit = a * s + b
                p = 1.0 / (1.0 + math.exp(-logit)) if logit > -500 else 0.0
                target = 1.0 if y else 0.0
                err = p - target
                grad_a += err * s
                grad_b += err
            a -= lr * grad_a / n
            b -= lr * grad_b / n

        self.a = a
        self.b = b
        self._fitted = True
        logger.info(f"[CALIBRATION] Platt scaling fitted: a={a:.4f}, b={b:.4f}")

    def calibrate(self, score: float) -> float:
        """Apply Platt scaling to a raw score."""
        if not self._fitted:
            return score
        logit = self.a * score + self.b
        if logit > 500:
            return 1.0
        if logit < -500:
            return 0.0
        return 1.0 / (1.0 + math.exp(-logit))

    def is_fitted(self) -> bool:
        return self._fitted


class TemperatureScaler:
    """Temperature scaling for LSTM logits.

    Divides logits by learned temperature T before softmax:
        calibrated = softmax(logits / T)

    T > 1 → softer (less confident) predictions
    T < 1 → sharper (more confident) predictions
    """

    def __init__(self):
        self.temperature: float = 1.0  # default = no scaling
        self._fitted = False

    def fit(self, logits_list: List[List[float]], labels: List[int], lr: float = 0.01, epochs: int = 100):
        """Fit temperature parameter from validation set.

        Args:
            logits_list: List of raw logit vectors from the LSTM.
            labels: Correct label indices.
            lr: Learning rate.
            epochs: Optimization iterations.
        """
        if len(logits_list) < 20:
            logger.warning("[CALIBRATION] Too few examples for temperature scaling (need >= 20)")
            return

        T = 1.5  # Start slightly above 1 (assume model is overconfident)

        for _ in range(epochs):
            grad = 0.0
            for logits, y in zip(logits_list, labels):
                scaled = [l / T for l in logits]
                max_s = max(scaled)
                exp_s = [math.exp(s - max_s) for s in scaled]
                Z = sum(exp_s)
                probs = [e / Z for e in exp_s]

                # Gradient of NLL w.r.t. T
                for j, p in enumerate(probs):
                    target = 1.0 if j == y else 0.0
                    grad += (p - target) * (-logits[j] / (T * T))

            T -= lr * grad / len(logits_list)
            T = max(0.1, min(10.0, T))  # Clamp to reasonable range

        self.temperature = T
        self._fitted = True
        logger.info(f"[CALIBRATION] Temperature scaling fitted: T={T:.4f}")

    def scale(self, logits: List[float]) -> List[float]:
        """Apply temperature scaling to logits."""
        if not self._fitted:
            return logits
        return [l / self.temperature for l in logits]

    def is_fitted(self) -> bool:
        return self._fitted


class CalibrationManager:
    """Manages loading/saving of calibration parameters."""

    def __init__(self):
        self.platt = PlattScaler()
        self.temperature = TemperatureScaler()
        self._load()

    def _load(self):
        """Load calibration parameters from disk."""
        if not os.path.exists(CALIBRATION_PATH):
            return
        try:
            with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not data:
                return

            if "platt" in data:
                self.platt.a = data["platt"].get("a", 1.0)
                self.platt.b = data["platt"].get("b", 0.0)
                self.platt._fitted = data["platt"].get("fitted", False)

            if "temperature" in data:
                self.temperature.temperature = data["temperature"].get("T", 1.0)
                self.temperature._fitted = data["temperature"].get("fitted", False)

            logger.info("[CALIBRATION] Loaded calibration parameters from disk")
        except Exception as e:
            logger.warning(f"[CALIBRATION] Failed to load: {e}")

    def save(self):
        """Save calibration parameters to disk."""
        data = {
            "platt": {
                "a": self.platt.a,
                "b": self.platt.b,
                "fitted": self.platt._fitted,
            },
            "temperature": {
                "T": self.temperature.temperature,
                "fitted": self.temperature._fitted,
            },
        }
        os.makedirs(os.path.dirname(CALIBRATION_PATH), exist_ok=True)
        tmp = CALIBRATION_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False)
        os.replace(tmp, CALIBRATION_PATH)
        logger.info("[CALIBRATION] Saved calibration parameters")

    def calibrate_gliner_score(self, score: float) -> float:
        """Calibrate a GLiNER detection score."""
        return self.platt.calibrate(score)

    def calibrate_lstm_logits(self, logits: List[float]) -> List[float]:
        """Apply temperature scaling to LSTM logits."""
        return self.temperature.scale(logits)
