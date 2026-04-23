"""Device detection and selection.

Single source of truth for "which device(s) does this process use".
No hardcoded device strings anywhere else in the codebase — every module
that needs a device asks DeviceManager.

Behaviour:
  - CUDA available  → use cuda:0 (and report all visible GPUs for sharding)
  - MPS available   → use mps
  - else            → cpu (with a loud warning, since the SLO is unlikely)

The manager is a process-level singleton so repeated lookups are free.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger("pii_engine.perf.device")

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


@dataclass
class DeviceInfo:
    """Snapshot of a single compute device."""
    index: int
    kind: str  # "cuda" | "mps" | "cpu"
    name: str
    total_vram_bytes: int = 0  # 0 for cpu/mps
    compute_capability: Optional[str] = None  # cuda only

    @property
    def torch_device(self) -> str:
        if self.kind == "cuda":
            return f"cuda:{self.index}"
        return self.kind

    @property
    def total_vram_gb(self) -> float:
        return self.total_vram_bytes / 1e9


@dataclass
class DeviceManager:
    """Process-wide device discovery and selection.

    All decisions are runtime-derived. Nothing is hardcoded to a specific GPU.
    """
    primary: DeviceInfo
    all_devices: List[DeviceInfo] = field(default_factory=list)

    @classmethod
    def detect(cls) -> "DeviceManager":
        """Probe the host and return a populated DeviceManager.

        Order of preference: CUDA > MPS > CPU.
        """
        if torch is None:
            return cls._cpu_only("torch not installed")

        # CUDA path: enumerate all visible GPUs
        if torch.cuda.is_available():
            devices: List[DeviceInfo] = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devices.append(
                    DeviceInfo(
                        index=i,
                        kind="cuda",
                        name=props.name,
                        total_vram_bytes=props.total_memory,
                        compute_capability=f"{props.major}.{props.minor}",
                    )
                )
            primary = devices[0]
            logger.info(
                f"[device] CUDA: {len(devices)} GPU(s); primary={primary.name} "
                f"({primary.total_vram_gb:.1f} GB, sm_{primary.compute_capability})"
            )
            return cls(primary=primary, all_devices=devices)

        # MPS path (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = DeviceInfo(index=0, kind="mps", name="Apple MPS")
            logger.info("[device] MPS available")
            return cls(primary=dev, all_devices=[dev])

        # CPU fallback
        return cls._cpu_only("no GPU detected")

    @classmethod
    def _cpu_only(cls, reason: str) -> "DeviceManager":
        n_cores = os.cpu_count() or 1
        dev = DeviceInfo(index=0, kind="cpu", name=f"CPU ({n_cores} cores)")
        logger.warning(
            f"[device] Falling back to CPU ({reason}). "
            f"<1s/row latency target will be very difficult on CPU for long inputs."
        )
        return cls(primary=dev, all_devices=[dev])

    @property
    def has_cuda(self) -> bool:
        return self.primary.kind == "cuda"

    @property
    def num_gpus(self) -> int:
        return sum(1 for d in self.all_devices if d.kind == "cuda")

    def cache_key(self, model_name: str) -> str:
        """Stable key for the autotune cache.

        Same GPU model + same NER model + same torch version → same tuning.
        """
        torch_ver = str(getattr(torch, "__version__", "unknown")) if torch else "none"
        return f"{self.primary.kind}::{self.primary.name}::{model_name}::torch{torch_ver}"


# ── Process-wide singleton (lazy) ───────────────────────────────────────
_singleton: Optional[DeviceManager] = None


def get_device_manager() -> DeviceManager:
    """Return the process-wide DeviceManager, detecting on first call."""
    global _singleton
    if _singleton is None:
        _singleton = DeviceManager.detect()
    return _singleton


def reset_device_manager() -> None:
    """Force re-detection on next get_device_manager() call. Test-only."""
    global _singleton
    _singleton = None
