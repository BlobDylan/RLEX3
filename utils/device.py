"""Device selection helpers (CPU / CUDA / Apple MPS)."""

from __future__ import annotations

import os

import torch


def get_torch_device(prefer: str | None = None) -> torch.device:
    """Pick a torch device.

    Order when ``prefer`` is None: CUDA → MPS → CPU.
    Set ``prefer`` to ``\"cpu\"`` / ``\"mps\"`` / ``\"cuda\"`` to force a choice
    (falls back if unavailable).
    """
    # Let unsupported MPS ops fall back to CPU instead of crashing.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if prefer is not None:
        pref = prefer.lower()
        if pref == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if pref == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if pref == "cpu":
            return torch.device("cpu")
        # Unavailable preference → keep going with auto order.

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int, device: torch.device | None = None) -> None:
    """Seed Python stack + the active accelerator."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device is not None and device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def describe_device(device: torch.device) -> str:
    """One-line human summary for notebook logs."""
    parts = [str(device)]
    if device.type == "mps":
        parts.append("(Apple Metal — float32 only; fallback enabled)")
    elif device.type == "cuda":
        parts.append(f"({torch.cuda.get_device_name(device)})")
    elif device.type == "cpu":
        parts.append("(no accelerator)")
    return " ".join(parts)
