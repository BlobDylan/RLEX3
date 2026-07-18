"""Training-curve helpers — save assignment graphs under ``graphs/``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np


def graphs_dir(folder: str | Path = "graphs") -> Path:
    """Return (and create) the project ``graphs/`` directory."""
    path = Path(folder)
    if not path.is_absolute():
        # Prefer repo root when running from a notebook cwd.
        here = Path.cwd()
        candidate = here / path
        if not candidate.exists() and (here / "algorithms").exists():
            candidate = here / path
        path = candidate
    path.mkdir(parents=True, exist_ok=True)
    return path


def rolling_mean(values: Sequence[float], window: int = 20) -> np.ndarray:
    """Causal rolling mean (same length as ``values``; early points use fewer samples)."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return x
    w = max(1, int(window))
    out = np.empty_like(x)
    csum = np.cumsum(x, dtype=np.float64)
    for i in range(x.size):
        j0 = max(0, i + 1 - w)
        total = csum[i] - (csum[j0 - 1] if j0 > 0 else 0.0)
        out[i] = total / (i - j0 + 1)
    return out


def save_training_history(
    history: dict[str, list[float]],
    path: str | Path,
) -> Path:
    """Persist a DQN-style history dict as JSON (lists of floats)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: [float(v) for v in vals] for k, vals in history.items()}
    path.write_text(json.dumps(serializable, indent=2))
    return path


def load_training_history(path: str | Path) -> dict[str, list[float]]:
    """Load history JSON written by ``save_training_history``."""
    data = json.loads(Path(path).read_text())
    return {k: [float(v) for v in vals] for k, vals in data.items()}


def plot_training_history(
    history: dict[str, list[float]],
    *,
    title: str = "Training",
    save_prefix: str = "train",
    folder: str | Path = "graphs",
    window: int = 20,
    show: bool = True,
) -> dict[str, Path]:
    """Plot assignment training curves and save PNGs under ``graphs/``.

    Required views (episode on x-axis; rolling averages where useful):
      - episode return
      - episode length (steps)
      - success rate (rolling)
      - cumulative env steps vs episode

    Returns a map of figure name → saved path.
    """
    out_dir = graphs_dir(folder)
    returns = np.asarray(history.get("episode_return", []), dtype=np.float64)
    lengths = np.asarray(history.get("episode_length", []), dtype=np.float64)
    success = np.asarray(history.get("episode_success", []), dtype=np.float64)
    steps = np.asarray(history.get("steps", []), dtype=np.float64)

    if returns.size == 0:
        raise ValueError("history has no episode_return entries — train first")

    episodes = np.arange(1, returns.size + 1)
    ret_ma = rolling_mean(returns, window)
    len_ma = rolling_mean(lengths, window) if lengths.size else ret_ma
    succ_ma = rolling_mean(success, window) if success.size else np.zeros_like(returns)
    if steps.size != returns.size:
        # Fallback: cumulative sum of episode lengths.
        steps = np.cumsum(lengths) if lengths.size else np.arange(1, returns.size + 1)

    saved: dict[str, Path] = {}

    def _save(fig: plt.Figure, name: str) -> Path:
        path = out_dir / f"{save_prefix}_{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        saved[name] = path
        return path

    # --- 1) Return ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episodes, returns, color="0.75", lw=0.8, label="episode")
    ax.plot(episodes, ret_ma, color="C0", lw=2, label=f"mean{window}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(f"{title} — return")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    _save(fig, "return")
    if not show:
        plt.close(fig)

    # --- 2) Length ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episodes, lengths, color="0.75", lw=0.8, label="episode")
    ax.plot(episodes, len_ma, color="C1", lw=2, label=f"mean{window}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps per episode")
    ax.set_title(f"{title} — episode length")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    _save(fig, "length")
    if not show:
        plt.close(fig)

    # --- 3) Success rate ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episodes, success, color="0.75", lw=0.8, label="episode (0/1)")
    ax.plot(episodes, succ_ma, color="C2", lw=2, label=f"mean{window}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{title} — success")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    _save(fig, "success")
    if not show:
        plt.close(fig)

    # --- 4) Cumulative env steps ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episodes, steps, color="C3", lw=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative env steps")
    ax.set_title(f"{title} — training budget")
    ax.grid(True, alpha=0.3)
    _save(fig, "cum_steps")
    if not show:
        plt.close(fig)

    # --- Combined panel (report-friendly) ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(episodes, returns, color="0.8", lw=0.6)
    ax.plot(episodes, ret_ma, color="C0", lw=2)
    ax.set_title("Return")
    ax.set_xlabel("Episode")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(episodes, lengths, color="0.8", lw=0.6)
    ax.plot(episodes, len_ma, color="C1", lw=2)
    ax.set_title("Episode length")
    ax.set_xlabel("Episode")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(episodes, success, color="0.8", lw=0.6)
    ax.plot(episodes, succ_ma, color="C2", lw=2)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Success rate")
    ax.set_xlabel("Episode")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(episodes, steps, color="C3", lw=2)
    ax.set_title("Cumulative env steps")
    ax.set_xlabel("Episode")
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    _save(fig, "overview")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved


def pick_history(
    candidates: dict[str, Any],
) -> tuple[str, dict[str, list[float]]]:
    """Return the first non-empty ``(name, history)`` from ``candidates``."""
    for name, hist in candidates.items():
        if isinstance(hist, dict) and hist.get("episode_return"):
            return name, hist
    raise KeyError(
        "No training history in memory. Re-run a train cell "
        f"(tried: {list(candidates)}) or load a saved JSON."
    )
