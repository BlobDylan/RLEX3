"""Shared training runner + env factory for the on-policy scripts.

Keeps REINFORCE / PPO consistent with the DQN script: same env factories, same
checkpoint-safe train loop (history / plots / weights saved every VIDEO_EVERY and
on Ctrl+C), same grid rollout videos, same greedy eval + ``result.json``.

DQN keeps its own inline ``run()`` (proven / in-flight); this runner is the shared
path for the newer agents. ComplexEnv shaping is imported from the DQN script so
there is a single source of truth for the tuned reward.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pipelines import make_complex_env, make_simple_env
from utils import grid_rollout_video, plot_training_history, save_training_history

# Single source of truth for the tuned ComplexEnv reward shaping.
from scripts.complex_dqn import SHAPING as COMPLEX_SHAPING

ROOT = Path(__file__).resolve().parents[1]


def make_env_fn(env_name: str, *, max_steps: int) -> Callable[[], Any]:
    """Return a zero-arg factory for a wrapped env. RGB obs for both (color matters)."""
    if env_name == "complex":
        return lambda: make_complex_env(
            max_steps=max_steps, grayscale=False, frame_stack=1, shaping=COMPLEX_SHAPING
        )
    if env_name == "simple":
        return lambda: make_simple_env(max_steps=max_steps, grayscale=False, frame_stack=1)
    raise ValueError(f"unknown env {env_name!r} (expected 'simple' or 'complex')")


def default_out_dir(algo: str, env_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "output" / f"{env_name}_{algo}" / stamp


def run_training(
    agent,
    env_fn: Callable[[], Any],
    out_dir: Path,
    *,
    total_steps: int,
    n_envs: int,
    max_steps: int,
    seed: int = 0,
    log_every_episodes: int = 10,
    eval_episodes: int = 50,
    video_every: int = 100_000,
    title: str = "training",
    config_dict: dict | None = None,
) -> dict:
    """Train ``agent`` with checkpointing + videos + greedy eval. Safe to Ctrl+C."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_env = env_fn()
    _last_video = {"milestone": 0}
    _ckpt: dict = {"history": None}

    def _save_progress(history) -> None:
        if history is None:
            return
        save_training_history(history, out_dir / "history.json")
        try:
            plot_training_history(
                history, title=title, save_prefix=out_dir.parent.name,
                folder=out_dir, window=20, show=False,
            )
        except Exception as exc:  # plotting must never sink a long run
            print(f"[plot] skipped: {exc}", flush=True)
        agent.save(out_dir / "agent.pt")

    def _video_cb(step: int, history) -> None:
        _ckpt["history"] = history
        milestone = step // video_every
        if milestone <= _last_video["milestone"]:
            return
        _last_video["milestone"] = milestone
        _save_progress(history)
        fname = str(video_dir / f"rollout_{milestone * (video_every // 1000)}k.mp4")
        try:
            results = grid_rollout_video(
                agent, video_env, fname, n_episodes=20, max_steps=max_steps,
                fps=8, seed=seed, explore=False,
            )
            rets = ", ".join(f"{r:.0f}" for _, r in results)
            print(f"[video] {fname}  returns=[{rets}]", flush=True)
        except Exception as exc:  # a bad render must never sink a long run
            print(f"[video] skipped at step {step}: {exc}", flush=True)

    t0 = time.time()
    interrupted = False
    history = None
    try:
        history = agent.train(
            env_fn, total_timesteps=total_steps, log_every_episodes=log_every_episodes,
            n_envs=n_envs, callback=_video_cb,
        )
    except KeyboardInterrupt:
        interrupted = True
        history = _ckpt["history"]
        print("\n[interrupted] stopping early — saving graphs / weights / eval …", flush=True)
    minutes = round((time.time() - t0) / 60, 1)
    video_env.close()

    _save_progress(history)  # final graphs + weights (covers completion and early stop)
    eval_stats = agent.evaluate(env_fn(), n_episodes=eval_episodes, seed=9999) if history else {}

    result = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "config": config_dict or {},
        "eval": eval_stats,
        "minutes": minutes,
        "interrupted": interrupted,
        "episodes": agent.total_episodes,
        "steps": agent.total_steps,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))

    print("\n=== RESULT ===", flush=True)
    print(f"eval success_rate = {eval_stats.get('success_rate', 0):.1%}", flush=True)
    for sk in ("stage_key", "stage_door", "stage_right", "stage_water", "stage_lava", "stage_goal"):
        if sk in eval_stats:
            print(f"  {sk:12s} {eval_stats[sk]:.0%}", flush=True)
    print(f"wall time = {minutes} min  |  saved → {out_dir}", flush=True)
    return result
