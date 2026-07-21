"""Lightweight random hyperparameter search for DQN on a frozen env pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

from algorithms.dqn import DQN


@dataclass
class DQNSearchSpace:
    """Ranges used by ``sample_dqn_hparams`` (edit here to steer the search)."""

    lr_log10: tuple[float, float] = (-4.5, -3.2)  # ~3e-5 .. ~6e-4
    gammas: tuple[float, ...] = (0.95, 0.99, 0.995)
    batch_sizes: tuple[int, ...] = (32, 64, 128)
    train_freqs: tuple[int, ...] = (1, 2, 4, 8)
    taus: tuple[float, ...] = (0.001, 0.005, 0.01)  # soft target only
    eps_ends: tuple[float, ...] = (0.02, 0.05, 0.1, 0.15)
    eps_decay_steps: tuple[int, ...] = (20_000, 40_000, 60_000, 80_000)
    learning_starts: tuple[int, ...] = (500, 1_000, 2_000)


def sample_dqn_hparams(
    rng: np.random.Generator,
    space: DQNSearchSpace | None = None,
) -> dict[str, Any]:
    """Draw one DQN hyperparameter config."""
    space = space or DQNSearchSpace()
    lo, hi = space.lr_log10
    return {
        "lr": float(10 ** rng.uniform(lo, hi)),
        "gamma": float(rng.choice(space.gammas)),
        "batch_size": int(rng.choice(space.batch_sizes)),
        "train_freq": int(rng.choice(space.train_freqs)),
        "tau": float(rng.choice(space.taus)),
        "eps_start": 1.0,
        "eps_end": float(rng.choice(space.eps_ends)),
        "eps_decay_steps": int(rng.choice(space.eps_decay_steps)),
        "learning_starts": int(rng.choice(space.learning_starts)),
        "double_dqn": True,
        "buffer_size": 50_000,
    }


def run_dqn_trial(
    env_fn: Callable[[], Any],
    *,
    obs_shape: tuple[int, ...],
    n_actions: int,
    device: str | Any,
    hparams: dict[str, Any],
    total_timesteps: int = 20_000,
    eval_episodes: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    """Train one DQN config and score it with greedy eval success_rate."""
    agent = DQN(
        obs_shape,
        n_actions,
        device=device,
        seed=seed,
        **hparams,
    )
    history = agent.train(env_fn, total_timesteps=total_timesteps, log_every=0)

    eval_env = env_fn()
    metrics = agent.evaluate(eval_env, n_episodes=eval_episodes, seed=10_000 + seed)
    diag = agent.diagnose_greedy(eval_env, seed=42)
    eval_env.close()

    late = history["episode_success"][-20:] if history["episode_success"] else [0.0]
    return {
        "hparams": dict(hparams),
        "eval": metrics,
        "diagnose": {
            "action_counts": diag["action_counts"],
            "success": diag["success"],
            "return": diag["return"],
        },
        "train_succ20_late": float(np.mean(late)),
        "total_steps": agent.total_steps,
        "total_episodes": agent.total_episodes,
        "agent": agent,
    }


def random_search_dqn(
    env_fn: Callable[[], Any],
    *,
    obs_shape: tuple[int, ...],
    n_actions: int,
    device: str | Any,
    n_trials: int = 8,
    total_timesteps: int = 20_000,
    eval_episodes: int = 10,
    seed: int = 0,
    space: DQNSearchSpace | None = None,
) -> list[dict[str, Any]]:
    """Run ``n_trials`` random configs; return results sorted by eval success_rate."""
    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []

    for t in range(n_trials):
        hp = sample_dqn_hparams(rng, space)
        trial_seed = seed + 1000 * (t + 1)
        print(f"\n=== trial {t + 1}/{n_trials}  seed={trial_seed} ===")
        print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in hp.items()})
        out = run_dqn_trial(
            env_fn,
            obs_shape=obs_shape,
            n_actions=n_actions,
            device=device,
            hparams=hp,
            total_timesteps=total_timesteps,
            eval_episodes=eval_episodes,
            seed=trial_seed,
        )
        # drop agent from stored table to keep memory light (caller can retrain best)
        agent = out.pop("agent")
        del agent
        print(
            f"→ eval success={out['eval']['success_rate']:.0%}  "
            f"return={out['eval']['mean_return']:.2f}  "
            f"succ20_late={out['train_succ20_late']:.0%}  "
            f"diag_counts={out['diagnose']['action_counts']}"
        )
        results.append(out)

    results.sort(key=lambda r: r["eval"]["success_rate"], reverse=True)
    return results


def format_search_leaderboard(results: list[dict[str, Any]], top_k: int = 5) -> str:
    """Pretty text table of the best trials."""
    lines = ["rank  succ  ret     late20  lr        eps_end  tau    train_freq  batch"]
    for i, r in enumerate(results[:top_k], start=1):
        hp = r["hparams"]
        lines.append(
            f"{i:4d}  {r['eval']['success_rate']:4.0%}  "
            f"{r['eval']['mean_return']:6.1f}  "
            f"{r['train_succ20_late']:5.0%}  "
            f"{hp['lr']:.2e}  {hp['eps_end']:.2f}    "
            f"{hp['tau']:.3f}  {hp['train_freq']:10d}  {hp['batch_size']:5d}"
        )
    return "\n".join(lines)
