"""ComplexEnv — pure from-scratch DQN (no demonstrations, no imitation).

Trains a Double DQN from random initialisation on the image observation. The agent
discovers the key→door→water→lava→goal chain on its own, guided only by:
  * **Monotonically-increasing milestone shaping** — each stage is worth more than the
    last (``enter_right_room`` > ``door_open``) so the agent is pulled *through* the
    doorway instead of camping at it, with the goal dominant. First-time only (anti-farm)
    and a small uniform step penalty; no potential-based Φ term (that eroded intermediate
    progress and made the agent unlearn the key).
  * **Reward scaling** (``reward_scale``) so Q ≈ O(1); large raw rewards otherwise push
    Q ≈ 185 and wreck the value SNR, which stalled every earlier from-scratch run.
  * **n-step returns** for faster credit assignment, and sustained ε-greedy exploration.
  * Optional **RND intrinsic exploration** (``--rnd-coef``) if the doorway still stalls.

A single **RGB** frame is sufficient (inventory is recoverable from what's missing on
screen); grayscale / frame-stacking both hurt.

Usage:
  .venv/bin/python -u scripts/complex_dqn.py
  .venv/bin/python -u scripts/complex_dqn.py --steps 400000 --device mps
  .venv/bin/python -u scripts/complex_dqn.py --rnd-coef 0.1   # add novelty exploration
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms import DQN
from pipelines import make_complex_env
from utils import (
    describe_device,
    get_torch_device,
    plot_training_history,
    save_training_history,
)

OUT_ROOT = ROOT / "graphs" / "complex_dqn"
# Monotonically-increasing 'pull-forward' shaping. Each stage is worth MORE than the
# previous (enter_right_room > door_open), so the agent is pulled *through* the door
# rather than camping at it, and the goal dominates. First-time only (anti-farm). A
# modest uniform step_penalty makes idling mildly negative WITHOUT the Φ-dependent
# erosion of potential-based shaping (which made holding the key net-negative and
# caused the agent to unlearn it). key_drop / leave_right_room stay off.
SHAPING: dict[str, float] = {
    "goal_scale": 50.0,        # dominant terminal reward
    "step_penalty": 0.05,
    "key_pickup": 5.0,
    "door_open": 10.0,
    "enter_right_room": 15.0,  # > door_open: pull the agent THROUGH the doorway
    "leave_right_room": 0.0,
    "key_drop": 0.0,
    "water_pickup": 8.0,
    "lava_extinguish": 15.0,
    "lava_death": 8.0,         # softened (was 15): the lava sits in the right room the
    # agent must now work inside; a big death penalty made it timid exactly where it
    # needs to toggle water onto lava. Still worse than a timeout, so no suicide bias.
}


@dataclass
class Config:
    steps: int = 500_000
    max_steps: int = 300
    # DQN
    n_step: int = 3
    lr: float = 2.5e-4
    gamma: float = 0.99
    reward_scale: float = 0.02    # keep Q ~ O(1) (goal reward is 50)
    rnd_coef: float = 0.0         # RND intrinsic exploration (off by default)
    batch_size: int = 64
    # Speed knob: on MPS each gradient update is ~19 ms, so throughput ≈ (updates/s).
    # train_freq=4 ≈ 205 steps/s (~8 min/100k); lower it (2 or 1) for more updates /
    # better sample efficiency at proportionally slower wall-clock.
    train_freq: int = 4
    gradient_steps: int = 1
    tau: float = 0.005
    target_update_freq: int = 1
    double_dqn: bool = True
    # Sustained exploration: each stage of the chain is discovered inside its own
    # high-ε window, and the later stages (water→lava→goal) become reachable only
    # after ~120k steps — so decay slowly and keep a non-trivial floor so they get
    # the same exploration budget the door did.
    eps_start: float = 1.0
    eps_end: float = 0.15
    eps_decay_steps: int = 300_000
    learning_starts: int = 5_000
    buffer_size: int = 200_000
    n_envs: int = 8              # SyncVectorEnv workers — batches GPU work (~1.8x faster)
    width_mult: int = 2         # 2M params — the capacity that cracked the door→right skill
    n_extra_conv: int = 1
    seed: int = 0
    device: str | None = None


def build_env_fn(cfg: Config):
    def _fn():
        return make_complex_env(
            max_steps=cfg.max_steps, grayscale=False, frame_stack=1, shaping=SHAPING
        )

    return _fn


def run(cfg: Config, out_dir: Path) -> dict:
    device = get_torch_device(cfg.device)
    print(f"device: {describe_device(device)}", flush=True)

    env_fn = build_env_fn(cfg)
    probe = env_fn()
    obs_shape = probe.observation_space.shape
    n_actions = probe.action_space.n
    probe.close()
    print(f"obs={obs_shape}  actions={n_actions}  (single-frame RGB, from scratch)", flush=True)

    agent = DQN(
        obs_shape,
        n_actions,
        device=device,
        seed=cfg.seed,
        lr=cfg.lr,
        gamma=cfg.gamma,
        batch_size=cfg.batch_size,
        train_freq=cfg.train_freq,
        gradient_steps=cfg.gradient_steps,
        tau=cfg.tau,
        target_update_freq=cfg.target_update_freq,
        eps_start=cfg.eps_start,
        eps_end=cfg.eps_end,
        eps_decay_steps=cfg.eps_decay_steps,
        learning_starts=cfg.learning_starts,
        buffer_size=cfg.buffer_size,
        n_step=cfg.n_step,
        reward_scale=cfg.reward_scale,
        rnd_coef=cfg.rnd_coef,
        width_mult=cfg.width_mult,
        n_extra_conv=cfg.n_extra_conv,
        double_dqn=cfg.double_dqn,
    )
    print(f"params={agent.n_parameters():,}  n_step={cfg.n_step}  reward_scale={cfg.reward_scale}  "
          f"n_envs={cfg.n_envs}", flush=True)

    t0 = time.time()
    history = agent.train(env_fn, total_timesteps=cfg.steps, log_every_episodes=10, n_envs=cfg.n_envs)
    minutes = round((time.time() - t0) / 60, 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_training_history(history, out_dir / "history.json")
    try:
        plot_training_history(
            history,
            title="ComplexEnv — from-scratch DQN",
            save_prefix="complex_dqn",
            folder=out_dir,
            window=20,
            show=False,
        )
    except Exception as exc:  # plotting must never sink a long run
        print(f"[plot] skipped: {exc}", flush=True)

    eval_stats = agent.evaluate(env_fn(), n_episodes=50, seed=9999)
    agent.save(out_dir / "agent.pt")

    result = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "eval": eval_stats,
        "minutes": minutes,
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


def _parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=cfg.steps)
    p.add_argument("--max-steps", type=int, default=cfg.max_steps)
    p.add_argument("--n-step", type=int, default=cfg.n_step)
    p.add_argument("--gamma", type=float, default=cfg.gamma)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--reward-scale", type=float, default=cfg.reward_scale)
    p.add_argument("--rnd-coef", type=float, default=cfg.rnd_coef, help="RND intrinsic weight (0 = off)")
    p.add_argument("--eps-end", type=float, default=cfg.eps_end)
    p.add_argument("--eps-decay-steps", type=int, default=cfg.eps_decay_steps)
    p.add_argument("--train-freq", type=int, default=cfg.train_freq)
    p.add_argument("--n-envs", type=int, default=cfg.n_envs, help="SyncVectorEnv workers (GPU batching)")
    p.add_argument("--width-mult", type=int, default=cfg.width_mult)
    p.add_argument("--n-extra-conv", type=int, default=cfg.n_extra_conv)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--device", type=str, default=None, help="cpu / mps / cuda (auto if omitted)")
    p.add_argument("--out", type=str, default=None, help="output dir (default graphs/complex_dqn/<ts>)")
    a = p.parse_args()

    cfg.steps = a.steps
    cfg.max_steps = a.max_steps
    cfg.n_step = a.n_step
    cfg.gamma = a.gamma
    cfg.lr = a.lr
    cfg.reward_scale = a.reward_scale
    cfg.rnd_coef = a.rnd_coef
    cfg.eps_end = a.eps_end
    cfg.eps_decay_steps = a.eps_decay_steps
    cfg.train_freq = a.train_freq
    cfg.n_envs = a.n_envs
    cfg.width_mult = a.width_mult
    cfg.n_extra_conv = a.n_extra_conv
    cfg.seed = a.seed
    cfg.device = a.device
    cfg._out = a.out  # type: ignore[attr-defined]
    return cfg


def main() -> None:
    cfg = _parse_args()
    out = getattr(cfg, "_out", None)
    out_dir = Path(out) if out else OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run(cfg, out_dir)


if __name__ == "__main__":
    main()
