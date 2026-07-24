"""Shared CLI builders for the per-algorithm/env train scripts.

The six ``train_<algo>_<env>.py`` entry points are thin wrappers that call one of these
functions with a fixed env. Keeping the argparse + agent construction here avoids
duplicating it across the wrappers. (``train_dqn_complex.py`` is intentionally its own
full script — it carries the tuned shaping, ε-ramp, PER and resume logic.)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from algorithms import DQN, PPO, REINFORCE
from utils import describe_device, get_torch_device
from scripts._common import default_out_dir, make_env_fn, run_training


def _probe(env_fn):
    probe = env_fn()
    obs_shape, n_actions = probe.observation_space.shape, probe.action_space.n
    probe.close()
    return obs_shape, n_actions


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------
def run_ppo(env_name: str) -> None:
    p = argparse.ArgumentParser(description=f"Train PPO on {env_name} env (pixels, from scratch).")
    p.add_argument("--steps", type=int, default=None, help="env steps (default: 1M simple / 3M complex)")
    p.add_argument("--max-steps", type=int, default=None, help="episode length (default: 100 / 200)")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--no-anneal-lr", action="store_true", help="disable linear lr decay to 0")
    p.add_argument("--width-mult", type=int, default=2)
    p.add_argument("--n-extra-conv", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    a = p.parse_args()

    steps = a.steps if a.steps is not None else (1_000_000 if env_name == "simple" else 3_000_000)
    max_steps = a.max_steps if a.max_steps is not None else (100 if env_name == "simple" else 200)

    device = get_torch_device(a.device)
    print(f"device: {describe_device(device)}", flush=True)
    env_fn = make_env_fn(env_name, max_steps=max_steps)
    obs_shape, n_actions = _probe(env_fn)
    print(f"PPO on {env_name}  obs={obs_shape}  actions={n_actions}", flush=True)

    agent = PPO(
        obs_shape, n_actions, device=device, seed=a.seed,
        gamma=a.gamma, gae_lambda=a.gae_lambda, lr=a.lr,
        clip_coef=a.clip_coef, ent_coef=a.ent_coef, vf_coef=a.vf_coef,
        rollout_steps=a.rollout_steps, update_epochs=a.update_epochs,
        num_minibatches=a.num_minibatches, n_envs=a.n_envs,
        anneal_lr=not a.no_anneal_lr,
        width_mult=a.width_mult, n_extra_conv=a.n_extra_conv,
    )
    print(f"params={agent.n_parameters():,}", flush=True)

    out_dir = Path(a.out) if a.out else default_out_dir("ppo", env_name)
    run_training(
        agent, env_fn, out_dir,
        total_steps=steps, n_envs=a.n_envs, max_steps=max_steps, seed=a.seed,
        title=f"{env_name.capitalize()}Env — PPO",
        config_dict={**vars(a), "env": env_name, "steps": steps, "max_steps": max_steps},
    )


# ---------------------------------------------------------------------------
# REINFORCE
# ---------------------------------------------------------------------------
def run_reinforce(env_name: str) -> None:
    p = argparse.ArgumentParser(description=f"Train REINFORCE on {env_name} env (pixels, from scratch).")
    p.add_argument("--steps", type=int, default=None, help="env steps (default: 300k simple / 2M complex)")
    p.add_argument("--max-steps", type=int, default=None, help="episode length (default: 100 / 200)")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--entropy-coef", type=float, default=0.05)
    p.add_argument("--episodes-per-update", type=int, default=16)
    p.add_argument("--width-mult", type=int, default=2)
    p.add_argument("--n-extra-conv", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    a = p.parse_args()

    steps = a.steps if a.steps is not None else (300_000 if env_name == "simple" else 2_000_000)
    max_steps = a.max_steps if a.max_steps is not None else (100 if env_name == "simple" else 200)

    device = get_torch_device(a.device)
    print(f"device: {describe_device(device)}", flush=True)
    env_fn = make_env_fn(env_name, max_steps=max_steps)
    obs_shape, n_actions = _probe(env_fn)
    print(f"REINFORCE on {env_name}  obs={obs_shape}  actions={n_actions}", flush=True)

    agent = REINFORCE(
        obs_shape, n_actions, device=device, seed=a.seed,
        gamma=a.gamma, lr=a.lr, entropy_coef=a.entropy_coef,
        episodes_per_update=a.episodes_per_update,
        width_mult=a.width_mult, n_extra_conv=a.n_extra_conv,
    )
    print(f"params={agent.n_parameters():,}", flush=True)

    out_dir = Path(a.out) if a.out else default_out_dir("reinforce", env_name)
    run_training(
        agent, env_fn, out_dir,
        total_steps=steps, n_envs=a.n_envs, max_steps=max_steps, seed=a.seed,
        title=f"{env_name.capitalize()}Env — REINFORCE",
        config_dict={**vars(a), "env": env_name, "steps": steps, "max_steps": max_steps},
    )


# ---------------------------------------------------------------------------
# DQN (SimpleRoom only — ComplexEnv DQN lives in train_dqn_complex.py)
# ---------------------------------------------------------------------------
def run_dqn_simple() -> None:
    p = argparse.ArgumentParser(description="Train DQN on the SimpleRoom env (pixels, from scratch).")
    p.add_argument("--steps", type=int, default=60_000, help="env steps (solves ~40-50k)")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--n-envs", type=int, default=8)
    # SimpleRoom "winner" hypers (see REPORT_MEMOS §7).
    p.add_argument("--lr", type=float, default=2.8e-4)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-freq", type=int, default=1)
    # With n_envs=8 the vec loop learns once per 8 env steps; gradient_steps=4 restores
    # ~0.5 updates/env-step so the update budget matches the single-env winner (~20-30k
    # updates solves SimpleRoom). Without this, n_envs=8 is 8x under-trained.
    p.add_argument("--gradient-steps", type=int, default=4)
    p.add_argument("--tau", type=float, default=0.001)
    p.add_argument("--eps-end", type=float, default=0.15)
    p.add_argument("--eps-decay-steps", type=int, default=20_000)
    p.add_argument("--learning-starts", type=int, default=1_000)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--width-mult", type=int, default=1)
    p.add_argument("--n-extra-conv", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    a = p.parse_args()

    device = get_torch_device(a.device)
    print(f"device: {describe_device(device)}", flush=True)
    env_fn = make_env_fn("simple", max_steps=a.max_steps)
    obs_shape, n_actions = _probe(env_fn)
    print(f"DQN on simple  obs={obs_shape}  actions={n_actions}", flush=True)

    agent = DQN(
        obs_shape, n_actions, device=device, seed=a.seed,
        lr=a.lr, gamma=a.gamma, batch_size=a.batch_size, train_freq=a.train_freq,
        gradient_steps=a.gradient_steps,
        tau=a.tau, eps_end=a.eps_end, eps_decay_steps=a.eps_decay_steps,
        learning_starts=a.learning_starts, buffer_size=a.buffer_size,
        double_dqn=True, width_mult=a.width_mult, n_extra_conv=a.n_extra_conv,
    )
    print(f"params={agent.n_parameters():,}", flush=True)

    out_dir = Path(a.out) if a.out else default_out_dir("dqn", "simple")
    run_training(
        agent, env_fn, out_dir,
        total_steps=a.steps, n_envs=a.n_envs, max_steps=a.max_steps, seed=a.seed,
        video_every=5_000,
        title="SimpleRoomEnv — DQN",
        config_dict={**vars(a), "env": "simple"},
    )
