"""Train PPO on SimpleRoomEnv or ComplexEnv (pixels only, from scratch).

Usage:
  .venv/bin/python -u scripts/train_ppo.py --env simple  --device mps
  .venv/bin/python -u scripts/train_ppo.py --env complex --device mps --steps 2000000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms import PPO
from utils import describe_device, get_torch_device
from scripts._common import default_out_dir, make_env_fn, run_training


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", choices=["simple", "complex"], required=True)
    p.add_argument("--steps", type=int, default=None, help="env steps (default: 1M simple / 3M complex)")
    p.add_argument("--max-steps", type=int, default=None, help="episode length (default: 100 simple / 200 complex)")
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

    steps = a.steps if a.steps is not None else (1_000_000 if a.env == "simple" else 3_000_000)
    max_steps = a.max_steps if a.max_steps is not None else (100 if a.env == "simple" else 200)

    device = get_torch_device(a.device)
    print(f"device: {describe_device(device)}", flush=True)
    env_fn = make_env_fn(a.env, max_steps=max_steps)
    probe = env_fn()
    obs_shape, n_actions = probe.observation_space.shape, probe.action_space.n
    probe.close()
    print(f"PPO on {a.env}  obs={obs_shape}  actions={n_actions}", flush=True)

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

    out_dir = Path(a.out) if a.out else default_out_dir("ppo", a.env)
    run_training(
        agent, env_fn, out_dir,
        total_steps=steps, n_envs=a.n_envs, max_steps=max_steps, seed=a.seed,
        title=f"{a.env.capitalize()}Env — PPO",
        config_dict={**vars(a), "steps": steps, "max_steps": max_steps},
    )


if __name__ == "__main__":
    main()
