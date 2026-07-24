"""Train PPO on SimpleRoomEnv — fast-convergence config (greedy ≈ 90%+ by ~30k steps).

The generic PPO defaults under-converged here (stochastic policy solved it but the greedy
argmax couldn't; see REPORT_MEMOS §5.5). This script uses the fast-convergence recipe:

  * **Orthogonal init** (sqrt(2) trunk, 0.01 policy head) → near-uniform start, then the
    policy sharpens to a *deterministic* optimum — closes the greedy-vs-stochastic gap.
  * **Entropy annealing** (0.01 → 0.002 over ~15k steps) → the PPO analogue of ε-decay:
    explore while discovering, consolidate greedily after.
  * **Reward scaling** (0.1; shaped goal ≈ 50 → value targets ~O(5)) → a stable critic.
  * **Small frequent rollouts** (32 × 8 envs = 256) + **lr 6e-4** → many updates per step,
    the driver of fast convergence on this short-horizon dense-shaped task.
  * γ=0.95 (short ≤100-step episodes), 5 epochs, clip 0.2, width_mult=1 — matching the DQN
    SimpleRoom capacity. (No lr anneal — entropy anneal handles consolidation here.)

  .venv/bin/python -u scripts/train_ppo_simple.py --device mps
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
    p.add_argument("--steps", type=int, default=40_000, help="env steps (converges ~30k)")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--ent-coef-final", type=float, default=0.002)
    p.add_argument("--ent-anneal-steps", type=int, default=15_000)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--rollout-steps", type=int, default=32)
    p.add_argument("--update-epochs", type=int, default=5)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--reward-scale", type=float, default=0.1)
    p.add_argument("--anneal-lr", action="store_true", help="also linearly decay lr → 0 (off by default)")
    p.add_argument("--width-mult", type=int, default=1)
    p.add_argument("--n-extra-conv", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    a = p.parse_args()

    device = get_torch_device(a.device)
    print(f"device: {describe_device(device)}", flush=True)
    env_fn = make_env_fn("simple", max_steps=a.max_steps)
    probe = env_fn()
    obs_shape, n_actions = probe.observation_space.shape, probe.action_space.n
    probe.close()
    print(f"PPO on simple  obs={obs_shape}  actions={n_actions}", flush=True)

    agent = PPO(
        obs_shape, n_actions, device=device, seed=a.seed,
        gamma=a.gamma, gae_lambda=a.gae_lambda, lr=a.lr, clip_coef=a.clip_coef,
        ent_coef=a.ent_coef, ent_coef_final=a.ent_coef_final, ent_anneal_steps=a.ent_anneal_steps,
        vf_coef=a.vf_coef, rollout_steps=a.rollout_steps, update_epochs=a.update_epochs,
        num_minibatches=a.num_minibatches, n_envs=a.n_envs, reward_scale=a.reward_scale,
        anneal_lr=a.anneal_lr, width_mult=a.width_mult, n_extra_conv=a.n_extra_conv,
    )
    print(f"params={agent.n_parameters():,}", flush=True)

    out_dir = Path(a.out) if a.out else default_out_dir("ppo", "simple")
    run_training(
        agent, env_fn, out_dir,
        total_steps=a.steps, n_envs=a.n_envs, max_steps=a.max_steps, seed=a.seed,
        video_every=5_000,
        title="SimpleRoomEnv — PPO",
        config_dict={**vars(a), "env": "simple"},
    )


if __name__ == "__main__":
    main()
