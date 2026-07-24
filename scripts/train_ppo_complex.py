"""Train PPO on ComplexEnv — best-shot config (pixels only, from scratch).

ComplexEnv is the hard key→door→water→lava→goal chain. DQN reached ~52% greedy at 4M steps
(heavy shaping + PER + ε consolidation tail). PPO is the more promising tool for the
water→lava→goal ferry — it's on-policy (no bootstrap-off-replay of the deadly triad) and,
per the SimpleRoom result, sharpens to a *deterministic* greedy policy (which DQN struggled
to consolidate). Config carries over everything that worked:

  * **Same tuned event shaping** as the DQN runs (imported via `make_env_fn("complex")`):
    monotonic pull-forward staircase, goal-dominant, anti-farm latches.
  * **reward_scale=0.02** — full-success returns are ~O(200); scale to ~O(4) so the shared
    critic stays well-conditioned (same scale the DQN runs used). Episode returns logged RAW.
  * **width_mult=2, n_extra_conv=1** — the capacity that cracked door→right for DQN.
  * **γ=0.99** (long 200-step horizon), GAE λ=0.95, **truncation bootstrap** (timeouts use
    V(final_obs), not a flat penalty — matters a lot with max_steps=200).
  * **Sustained-then-annealed entropy** (0.01 → 0.005 over ~3M) — keep exploring the rare
    ferry through discovery, then consolidate; plus **linear LR anneal → 0** so the greedy
    policy commits (the fix that closed the SimpleRoom greedy gap).
  * Orthogonal init, clipped surrogate, 4 epochs × 4 minibatches.

  .venv/bin/python -u scripts/train_ppo_complex.py --device mps
  .venv/bin/python -u scripts/train_ppo_complex.py --device mps --steps 8000000   # longer
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
    p.add_argument("--steps", type=int, default=5_000_000, help="env steps (DQN needed ~4M)")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--ent-coef-final", type=float, default=0.005)
    p.add_argument("--ent-anneal-steps", type=int, default=3_000_000)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--reward-scale", type=float, default=0.02)
    p.add_argument("--no-anneal-lr", action="store_true", help="disable linear lr decay to 0")
    p.add_argument("--width-mult", type=int, default=2)
    p.add_argument("--n-extra-conv", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    a = p.parse_args()

    device = get_torch_device(a.device)
    print(f"device: {describe_device(device)}", flush=True)
    env_fn = make_env_fn("complex", max_steps=a.max_steps)
    probe = env_fn()
    obs_shape, n_actions = probe.observation_space.shape, probe.action_space.n
    probe.close()
    print(f"PPO on complex  obs={obs_shape}  actions={n_actions}", flush=True)

    agent = PPO(
        obs_shape, n_actions, device=device, seed=a.seed,
        gamma=a.gamma, gae_lambda=a.gae_lambda, lr=a.lr, clip_coef=a.clip_coef,
        ent_coef=a.ent_coef, ent_coef_final=a.ent_coef_final, ent_anneal_steps=a.ent_anneal_steps,
        vf_coef=a.vf_coef, rollout_steps=a.rollout_steps, update_epochs=a.update_epochs,
        num_minibatches=a.num_minibatches, n_envs=a.n_envs, reward_scale=a.reward_scale,
        anneal_lr=not a.no_anneal_lr, width_mult=a.width_mult, n_extra_conv=a.n_extra_conv,
    )
    print(f"params={agent.n_parameters():,}", flush=True)

    out_dir = Path(a.out) if a.out else default_out_dir("ppo", "complex")
    run_training(
        agent, env_fn, out_dir,
        total_steps=a.steps, n_envs=a.n_envs, max_steps=a.max_steps, seed=a.seed,
        title="ComplexEnv — PPO",
        config_dict={**vars(a), "env": "complex"},
    )


if __name__ == "__main__":
    main()
