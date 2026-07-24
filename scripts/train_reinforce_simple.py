"""Train REINFORCE-with-baseline on SimpleRoomEnv (pixels only, from scratch).

REINFORCE learns a competent **stochastic** policy here (~80-85% sampled success) but its
**greedy/argmax policy stays degenerate** (~4%): the argmax is "forward" in every state, so
the deterministic policy walks into a wall and times out. This is REINFORCE's high-variance
limitation, not a bug — its Monte-Carlo advantages can't reliably credit the minority
"turn" action against the majority-action bias, and its stochastic training distribution
differs from the greedy one. PPO's low-variance GAE + clipped updates close this gap (see
REPORT_MEMOS §5.4). Evaluate REINFORCE by sampling; the greedy gap is a reported finding.

Implementation (all applied; they fixed the earlier single-action *collapse* and lifted the
stochastic policy, but did not close the greedy gap):

  * **Orthogonal init** (sqrt(2) trunk, 0.01 logit head) — near-uniform start, no collapse.
  * **Separate value-baseline network** (Sutton & Barto §13.4) — advantage = MC return - V(s),
    no bootstrap, so this stays REINFORCE (not actor-critic). Reward-scaled so the baseline
    MSE is well-conditioned.
  * **Truncation-tail bootstrap** — timed-out episodes bootstrap V(final_obs) instead of a
    flat penalty (SAME_STEP autoreset).
  * **Entropy annealing**; γ=0.95 (short ≤100-step episodes); width_mult=1 (DQN/PPO capacity).

  .venv/bin/python -u scripts/train_reinforce_simple.py --device mps
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms import REINFORCE
from utils import describe_device, get_torch_device
from scripts._common import default_out_dir, make_env_fn, run_training


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=120_000, help="env steps")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--entropy-coef", type=float, default=0.05)
    p.add_argument("--ent-coef-final", type=float, default=0.002)
    p.add_argument("--ent-anneal-steps", type=int, default=50_000)
    p.add_argument("--episodes-per-update", type=int, default=8)
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
    print(f"REINFORCE on simple  obs={obs_shape}  actions={n_actions}", flush=True)

    agent = REINFORCE(
        obs_shape, n_actions, device=device, seed=a.seed,
        gamma=a.gamma, lr=a.lr,
        entropy_coef=a.entropy_coef, ent_coef_final=a.ent_coef_final,
        ent_anneal_steps=a.ent_anneal_steps,
        episodes_per_update=a.episodes_per_update,
        width_mult=a.width_mult, n_extra_conv=a.n_extra_conv,
    )
    print(f"params={agent.n_parameters():,}", flush=True)

    out_dir = Path(a.out) if a.out else default_out_dir("reinforce", "simple")
    run_training(
        agent, env_fn, out_dir,
        total_steps=a.steps, n_envs=a.n_envs, max_steps=a.max_steps, seed=a.seed,
        video_every=5_000,
        title="SimpleRoomEnv — REINFORCE",
        config_dict={**vars(a), "env": "simple"},
    )


if __name__ == "__main__":
    main()
