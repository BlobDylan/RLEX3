"""Train REINFORCE-with-baseline on ComplexEnv (pixels only, from scratch).

Honest expectation: REINFORCE is **not expected to solve** the long key→door→water→lava→goal
chain — it can't even make the greedy argmax navigate SimpleRoom (§5.4), and its high-variance
MC advantages only get worse as the horizon lengthens and rewards get sparser/later. The value
of this run is the **stage-progress analysis** (how far up the cascade does it get?) and the
DQN→REINFORCE→PPO family comparison — partial/negative results are explicitly valued.

Given a *fair* shot with the same levers as PPO-complex (so the comparison is apples-to-apples,
not sandbagged):

  * **Same tuned event shaping** (via `make_env_fn("complex")`), `max_steps=200`.
  * **reward_scale=0.02** — large shaped returns (~O(200)) → baseline MSE ~O(4).
  * **width_mult=2, n_extra_conv=1** — the ComplexEnv capacity DQN needed.
  * **γ=0.99** (long horizon), separate value baseline, truncation-tail bootstrap.
  * **Sustained entropy** (0.05 → 0.01 over ~1.5M) — keep exploring the chain.

Moderate 2M budget (don't burn compute on an expected early plateau); the runner is
checkpoint-safe, so **Ctrl+C once the stage cascade flatlines** and it still saves everything.

  .venv/bin/python -u scripts/train_reinforce_complex.py --device mps
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
    p.add_argument("--steps", type=int, default=2_000_000, help="env steps (expect an early plateau)")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--value-lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--entropy-coef", type=float, default=0.05)
    p.add_argument("--ent-coef-final", type=float, default=0.01)
    p.add_argument("--ent-anneal-steps", type=int, default=1_500_000)
    p.add_argument("--episodes-per-update", type=int, default=8)
    p.add_argument("--reward-scale", type=float, default=0.02)
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
    print(f"REINFORCE on complex  obs={obs_shape}  actions={n_actions}", flush=True)

    agent = REINFORCE(
        obs_shape, n_actions, device=device, seed=a.seed,
        gamma=a.gamma, lr=a.lr, value_lr=a.value_lr,
        entropy_coef=a.entropy_coef, ent_coef_final=a.ent_coef_final,
        ent_anneal_steps=a.ent_anneal_steps, episodes_per_update=a.episodes_per_update,
        reward_scale=a.reward_scale, width_mult=a.width_mult, n_extra_conv=a.n_extra_conv,
    )
    print(f"params={agent.n_parameters():,}", flush=True)

    out_dir = Path(a.out) if a.out else default_out_dir("reinforce", "complex")
    run_training(
        agent, env_fn, out_dir,
        total_steps=a.steps, n_envs=a.n_envs, max_steps=a.max_steps, seed=a.seed,
        title="ComplexEnv — REINFORCE",
        config_dict={**vars(a), "env": "complex"},
    )


if __name__ == "__main__":
    main()
