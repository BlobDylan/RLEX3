"""Plumbing smoke test: tiny runs of every algorithm on both envs.

Verifies the whole pipeline end-to-end (env factory -> agent -> vectorized train
loop -> history -> checkpoint save -> greedy eval -> plots -> result.json) without
any GPU-hours. Intentionally tiny: a few thousand steps, 2 envs, no videos.

  .venv/bin/python -u scripts/smoke_test.py                 # all algos, both envs
  .venv/bin/python -u scripts/smoke_test.py --only ppo      # one algorithm
  .venv/bin/python -u scripts/smoke_test.py --env simple    # one env
"""

from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms import DQN, PPO, REINFORCE
from utils import describe_device, get_torch_device
from scripts._common import make_env_fn, run_training

STEPS = 1_500
N_ENVS = 2
MAX_STEPS = 60
EVAL_EPISODES = 2
NO_VIDEO = 10**12  # video_every so large it never triggers


def build_agent(algo: str, obs_shape, n_actions, device, seed):
    """Tiny (width 1, no extra conv) agent for the given algorithm."""
    if algo == "dqn":
        return DQN(
            obs_shape, n_actions, device=device, seed=seed,
            buffer_size=2_000, learning_starts=200, train_freq=4,
            width_mult=1, n_extra_conv=0, reward_scale=0.02, n_step=3,
        )
    if algo == "reinforce":
        return REINFORCE(
            obs_shape, n_actions, device=device, seed=seed,
            episodes_per_update=4, width_mult=1, n_extra_conv=0,
        )
    if algo == "ppo":
        return PPO(
            obs_shape, n_actions, device=device, seed=seed,
            rollout_steps=64, num_minibatches=2, update_epochs=2,
            n_envs=N_ENVS, width_mult=1, n_extra_conv=0,
        )
    raise ValueError(algo)


def run_case(algo: str, env_name: str, device, out_root: Path) -> bool:
    print(f"\n{'=' * 60}\n[{algo} | {env_name}] starting\n{'=' * 60}", flush=True)
    env_fn = make_env_fn(env_name, max_steps=MAX_STEPS)
    probe = env_fn()
    obs_shape, n_actions = probe.observation_space.shape, probe.action_space.n
    probe.close()
    agent = build_agent(algo, obs_shape, n_actions, device, seed=0)
    out_dir = out_root / f"{env_name}_{algo}"

    run_training(
        agent, env_fn, out_dir,
        total_steps=STEPS, n_envs=N_ENVS, max_steps=MAX_STEPS, seed=0,
        log_every_episodes=5, eval_episodes=EVAL_EPISODES, video_every=NO_VIDEO,
        title=f"smoke {algo} {env_name}",
    )
    # Confirm the expected artifacts landed.
    missing = [f for f in ("history.json", "agent.pt", "result.json") if not (out_dir / f).exists()]
    if missing:
        raise RuntimeError(f"missing artifacts: {missing}")
    print(f"[{algo} | {env_name}] OK", flush=True)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", choices=["dqn", "reinforce", "ppo"], help="run a single algorithm")
    p.add_argument("--env", choices=["simple", "complex"], help="run a single env")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--keep", action="store_true", help="keep the smoke output dir")
    a = p.parse_args()

    device = get_torch_device(a.device)
    print(f"device: {describe_device(device)}", flush=True)
    algos = [a.only] if a.only else ["dqn", "reinforce", "ppo"]
    envs = [a.env] if a.env else ["simple", "complex"]
    out_root = ROOT / "output" / "_smoke"
    if out_root.exists():
        shutil.rmtree(out_root)

    results: dict[str, bool] = {}
    for algo in algos:
        for env_name in envs:
            key = f"{algo}/{env_name}"
            try:
                results[key] = run_case(algo, env_name, device, out_root)
            except Exception:
                results[key] = False
                print(f"[{key}] FAILED", flush=True)
                traceback.print_exc()

    if not a.keep and out_root.exists():
        shutil.rmtree(out_root)

    print(f"\n{'=' * 60}\nSMOKE SUMMARY\n{'=' * 60}", flush=True)
    for key, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {key}", flush=True)
    n_pass = sum(results.values())
    print(f"\n{n_pass}/{len(results)} passed", flush=True)
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
