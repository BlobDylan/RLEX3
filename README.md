# Deep RL in MiniGrid — Final Project (2026 B)

Train deep RL agents **from pixels** on two image-based MiniGrid environments and
compare them. Everything runs from **scripts** (the original assignment notebook has
been retired) so experiments are reproducible and version-control friendly.

## Environments

| Env             | Task                                    | Win condition                                                |
| --------------- | --------------------------------------- | ------------------------------------------------------------ |
| `SimpleRoomEnv` | empty 10×10 room (sanity)               | step onto the green goal                                     |
| `ComplexEnv`    | key → locked door → water → lava → goal | ferry water to extinguish the lava ring, then reach the goal |

Both expose `Discrete(7)` actions and a raw full-grid RGB observation (`uint8`). The
env classes are **fixed** — all preprocessing / reward shaping lives in wrappers.

## Layout

```
envs/        ComplexEnv, SimpleRoomEnv (fixed environment classes)
wrappers/    obs / action / reward-shaping wrappers (grayscale, crop, tile-inset,
             resize, frame-stack, ComplexShapingWrapper, SimpleRoomShapingWrapper)
algorithms/  BaseAlgorithm, DQN (Double + n-step + reward-scaling), hparam_search,
             rainbow (Double+Dueling+Noisy+PER+C51+n-step — optional escalation)
utils/       device selection, training-curve plotting, rollout videos
pipelines.py make_complex_env / make_simple_env — the single source of truth for
             the wrapper stacks used by every script
scripts/     experiment runners (see below)
colab/       Google-Drive / Colab repo-root bootstrap helpers
env_explorer.py   optional inline, button-driven env explorer (assignment tool)
```

## Setup

Dependencies are pinned in `pyproject.toml` / `poetry.lock`. A local `.venv` is
already provisioned; use it directly, or recreate with Poetry:

```bash
poetry install            # or: python -m venv .venv && .venv/bin/pip install -e .
```

All commands below assume the repo root as the working directory and use the venv
interpreter (`.venv/bin/python`).

## Running experiments

### ComplexEnv — from-scratch DQN (recommended)

A Double DQN trained **from scratch** (no demonstrations / imitation) on a single
**RGB** frame. Key ingredients that make it learn the long key→door→water→lava→goal
chain:

- **Event-based reward shaping** — small, anti-farm milestone bonuses that guide
  exploration (the legitimate signal, not an expert policy) with the goal dominant so
  the agent can't bank a comfortable return by camping before a bottleneck.
- **Reward scaling** (`reward_scale`) so Q ≈ O(1); the raw shaping rewards otherwise
  push Q ≈ 185 and wreck the value SNR, which stalled every earlier run.
- **RND intrinsic exploration** (`rnd_coef`) — a novelty bonus that drives the agent
  through the door into the unseen right room and onward (Burda et al. 2018); the
  pure-RL fix for the exploration deadlock. `--rnd-coef 0` ablates it.
- **n-step returns** for faster credit assignment, and sustained ε-greedy exploration.
- A single RGB frame is enough (inventory is recoverable from what's missing on
  screen); grayscale / frame-stacking both hurt.

```bash
.venv/bin/python -u scripts/complex_dqn.py                     # defaults (300k steps)
.venv/bin/python -u scripts/complex_dqn.py --steps 400000 --device mps
.venv/bin/python -u scripts/complex_dqn.py --help             # all knobs
```

Outputs land in `graphs/complex_dqn/<timestamp>/`:
`history.json`, training-curve PNGs, `agent.pt`, and `result.json` (config + eval).

Key flags: `--steps`, `--max-steps`, `--reward-scale`, `--rnd-coef`, `--n-step`,
`--gamma`, `--lr`, `--eps-end`, `--eps-decay-steps`, `--train-freq`, `--width-mult`,
`--n-extra-conv`, `--seed`, `--device`, `--out`.

### Other runners

```bash
.venv/bin/python -u scripts/overnight_complex.py   # curated DQN × shaping sweep + leaderboard
.venv/bin/python -u scripts/ab_tile16.py           # tile-size / architecture A/B
```

## Using the pieces from Python

```python
from pipelines import make_complex_env
from algorithms import DQN
from utils import get_torch_device

env_fn = lambda: make_complex_env(max_steps=300, grayscale=False, frame_stack=1)
agent = DQN(env_fn().observation_space.shape, 6, device=get_torch_device(),
            n_step=3, reward_scale=0.03)
history = agent.train(env_fn, total_timesteps=300_000)
```

## Env explorer (optional)

`env_explorer.py` drives any (wrapped) env by hand with inline buttons for debugging
shaping / preprocessing — import `explore_env` from it inside a Jupyter/VS Code cell.
