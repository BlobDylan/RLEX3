"""Train PPO on SimpleRoomEnv (pixels only, from scratch).

  .venv/bin/python -u scripts/train_ppo_simple.py --device mps
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._algo_cli import run_ppo

if __name__ == "__main__":
    run_ppo("simple")
