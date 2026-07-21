"""Train REINFORCE on ComplexEnv (pixels only, from scratch).

  .venv/bin/python -u scripts/train_reinforce_complex.py --device mps --steps 2000000
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._algo_cli import run_reinforce

if __name__ == "__main__":
    run_reinforce("complex")
