"""2×2 A/B: tile_size × network width on the overnight winner recipe.

  |                | old arch (width_mult=1) | new arch (width_mult=2) |
  | tile_size=12   | tile12_arch1 (baseline) | tile12_arch2            |
  | tile_size=16   | tile16_arch1            | tile16_arch2            |

Shared recipe (from E_deep_frequent): frequent_updates + door_heavy, 250k steps.

Usage:
  .venv/bin/python -u scripts/ab_tile16.py
  .venv/bin/python -u scripts/ab_tile16.py --steps 150000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.device import describe_device, get_torch_device
from scripts.overnight_complex import Trial, run_trial, write_leaderboard

OUT = ROOT / "graphs" / "ab_tile_arch"


def build_trials(steps: int) -> list[Trial]:
    grid = [
        ("tile12_arch1", 12, 1),  # already seen overnight; re-run same seed for fair A/B
        ("tile12_arch2", 12, 2),
        ("tile16_arch1", 16, 1),
        ("tile16_arch2", 16, 2),
    ]
    trials: list[Trial] = []
    for i, (name, tile, width) in enumerate(grid):
        trials.append(
            Trial(
                name=name,
                dqn="frequent_updates",
                shaping="door_heavy",
                tile_size=tile,
                width_mult=width,
                steps=steps,
                seed=42,
                tags=["ab", "tile_arch", f"tile{tile}", f"w{width}"],
            )
        )
    return trials


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=250_000)
    ap.add_argument("--eval-episodes", type=int, default=15)
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()

    device = get_torch_device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device: {describe_device(device)}", flush=True)
    print(
        f"2×2 tile×arch A/B  steps={args.steps}  recipe=frequent_updates+door_heavy",
        flush=True,
    )

    trials = build_trials(args.steps)
    for i, trial in enumerate(trials, 1):
        print(
            f"\n[{i}/{len(trials)}] {trial.name}  "
            f"tile={trial.tile_size} width_mult={trial.width_mult}",
            flush=True,
        )
        run_trial(
            trial,
            device=device,
            default_steps=args.steps,
            eval_episodes=args.eval_episodes,
            out_dir=out / trial.name,
        )
        write_leaderboard(out)

    print(f"\nDone → {out / 'LEADERBOARD.md'}", flush=True)


if __name__ == "__main__":
    main()
