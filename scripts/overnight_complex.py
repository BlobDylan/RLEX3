"""ComplexEnv overnight DQN sweep (Step 6c).

Runs a curated grid of DQN variants + reward-shaping configs. Resumable: skips
trials that already have ``result.json``. Rank by stage progress (door → right → …).

Usage:
  .venv/bin/python -u scripts/overnight_complex.py
  .venv/bin/python -u scripts/overnight_complex.py --steps 80000 --device mps
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from algorithms.device import describe_device, get_torch_device
from algorithms.dqn import DQN
from algorithms.plotting import plot_training_history, save_training_history
from envs import ComplexEnv
from wrappers import (
    DEFAULT_COMPLEX_SHAPING,
    ActionSubsetWrapper,
    ComplexShapingWrapper,
    CropOuterWallsWrapper,
    GrayscaleWrapper,
    ResizeObsWrapper,
    TileInsetWrapper,
)

OUT_ROOT = ROOT / "graphs" / "overnight_complex"


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

COMPLEX_ACTIONS = (0, 1, 2, 3, 4, 5)


def make_complex_env(
    *,
    max_steps: int = 200,
    tile_size: int = 12,
    keep_fraction: float = 0.75,
    cnn_size: int = 64,
    grayscale: bool = True,
    shaping: dict[str, float] | None = None,
    use_enter_right_room: bool = True,
):
    kw = dict(DEFAULT_COMPLEX_SHAPING)
    if shaping:
        kw.update(shaping)
    env = ComplexEnv(max_steps=max_steps, tile_size=tile_size)
    env = ComplexShapingWrapper(
        env, use_enter_right_room=use_enter_right_room, **kw
    )
    env = ActionSubsetWrapper(env, action_ids=COMPLEX_ACTIONS)
    env = CropOuterWallsWrapper(env)
    env = TileInsetWrapper(env, keep_fraction=keep_fraction)
    if grayscale:
        env = GrayscaleWrapper(env)
    env = ResizeObsWrapper(env, size=cnn_size)
    return env


# ---------------------------------------------------------------------------
# Shaping presets (Exp11: door is the bottleneck)
# ---------------------------------------------------------------------------

SHAPING: dict[str, dict[str, Any]] = {
    "default6a": {},
    "door_heavy": {
        "door_open": 25.0,
        "key_pickup": 3.0,
        "enter_right_room": 8.0,  # first time only
        "leave_right_room": 8.0,  # every backtrack to left
        "key_drop": 10.0,  # first drop in right room (free hand for water)
        "key_drop_locked_left": 8.0,  # drop in left before door open
        "lava_extinguish": 20.0,  # per lava tile (push ferrying water)
    },
    "door_extreme": {
        "door_open": 40.0,
        "key_pickup": 2.0,
        "enter_right_room": 15.0,
        "step_penalty": 0.05,
    },
    "key_then_door": {
        "key_pickup": 15.0,
        "door_open": 30.0,
        "enter_right_room": 10.0,
    },
    "low_step_cost": {
        "step_penalty": 0.02,
        "door_open": 20.0,
    },
    "high_step_cost": {
        "step_penalty": 0.2,
        "door_open": 20.0,
    },
    "goal_dominant": {
        "goal_scale": 100.0,
        "door_open": 15.0,
        "key_pickup": 5.0,
    },
    "sparse_mid": {
        # Only key/door/goal matter; mute water/lava mid rewards
        "water_pickup": 0.0,
        "lava_extinguish": 0.0,
        "enter_right_room": 0.0,
        "door_open": 25.0,
        "key_pickup": 8.0,
    },
    "no_enter_bonus": {
        "use_enter_right_room": False,
        "door_open": 25.0,
    },
    "lava_aware": {
        "lava_death": 25.0,
        "lava_extinguish": 10.0,
        "door_open": 20.0,
        "water_pickup": 5.0,
    },
}


# ---------------------------------------------------------------------------
# DQN hyper presets
# ---------------------------------------------------------------------------

def _base_hp(**overrides: Any) -> dict[str, Any]:
    hp = {
        "lr": 2.81e-4,
        "gamma": 0.95,
        "batch_size": 64,
        "train_freq": 8,
        "gradient_steps": 1,
        "tau": 1.0,
        "target_update_freq": 500,
        "eps_start": 1.0,
        "eps_end": 0.15,
        "eps_decay_steps": 20_000,
        "learning_starts": 1_000,
        "double_dqn": True,
        "buffer_size": 100_000,
        "log_loss_every": 100,
    }
    hp.update(overrides)
    return hp


DQN_PRESETS: dict[str, dict[str, Any]] = {
    "exp10b_fast": _base_hp(),
    "slow_eps": _base_hp(eps_decay_steps=80_000, eps_end=0.1),
    "very_slow_eps": _base_hp(eps_decay_steps=120_000, eps_end=0.05),
    "soft_target": _base_hp(tau=0.005, target_update_freq=1),
    "soft_fast": _base_hp(tau=0.01, target_update_freq=1),
    "vanilla_dqn": _base_hp(double_dqn=False),
    "high_gamma": _base_hp(gamma=0.99, eps_decay_steps=60_000),
    "low_gamma": _base_hp(gamma=0.9),
    "big_batch": _base_hp(batch_size=128, train_freq=4),
    "small_batch": _base_hp(batch_size=32, train_freq=4),
    "frequent_updates": _base_hp(train_freq=2, batch_size=64, eps_decay_steps=60_000),
    "lr_high": _base_hp(lr=5e-4, eps_decay_steps=60_000),
    "lr_low": _base_hp(lr=1e-4, eps_decay_steps=60_000),
    "long_buffer": _base_hp(buffer_size=200_000, eps_decay_steps=80_000),
}


@dataclass
class Trial:
    name: str
    dqn: str
    shaping: str
    grayscale: bool = True
    max_steps: int = 200
    tile_size: int = 12
    width_mult: int = 1
    steps: int | None = None  # override global
    seed: int = 0
    tags: list[str] = field(default_factory=list)


def build_trial_list(*, short_steps: int, long_steps: int) -> list[Trial]:
    """Curated overnight grid. Door-focused first (Exp11)."""
    trials: list[Trial] = []

    # --- Phase A: shaping sweep @ short budget (screen door rate) ---
    for i, sk in enumerate(SHAPING):
        trials.append(
            Trial(
                name=f"A_shape_{sk}",
                dqn="slow_eps",
                shaping=sk,
                steps=short_steps,
                seed=1000 + i,
                tags=["phaseA", "shaping"],
            )
        )

    # --- Phase B: DQN variant sweep with door_heavy shaping ---
    for i, dk in enumerate(DQN_PRESETS):
        trials.append(
            Trial(
                name=f"B_dqn_{dk}",
                dqn=dk,
                shaping="door_heavy",
                steps=short_steps,
                seed=2000 + i,
                tags=["phaseB", "dqn"],
            )
        )

    # --- Phase C: RGB vs gray (door/key are color-coded) ---
    for i, gray in enumerate((True, False)):
        trials.append(
            Trial(
                name=f"C_{'gray' if gray else 'rgb'}_door",
                dqn="slow_eps",
                shaping="door_heavy",
                grayscale=gray,
                steps=short_steps,
                seed=3000 + i,
                tags=["phaseC", "obs"],
            )
        )

    # --- Phase D: horizon ---
    for i, ms in enumerate((150, 200, 300)):
        trials.append(
            Trial(
                name=f"D_horizon_{ms}",
                dqn="slow_eps",
                shaping="door_heavy",
                max_steps=ms,
                steps=short_steps,
                seed=4000 + i,
                tags=["phaseD", "horizon"],
            )
        )

    # --- Phase E: deep runs (best-bet combos from Exp11 diagnosis) ---
    deep = [
        ("E_deep_door_sloweps", "slow_eps", "door_heavy", True),
        ("E_deep_door_extreme", "very_slow_eps", "door_extreme", True),
        ("E_deep_keydoor", "slow_eps", "key_then_door", True),
        ("E_deep_soft_door", "soft_target", "door_heavy", True),
        ("E_deep_rgb_door", "slow_eps", "door_heavy", False),
        ("E_deep_frequent", "frequent_updates", "door_heavy", True),
        ("E_deep_high_gamma", "high_gamma", "door_extreme", True),
        ("E_deep_default_long", "exp10b_fast", "default6a", True),
        ("E_deep_rgb_extreme", "very_slow_eps", "door_extreme", False),
        ("E_deep_lowstep_slow", "slow_eps", "low_step_cost", True),
        ("E_deep_vanilla_door", "vanilla_dqn", "door_heavy", True),
        ("E_deep_lrhigh_door", "lr_high", "door_heavy", True),
    ]
    for i, (name, dqn, shp, gray) in enumerate(deep):
        trials.append(
            Trial(
                name=name,
                dqn=dqn,
                shaping=shp,
                grayscale=gray,
                steps=long_steps,
                seed=5000 + i,
                tags=["phaseE", "deep"],
            )
        )

    # --- Phase F: extra longshots (fill the night) ---
    longshots = [
        ("F_sparse_mid_slow", "very_slow_eps", "sparse_mid", True, 300),
        ("F_lava_aware", "slow_eps", "lava_aware", True, 200),
        ("F_goal_dom", "high_gamma", "goal_dominant", True, 200),
        ("F_rgb_soft", "soft_target", "door_extreme", False, 200),
        ("F_bigbatch_door", "big_batch", "door_heavy", True, 200),
        ("F_horizon300_extreme", "very_slow_eps", "door_extreme", True, 300),
    ]
    for i, (name, dqn, shp, gray, hz) in enumerate(longshots):
        trials.append(
            Trial(
                name=name,
                dqn=dqn,
                shaping=shp,
                grayscale=gray,
                max_steps=hz,
                steps=long_steps,
                seed=6000 + i,
                tags=["phaseF", "deep"],
            )
        )

    return trials


def _stage_rates(history: dict[str, list[float]], last_n: int = 50) -> dict[str, float]:
    out = {}
    for sk in (
        "stage_key",
        "stage_door",
        "stage_right",
        "stage_water",
        "stage_lava",
        "stage_goal",
        "episode_success",
    ):
        vals = history.get(sk) or []
        if not vals:
            out[sk] = 0.0
            continue
        chunk = vals[-last_n:]
        out[sk] = float(np.mean(chunk))
    return out


def _score(result: dict[str, Any]) -> float:
    """Lexicographic-ish scalar for ranking (door is king after Exp11)."""
    ev = result.get("eval", {})
    late = result.get("late_stages", {})
    return (
        1000.0 * float(ev.get("success_rate", 0.0))
        + 100.0 * float(ev.get("stage_goal", late.get("stage_goal", 0.0)))
        + 40.0 * float(ev.get("stage_lava", late.get("stage_lava", 0.0)))
        + 30.0 * float(ev.get("stage_water", late.get("stage_water", 0.0)))
        + 20.0 * float(ev.get("stage_right", late.get("stage_right", 0.0)))
        + 10.0 * float(ev.get("stage_door", late.get("stage_door", 0.0)))
        + 2.0 * float(ev.get("stage_key", late.get("stage_key", 0.0)))
        + 0.01 * float(ev.get("mean_return", 0.0))
    )


def run_trial(
    trial: Trial,
    *,
    device: torch.device,
    default_steps: int,
    eval_episodes: int,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"
    if result_path.exists():
        print(f"⏭ skip existing {trial.name}", flush=True)
        return json.loads(result_path.read_text())

    shaping_cfg = dict(SHAPING[trial.shaping])
    use_enter = bool(shaping_cfg.pop("use_enter_right_room", True))
    hp = dict(DQN_PRESETS[trial.dqn])
    hp["width_mult"] = int(trial.width_mult)
    steps = int(trial.steps or default_steps)

    def env_fn():
        return make_complex_env(
            max_steps=trial.max_steps,
            tile_size=trial.tile_size,
            grayscale=trial.grayscale,
            shaping=shaping_cfg,
            use_enter_right_room=use_enter,
        )

    probe = env_fn()
    obs_shape = probe.observation_space.shape
    n_actions = int(probe.action_space.n)
    probe.close()

    meta = {
        "name": trial.name,
        "tags": trial.tags,
        "dqn_preset": trial.dqn,
        "shaping_preset": trial.shaping,
        "shaping": shaping_cfg,
        "use_enter_right_room": use_enter,
        "hparams": hp,
        "grayscale": trial.grayscale,
        "max_steps": trial.max_steps,
        "tile_size": trial.tile_size,
        "width_mult": trial.width_mult,
        "total_timesteps": steps,
        "seed": trial.seed,
        "obs_shape": list(obs_shape),
        "n_actions": n_actions,
        "device": str(device),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2))

    print(
        f"\n{'='*60}\n▶ {trial.name}  steps={steps}  dqn={trial.dqn}  "
        f"shape={trial.shaping}  gray={trial.grayscale}  horizon={trial.max_steps}\n",
        flush=True,
    )
    t0 = time.perf_counter()
    agent = DQN(obs_shape, n_actions, device=device, seed=trial.seed, **hp)
    history = agent.train(env_fn, total_timesteps=steps, log_every=max(2000, steps // 20))
    train_s = time.perf_counter() - t0

    save_training_history(history, out_dir / "history.json")
    try:
        plot_training_history(
            history,
            title=trial.name,
            save_prefix=trial.name,
            folder=out_dir,
            show=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  (plot warn: {exc})", flush=True)

    eval_env = env_fn()
    metrics = agent.evaluate(eval_env, n_episodes=eval_episodes, seed=10_000 + trial.seed)
    diag = agent.diagnose_greedy(eval_env, seed=42)
    eval_env.close()

    ckpt = out_dir / "agent.pt"
    agent.save(ckpt)

    late = _stage_rates(history, last_n=50)
    result = {
        **meta,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": train_s,
        "steps_per_sec": steps / max(train_s, 1e-6),
        "eval": metrics,
        "late_stages": late,
        "diagnose": {
            "action_counts": diag["action_counts"],
            "success": diag["success"],
            "return": diag["return"],
            "steps": diag["steps"],
        },
        "checkpoint": str(ckpt),
    }
    result["score"] = _score(result)
    result_path.write_text(json.dumps(result, indent=2))
    print(
        f"✓ {trial.name}  {train_s/60:.1f}min  "
        f"door_late={late.get('stage_door', 0):.0%}  "
        f"key_late={late.get('stage_key', 0):.0%}  "
        f"eval_succ={metrics.get('success_rate', 0):.0%}  "
        f"eval_door={metrics.get('stage_door', 0):.0%}  "
        f"score={result['score']:.1f}",
        flush=True,
    )
    del agent
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def write_leaderboard(out_root: Path) -> Path:
    rows = []
    for p in sorted(out_root.glob("*/result.json")):
        try:
            rows.append(json.loads(p.read_text()))
        except Exception:  # noqa: BLE001
            continue
    rows.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
    lines = [
        "# Overnight ComplexEnv leaderboard",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Trials finished: {len(rows)}",
        "",
        "| rank | name | score | eval_succ | eval_door | late_door | late_key | steps | min |",
        "|-----:|------|------:|----------:|----------:|----------:|---------:|------:|----:|",
    ]
    for i, r in enumerate(rows, 1):
        late = r.get("late_stages", {})
        ev = r.get("eval", {})
        lines.append(
            f"| {i} | `{r.get('name')}` | {r.get('score', 0):.1f} | "
            f"{ev.get('success_rate', 0):.0%} | {ev.get('stage_door', 0):.0%} | "
            f"{late.get('stage_door', 0):.0%} | {late.get('stage_key', 0):.0%} | "
            f"{r.get('total_timesteps', 0)} | {r.get('wall_seconds', 0)/60:.1f} |"
        )
    path = out_root / "LEADERBOARD.md"
    path.write_text("\n".join(lines) + "\n")
    (out_root / "leaderboard.json").write_text(json.dumps(rows, indent=2))
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--short-steps", type=int, default=80_000)
    ap.add_argument("--long-steps", type=int, default=250_000)
    ap.add_argument("--eval-episodes", type=int, default=15)
    ap.add_argument("--out", type=str, default=str(OUT_ROOT))
    ap.add_argument("--only-tag", type=str, default=None, help="e.g. phaseA / phaseE")
    ap.add_argument("--limit", type=int, default=0, help="max trials (0=all)")
    args = ap.parse_args()

    device = get_torch_device(args.device)
    print(f"device: {describe_device(device)}", flush=True)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    trials = build_trial_list(short_steps=args.short_steps, long_steps=args.long_steps)
    if args.only_tag:
        trials = [t for t in trials if args.only_tag in t.tags]
    if args.limit > 0:
        trials = trials[: args.limit]

    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "short_steps": args.short_steps,
        "long_steps": args.long_steps,
        "n_trials": len(trials),
        "trials": [asdict(t) for t in trials],
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"scheduled {len(trials)} trials → {out_root}", flush=True)

    errors = []
    for i, trial in enumerate(trials, 1):
        print(f"\n[{i}/{len(trials)}] ", end="", flush=True)
        try:
            run_trial(
                trial,
                device=device,
                default_steps=args.short_steps,
                eval_episodes=args.eval_episodes,
                out_dir=out_root / trial.name,
            )
        except Exception as exc:  # noqa: BLE001
            err = f"{trial.name}: {exc}\n{traceback.format_exc()}"
            print(f"✗ FAILED {trial.name}: {exc}", flush=True)
            errors.append(err)
            (out_root / trial.name).mkdir(parents=True, exist_ok=True)
            (out_root / trial.name / "error.txt").write_text(err)
        write_leaderboard(out_root)

    if errors:
        (out_root / "errors.log").write_text("\n\n".join(errors))
    lb = write_leaderboard(out_root)
    print(f"\nDone. Leaderboard: {lb}", flush=True)


if __name__ == "__main__":
    main()
