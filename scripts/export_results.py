"""Export a run's report assets from output/ into the tracked results/ tree.

Raw runs live under output/ (gitignored: weights, history.json, plots, videos). This
copies ONLY the report-worthy assets — graph images + rollout videos — into

    results/<run_name>/graphs/   (the *.png training curves)
    results/<run_name>/videos/   (the rollout .mp4s)
    results/<run_name>/result.json   (small eval/config summary; no weights, no history)

so they can be version-controlled and dropped straight into the report.

Usage:
  .venv/bin/python -u scripts/export_results.py output/complex_dqn/20260721_103304
  .venv/bin/python -u scripts/export_results.py output/simple_ppo/2026... --name ppo_simple_best
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GRAPH_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
VIDEO_EXTS = {".mp4", ".gif", ".webm", ".mov"}


def derive_run_name(run_path: Path) -> str:
    """output/<algo>/<timestamp> -> '<algo>_<timestamp>'; else just the folder name."""
    parent = run_path.parent.name
    if parent and parent not in ("output", "", "."):
        return f"{parent}_{run_path.name}"
    return run_path.name


def export(run_path: Path, results_dir: Path, name: str | None, overwrite: bool) -> Path:
    if not run_path.is_dir():
        raise SystemExit(f"error: not a directory: {run_path}")

    run_name = name or derive_run_name(run_path)
    dest = results_dir / run_name
    graphs_dest = dest / "graphs"
    videos_dest = dest / "videos"

    if dest.exists():
        if not overwrite:
            raise SystemExit(f"error: {dest} already exists (use --overwrite to replace)")
        shutil.rmtree(dest)
    graphs_dest.mkdir(parents=True, exist_ok=True)
    videos_dest.mkdir(parents=True, exist_ok=True)

    # Graph images: top-level image files in the run dir (skip weights/history).
    n_graphs = 0
    for f in sorted(run_path.iterdir()):
        if f.is_file() and f.suffix.lower() in GRAPH_EXTS:
            shutil.copy2(f, graphs_dest / f.name)
            n_graphs += 1

    # Rollout videos: anything under the run's videos/ (or video files at the root).
    n_videos = 0
    video_sources = list((run_path / "videos").glob("*")) if (run_path / "videos").is_dir() else []
    video_sources += [f for f in run_path.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTS]
    for f in video_sources:
        if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
            shutil.copy2(f, videos_dest / f.name)
            n_videos += 1

    # Small summary (config + eval), explicitly NOT weights or history.
    summary = run_path / "result.json"
    if summary.is_file():
        shutil.copy2(summary, dest / "result.json")

    if n_graphs == 0 and n_videos == 0:
        print(f"warning: no graph images or videos found under {run_path}", flush=True)

    print(f"exported '{run_name}':  {n_graphs} graph(s) + {n_videos} video(s) -> {dest}", flush=True)
    return dest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_path", type=str, help="path to an output/ run folder")
    p.add_argument("--name", type=str, default=None, help="override results/<name> (default: <algo>_<timestamp>)")
    p.add_argument("--results-dir", type=str, default=str(ROOT / "results"))
    p.add_argument("--overwrite", action="store_true", help="replace results/<name> if it exists")
    a = p.parse_args()
    export(Path(a.run_path).resolve(), Path(a.results_dir).resolve(), a.name, a.overwrite)


if __name__ == "__main__":
    main()
