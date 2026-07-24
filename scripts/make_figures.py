"""Report figures from run outputs.

Reads ``output/<env>_<algo>/<timestamp>/`` run dirs (which have history.json + result.json
+ agent.pt) and writes report-ready PNGs into the tracked ``results/`` tree:
  * per-run  → results/<name>/graphs/stage_cascade.png
  * cross-run → results/_figures/{sample_efficiency_*, stage_cascade_bars_complex,
                inference_success, greedy_vs_stochastic, ...}.png

Data-only figures (stage cascade, sample-efficiency, stage/inference bars) need no retrain.
``--with-agent`` additionally reloads each agent.pt to render greedy-vs-stochastic bars and
per-run eval histograms (a re-eval, not a retrain).

  .venv/bin/python -u scripts/make_figures.py                       # auto-discover output/
  .venv/bin/python -u scripts/make_figures.py output/complex_ppo/2026...   # specific runs
  .venv/bin/python -u scripts/make_figures.py --with-agent --device mps
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STAGES = ["stage_key", "stage_door", "stage_right", "stage_water", "stage_lava", "stage_goal"]
STAGE_LABELS = ["key", "door", "right", "water", "lava", "goal"]
ALGO_ORDER = ["dqn", "ppo", "reinforce"]
ALGO_COLOR = {"dqn": "#1f77b4", "ppo": "#2ca02c", "reinforce": "#d62728"}
ALGO_LABEL = {"dqn": "DQN", "ppo": "PPO", "reinforce": "REINFORCE"}
FIGS = ROOT / "results" / "_figures"


def derive_name(run_dir: Path) -> str:
    return f"{run_dir.parent.name}_{run_dir.name}"


def parse_env_algo(run_dir: Path) -> tuple[str, str]:
    env, _, algo = run_dir.parent.name.partition("_")  # "<env>_<algo>"
    return env, algo


def _algo_key(run_dir: Path) -> int:
    algo = parse_env_algo(run_dir)[1]
    return ALGO_ORDER.index(algo) if algo in ALGO_ORDER else 99


def rolling(y, w: int = 200):
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return y
    w = min(w, len(y))
    c = np.cumsum(np.insert(y, 0, 0.0))
    r = (c[w:] - c[:-w]) / w
    pad = np.full(w - 1, r[0] if len(r) else np.nan)
    return np.concatenate([pad, r]) if len(r) else y


def load(run_dir: Path):
    h = json.load(open(run_dir / "history.json"))
    rp = run_dir / "result.json"
    r = json.load(open(rp)) if rp.is_file() else {}
    env, algo = parse_env_algo(run_dir)
    return h, r, env, algo


def _safe_load(run_dir: Path):
    """Load a run, returning None if its JSON is unreadable (e.g. a live run mid-write)."""
    try:
        return load(run_dir)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  skip {derive_name(run_dir)}: {exc}", flush=True)
        return None


def discover() -> list[Path]:
    """The latest timestamped run per ``output/<env>_<algo>/`` group (the canonical
    final runs). Test/scratch dirs (non ``YYYYMMDD_HHMMSS`` names) are ignored.
    """
    ts = re.compile(r"^\d{8}_\d{6}$")
    groups: dict[str, list[Path]] = {}
    for p in glob.glob(str(ROOT / "output" / "*" / "*")):
        rp = Path(p)
        if (rp / "history.json").is_file() and ts.match(rp.name):
            groups.setdefault(rp.parent.name, []).append(rp)
    return sorted(max(v, key=lambda d: d.name) for v in groups.values())


def _results_graphs(run_dir: Path) -> Path:
    d = ROOT / "results" / derive_name(run_dir) / "graphs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig, path: Path) -> Path:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}", flush=True)
    return path


# --- A1: per-run stage cascade over training (ComplexEnv) --------------------
def fig_stage_cascade(run_dir: Path):
    loaded = _safe_load(run_dir)
    if loaded is None:
        return None
    h, _, env, algo = loaded
    if not h.get("stage_goal") or not any(h["stage_goal"]):
        return None
    steps = np.asarray(h["steps"], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    for sk, lbl in zip(STAGES, STAGE_LABELS):
        ax.plot(steps, rolling(h[sk]) * 100, label=lbl, lw=1.6)
    ax.set(xlabel="cumulative env steps", ylabel="reach rate (%)", ylim=(-2, 102),
           title=f"{env.capitalize()}Env {ALGO_LABEL.get(algo, algo)} — stage progress (rolling 200 ep)")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, _results_graphs(run_dir) / "stage_cascade.png")


# --- comprehensive per-run overview (training curves + stage cascade + eval hists) ---
def fig_overview(run_dir: Path, greedy_eval: dict | None = None):
    loaded = _safe_load(run_dir)
    if loaded is None:
        return None
    h, _, env, algo = loaded
    ret = np.asarray(h.get("episode_return", []), dtype=float)
    if ret.size == 0:
        return None
    length = np.asarray(h.get("episode_length", []), dtype=float)
    succ = np.asarray(h.get("episode_success", []), dtype=float)
    steps = np.asarray(h.get("steps", []), dtype=float)
    loss = np.asarray(h.get("loss", []), dtype=float)
    ep = np.arange(1, ret.size + 1)
    w = max(20, ret.size // 500)
    has_stage = bool(h.get("stage_goal")) and any(h["stage_goal"])

    panels: list[tuple[str, str, object]] = []  # (title, xlabel, draw)

    def add(title, xlabel, draw):
        panels.append((title, xlabel, draw))

    def d_return(ax):
        ax.plot(ep, ret, "0.8", lw=0.5)
        ax.plot(ep, rolling(ret, w), "C0", lw=1.8)

    def d_length(ax):
        ax.plot(ep, length, "0.8", lw=0.5)
        ax.plot(ep, rolling(length, w), "C1", lw=1.8)

    def d_success(ax):
        ax.plot(ep, rolling(succ, w) * 100, "C2", lw=1.8)
        ax.set_ylim(-2, 102)

    def d_steps(ax):
        ax.plot(ep, steps, "C3", lw=1.8)

    def d_loss(ax):
        li = np.arange(1, loss.size + 1)
        ax.plot(li, rolling(loss, max(5, loss.size // 300)), "C4", lw=1.2)

    def d_stage(ax):
        for sk, lbl in zip(STAGES, STAGE_LABELS):
            ax.plot(steps, rolling(h[sk], w) * 100, lw=1.2, label=lbl)
        ax.set_ylim(-2, 102)
        ax.legend(ncol=3, fontsize=6)

    add("Return", "episode", d_return)
    add("Episode length", "episode", d_length)
    add("Success rate (%)", "episode", d_success)
    add("Cumulative env steps", "episode", d_steps)
    if loss.size:
        add("Loss", "update", d_loss)
    if has_stage:
        add("Stage progress (%)", "cumulative env steps", d_stage)
    if greedy_eval and greedy_eval.get("episode_returns"):
        add("Greedy eval returns", "return",
            lambda ax: ax.hist(greedy_eval["episode_returns"], bins=15, color="C0"))
        add("Greedy eval lengths", "length",
            lambda ax: ax.hist(greedy_eval["episode_lengths"], bins=15, color="C1"))

    n = len(panels)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.1 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, xlabel, draw) in zip(axes, panels):
        draw(ax)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.grid(alpha=0.3)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"{env.capitalize()}Env — {ALGO_LABEL.get(algo, algo)} (overview)", fontsize=13)
    out = _results_graphs(run_dir) / f"{run_dir.parent.name}_overview.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}", flush=True)
    return out


# --- A2: cross-run sample efficiency (success vs cumulative steps) -----------
def fig_sample_efficiency(runs: list[Path], env: str):
    sub = sorted([r for r in runs if parse_env_algo(r)[0] == env], key=_algo_key)
    if not sub:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    for run in sub:
        loaded = _safe_load(run)
        if loaded is None:
            continue
        h, _, _, algo = loaded
        ax.plot(np.asarray(h["steps"], dtype=float), rolling(h["episode_success"]) * 100,
                label=ALGO_LABEL.get(algo, algo), color=ALGO_COLOR.get(algo), lw=1.8)
    ax.set(xlabel="cumulative env steps", ylabel="success rate (%, rolling 200 ep)", ylim=(-2, 102),
           title=f"{env.capitalize()}Env — sample efficiency")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, FIGS / f"sample_efficiency_{env}.png")


# --- A3: cross-run final stage cascade bars (ComplexEnv greedy eval) ---------
def fig_stage_bars(runs: list[Path]):
    data = {}
    for run in runs:
        if parse_env_algo(run)[0] != "complex":
            continue
        loaded = _safe_load(run)
        if loaded is None:
            continue
        _, res, _, algo = loaded
        ev = res.get("eval", {})
        if all(s in ev for s in STAGES):
            data[algo] = [ev[s] * 100 for s in STAGES]
    if not data:
        return None
    algos = [a for a in ALGO_ORDER if a in data] + [a for a in data if a not in ALGO_ORDER]
    x = np.arange(len(STAGE_LABELS))
    w = 0.8 / len(algos)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, a in enumerate(algos):
        ax.bar(x + i * w, data[a], w, label=ALGO_LABEL.get(a, a), color=ALGO_COLOR.get(a))
    ax.set_xticks(x + w * (len(algos) - 1) / 2)
    ax.set_xticklabels(STAGE_LABELS)
    ax.set(ylabel="greedy eval reach rate (%)", ylim=(0, 105),
           title="ComplexEnv — stage cascade (greedy eval)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, FIGS / "stage_cascade_bars_complex.png")


# --- A4: cross-run inference success bars (per env × algo) -------------------
def fig_inference_bars(runs: list[Path]):
    rows = {}
    for run in runs:
        loaded = _safe_load(run)
        if loaded is None:
            continue
        _, res, env, algo = loaded
        ev = res.get("eval", {})
        if "success_rate" in ev:
            rows[(env, algo)] = ev["success_rate"] * 100
    if not rows:
        return None
    envs = sorted({e for e, _ in rows})
    algos = [a for a in ALGO_ORDER if any((e, a) in rows for e in envs)]
    x = np.arange(len(envs))
    w = 0.8 / max(1, len(algos))
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, a in enumerate(algos):
        ax.bar(x + i * w, [rows.get((e, a), 0) for e in envs], w,
               label=ALGO_LABEL.get(a, a), color=ALGO_COLOR.get(a))
    ax.set_xticks(x + w * (len(algos) - 1) / 2)
    ax.set_xticklabels([e.capitalize() + "Env" for e in envs])
    ax.set(ylabel="greedy success rate (%)", ylim=(0, 105), title="Inference — greedy success")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, FIGS / "inference_success.png")


# --- A5/A6: agent re-eval (greedy-vs-stochastic bars + eval histograms) ------
def _load_agent(run_dir: Path, device):
    from algorithms import DQN, PPO, REINFORCE
    from scripts._common import make_env_fn

    _, res, env, algo = load(run_dir)
    cfg = res.get("config", {})
    max_steps = int(cfg.get("max_steps", 100 if env == "simple" else 200))
    env_fn = make_env_fn(env, max_steps=max_steps)
    probe = env_fn()
    obs_shape, n_actions = probe.observation_space.shape, probe.action_space.n
    probe.close()
    arch = dict(width_mult=int(cfg.get("width_mult", 1)), n_extra_conv=int(cfg.get("n_extra_conv", 0)))
    ctor = {"dqn": DQN, "ppo": PPO, "reinforce": REINFORCE}.get(algo)
    if ctor is None:
        return None
    agent = ctor(obs_shape, n_actions, device=device, **arch)
    agent.load(run_dir / "agent.pt")
    return agent, env_fn, env, algo, max_steps


def agent_evals(runs: list[Path], device, n_episodes: int = 50):
    from utils import get_torch_device

    dev = get_torch_device(device)
    gs = {}  # (env,algo) -> (greedy%, stoch%)
    for run in runs:
        loaded = _load_agent(run, dev)
        if loaded is None:
            continue
        agent, env_fn, env, algo, max_steps = loaded
        g = agent.evaluate(env_fn(), n_episodes=n_episodes, max_steps=max_steps, seed=9999, explore=False,
                           return_episodes=True)
        s = agent.evaluate(env_fn(), n_episodes=n_episodes, max_steps=max_steps, seed=9999, explore=True)
        gs[(env, algo)] = (g["success_rate"] * 100, s["success_rate"] * 100)
        # A6: per-run eval return/length histograms
        fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
        axs[0].hist(g["episode_returns"], bins=15, color=ALGO_COLOR.get(algo, "#555"))
        axs[0].set(xlabel="episode return", ylabel="count", title="greedy eval returns")
        axs[1].hist(g["episode_lengths"], bins=15, color=ALGO_COLOR.get(algo, "#555"))
        axs[1].set(xlabel="episode length", title="greedy eval lengths")
        fig.suptitle(f"{env.capitalize()}Env {ALGO_LABEL.get(algo, algo)} — greedy eval ({n_episodes} eps)")
        _save(fig, _results_graphs(run) / "eval_hist.png")
        # regenerate the comprehensive overview WITH the eval histograms included
        fig_overview(run, greedy_eval=g)

    # A5: greedy-vs-stochastic grouped bars across all runs
    if gs:
        keys = sorted(gs, key=lambda k: (k[0], ALGO_ORDER.index(k[1]) if k[1] in ALGO_ORDER else 9))
        labels = [f"{ALGO_LABEL.get(a, a)}\n{e}" for e, a in keys]
        x = np.arange(len(keys))
        fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(keys)), 4))
        ax.bar(x - 0.2, [gs[k][0] for k in keys], 0.4, label="greedy (argmax)", color="#1f77b4")
        ax.bar(x + 0.2, [gs[k][1] for k in keys], 0.4, label="stochastic (sampled)", color="#ff7f0e")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set(ylabel="success rate (%)", ylim=(0, 105), title="Greedy vs stochastic policy")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        _save(fig, FIGS / "greedy_vs_stochastic.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", help="output run dirs (default: auto-discover output/*/*)")
    p.add_argument("--with-agent", action="store_true", help="also re-eval agents for greedy-vs-stochastic + histograms")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--eval-episodes", type=int, default=50)
    a = p.parse_args()

    runs = [Path(r).resolve() for r in a.runs] if a.runs else discover()
    runs = [r for r in runs if (r / "history.json").is_file()]
    if not runs:
        raise SystemExit("no run dirs with history.json found under output/")
    print(f"figures for {len(runs)} run(s):", ", ".join(derive_name(r) for r in runs), flush=True)

    print("per-run stage cascades + overviews:", flush=True)
    for run in runs:
        fig_stage_cascade(run)
        fig_overview(run)
    print("cross-run comparison figures:", flush=True)
    for env in sorted({parse_env_algo(r)[0] for r in runs}):
        fig_sample_efficiency(runs, env)
    fig_stage_bars(runs)
    fig_inference_bars(runs)

    if a.with_agent:
        print("agent re-eval (greedy-vs-stochastic + histograms):", flush=True)
        agent_evals(runs, a.device, a.eval_episodes)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
