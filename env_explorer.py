"""Inline, button-driven MiniGrid env explorer for the HW3 (2026B) notebook.

Drive *any* MiniGrid env by hand with buttons, rendered *inline*
in local Jupyter, VS Code, or Google Colab -- no Tk window needed. The panel shows the
running reward plus auto-discovered state, so it adapts to whatever env you pass with no
per-env wiring:

* a generic **grid-object scan** (keys, balls, doors, goals, lava, ... with their
  positions, colours and door open/locked state),
* a **status-flag scan** that introspects the env for zero-arg query helpers -- the
  current-state getters (``is_*`` and ``*_count``) and their start-of-step ``prev_*``
  snapshots -- and pairs each ``now`` value with its ``prev`` (marking what the last
  action changed), e.g. ``is_carrying_key`` / ``prev_carrying_key`` or
  ``extinguished_lava_count``, without naming any of them here, and
* a rolling **action history** so you can see the path that led to the current state.

Usage (the env class is the source of truth and is passed in, so there is no second
copy to keep in sync)::

    from env_explorer import explore_env
    explore_env(ComplexEnv, seed=None)            # a bare env class / zero-arg factory
    explore_env(lambda: GrayscaleWrapper(EventShapingWrapper(ComplexEnv())))  # wrapped env

It also drives a **wrapped** env: pass any zero-arg factory that returns your wrapper
stack. The full stack is stepped, so the reward shown is your *shaped / scaled* reward
and the observation is preprocessed, while state reads (grid scan + status flags) come
from ``env.unwrapped``. If the stack restricts the action set (a ``gym.ActionWrapper``),
each button is mapped to the right base action automatically and buttons for actions the
env doesn't expose are disabled.

On **Google Colab**, drag this file into the *Files* panel first so the import resolves;
``explore_env`` then enables Colab's custom widget manager automatically so the buttons
are interactive. Re-run ``explore_env(...)`` after editing the env / wrappers to pick up
the changes.
"""

from __future__ import annotations

import html
import inspect
import io
import subprocess
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

try:                                            # only used to introspect the wrapper stack
    import gymnasium as gym
    _WRAPPER, _ACTION_WRAPPER = gym.Wrapper, gym.ActionWrapper
except Exception:                               # gymnasium missing -> treat every env as bare
    _WRAPPER, _ACTION_WRAPPER = (), ()

try:
    import ipywidgets as widgets
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "ipywidgets"])
    import ipywidgets as widgets
from IPython.display import display

# Standard MiniGrid action ids -> button labels and a compact glyph for the history line.
_ACTIONS = [("⟲ left", 0), ("↑ forward", 2), ("⟳ right", 1),
            ("pickup", 3), ("toggle", 5), ("drop", 4)]
_GLYPHS = {0: "⟲", 1: "⟳", 2: "↑", 3: "P", 4: "D", 5: "T"}
_DIRS = {0: "right", 1: "down", 2: "left", 3: "up"}

# How many recent actions to keep on the history line.
_HISTORY_LEN = 16

# Name patterns for env query helpers worth surfacing as status flags: the current-state
# getters (``is_*`` and ``*_count``) plus their start-of-step ``prev_*`` snapshots, which
# the panel pairs up so you can see what the last action changed.
_FLAG_PREFIXES = ("is_", "prev_")
_FLAG_SUFFIXES = ("_count",)
_PREV_PREFIX = "prev_"
# Prefixes stripped to get the shared base name that pairs a getter with its prev twin
# (e.g. ``is_carrying_key`` / ``prev_carrying_key`` -> ``carrying_key``).
_FLAG_STRIP = ("is_", "prev_")
# Object types that carry no meaningful per-instance colour (don't clutter output).
_COLORLESS = {"goal", "lava", "floor", "wall", "unseen", "empty"}

# Status -> badge colour for the info panel header.
_STATUS_COLORS = {"running": "#2d7dd2", "TERMINATED": "#2e9e5b", "TRUNCATED": "#d98e04"}
# Target on-screen size (px) for the rendered frame. Small grids are upscaled
# nearest-neighbour to about this size so the pixels stay crisp instead of being
# blurred by the browser when it stretches a tiny image.
_TARGET_PX = 512


def _describe_obj(i, j, cell):
    """One-line description of a grid object: position, colour, and door state."""
    s = f"({i},{j})"
    color = getattr(cell, "color", None)
    if color and cell.type not in _COLORLESS:
        s += f" {color}"
    if cell.type == "door":
        s += " [locked]" if cell.is_locked else (" [open]" if cell.is_open else " [closed]")
    return s


def _grid_objects(env):
    """Group every non-wall object on the grid by type -> list of descriptions."""
    groups: dict[str, list[str]] = defaultdict(list)
    grid = env.grid
    for j in range(grid.height):
        for i in range(grid.width):
            cell = grid.get(i, j)
            if cell is None or cell.type == "wall":
                continue
            groups[cell.type].append(_describe_obj(i, j, cell))
    return groups


def _status_flags(env):
    """Introspect the env for zero-arg query helpers and return name -> value.

    Picks methods whose name looks like a state query (``is_``/``prev_`` prefix or
    ``_count`` suffix) that take no required args and return a scalar
    (bool/int/float/str). Anything that needs args, raises, or returns a container
    (already shown in the grid scan) is skipped. No env-specific method names are
    baked in here.
    """
    flags: dict[str, object] = {}
    for name in sorted(dir(env)):
        if name.startswith("_"):
            continue
        if not (name.startswith(_FLAG_PREFIXES) or name.endswith(_FLAG_SUFFIXES)):
            continue
        method = getattr(env, name, None)
        if not callable(method):
            continue
        try:
            params = inspect.signature(method).parameters.values()
            if any(p.default is p.empty
                   and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                   for p in params):
                continue                       # needs arguments -> not a simple query
            value = method()
        except Exception:
            continue
        if isinstance(value, (bool, int, float, str)):
            flags[name] = value
    return flags


def _fmt_flag(value):
    """Render a scalar flag value compactly (bools as True/False, ints as-is)."""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _paired_flags(flags):
    """Pair each current-state getter with its ``prev_*`` snapshot -> ordered rows.

    Returns ``[(base, now, prev, has_prev), ...]`` where ``base`` is the getter name
    with its ``is_``/``prev_`` prefix stripped, so ``is_carrying_key`` and
    ``prev_carrying_key`` share one row and the panel can flag what the last action
    changed. Getters with no ``prev_`` twin (e.g. ``is_on_goal``) render now-only.
    """
    def _base(name):
        for prefix in _FLAG_STRIP:
            if name.startswith(prefix):
                return name[len(prefix):]
        return name

    now, prev = {}, {}
    for name, value in flags.items():
        (prev if name.startswith(_PREV_PREFIX) else now)[_base(name)] = value

    rows, seen = [], set()
    for base in sorted(now):                    # current getters drive the row order
        rows.append((base, now[base], prev.get(base), base in prev))
        seen.add(base)
    for base in sorted(prev):                   # prev-only getters (rare) come after
        if base not in seen:
            rows.append((base, None, prev[base], True))
    return rows


def _flag_value_html(now, prev, has_prev):
    """Value cell for a paired flag: ``now ← prev`` with a marker when it changed."""
    now_txt = html.escape(_fmt_flag(now)) if now is not None else "—"
    if not has_prev:
        return now_txt
    prev_txt = html.escape(_fmt_flag(prev))
    cell = (f"{now_txt} <span style='color:#bbb'>&larr;</span> "
            f"<span style='color:#888'>{prev_txt}</span>")
    if now != prev:                             # the last action changed this flag
        cell += (" <span style='color:#d98e04;font-weight:700' "
                 "title='changed this step'>&#10033;</span>")
    return cell


def _render_png(env):
    """Render the env to PNG bytes, upscaled nearest-neighbour for a crisp, larger image."""
    frame = np.asarray(env.render())
    h = frame.shape[0]
    factor = max(1, min(4, round(_TARGET_PX / h))) if h else 1
    if factor > 1:                              # nearest-neighbour: keeps the tiles crisp
        frame = frame.repeat(factor, axis=0).repeat(factor, axis=1)
    buf = io.BytesIO()
    plt.imsave(buf, frame, format="png")
    return buf.getvalue()


def _badge(text):
    """A small coloured status pill (running / TERMINATED / TRUNCATED)."""
    color = _STATUS_COLORS.get(text, "#777")
    return (f"<span style='background:{color};color:#fff;border-radius:9px;"
            f"padding:1px 9px;font-size:11px;font-weight:600'>{html.escape(text)}</span>")


def _row(key, value):
    """One aligned key/value line in the info panel."""
    return (f"<div style='display:flex;gap:10px;padding:1px 0'>"
            f"<span style='color:#888;min-width:120px'>{html.escape(str(key))}</span>"
            f"<span style='color:#111'>{html.escape(str(value))}</span></div>")


def _kv_html(key, value_html):
    """One aligned key/value line whose value is raw (already-escaped) HTML."""
    return (f"<div style='display:flex;gap:10px;padding:1px 0'>"
            f"<span style='color:#888;min-width:120px'>{html.escape(str(key))}</span>"
            f"<span style='color:#111'>{value_html}</span></div>")


def _section_html(title, body):
    """A titled group wrapping a pre-built body of rows; empty bodies are skipped."""
    if not body:
        return ""
    return (f"<div style='margin-top:11px'>"
            f"<div style='font-weight:700;font-size:10.5px;color:#2e9e5b;"
            f"text-transform:uppercase;letter-spacing:.6px;"
            f"border-bottom:1px solid #e1e1e1;padding-bottom:2px;margin-bottom:3px'>"
            f"{html.escape(title)}</div>{body}</div>")


def _section(title, rows):
    """A titled group of key/value rows; empty groups are skipped."""
    return _section_html(title, "".join(_row(k, v) for k, v in rows))


def _wrapper_names(env):
    """Names of the gym wrappers around the base env, outermost first ([] if bare)."""
    names, e = [], env
    while isinstance(e, _WRAPPER):
        names.append(type(e).__name__)
        e = e.env
    return names


def _net_action(env, top_action):
    """Push a top-level action through the chain's ActionWrappers -> base MiniGrid id."""
    a, e = top_action, env
    while isinstance(e, _WRAPPER):
        if isinstance(e, _ACTION_WRAPPER):
            a = e.action(a)
        e = e.env
    return int(a)


def _button_actions(env):
    """Map each standard button to a *top-level* action for this (possibly wrapped) env.

    Returns ``[(label, base_id, top_action_or_None), ...]``. With no action-remapping
    wrapper the top-level action is just the base MiniGrid id. If the env restricts /
    remaps actions (a ``gym.ActionWrapper``), the top-level index that reaches each base
    action is found by probing ``action_space``; a base action the env doesn't expose
    gets ``None`` so its button is disabled instead of silently doing the wrong thing.
    """
    has_action_wrapper = any(isinstance(w, _ACTION_WRAPPER)
                             for w in _iter_wrappers(env))
    if not has_action_wrapper:
        return [(label, base_id, base_id) for label, base_id in _ACTIONS]
    inv = {}
    try:
        n = int(env.action_space.n)
    except Exception:
        n = 7
    for i in range(n):                          # base MiniGrid id -> top-level index
        try:
            inv.setdefault(_net_action(env, i), i)
        except Exception:
            continue
    return [(label, base_id, inv.get(base_id)) for label, base_id in _ACTIONS]


def _iter_wrappers(env):
    """Yield each wrapper layer around the base env (outermost first)."""
    e = env
    while isinstance(e, _WRAPPER):
        yield e
        e = e.env


def explore_env(make_env, seed=0):
    """Drive any MiniGrid env by hand with buttons, inline.

    ``make_env`` is the env *class* (or any zero-arg factory) defined in the
    notebook -- e.g. ``explore_env(ComplexEnv)`` or ``explore_env(SimpleRoomEnv)``. The
    info panel is fully generic: a grid-object scan plus an auto-discovered set of
    status flags, so it works for every env without per-env code.
    """
    # On Colab, ipywidgets callbacks only fire once the custom widget manager is on.
    if "google.colab" in sys.modules:
        try:
            from google.colab import output as _colab_output
            _colab_output.enable_custom_widget_manager()
        except Exception:
            pass

    state = {"env": make_env(), "steps": 0, "ret": 0.0, "last": "—", "last_r": 0.0,
             "term": False, "trunc": False, "history": []}
    state["env"].reset(seed=seed)

    img = widgets.Image(format="png", layout=widgets.Layout(border="1px solid #ddd"))
    info = widgets.HTML()

    def _refresh():
        env = state["env"]
        base = getattr(env, "unwrapped", env)

        img.value = _render_png(env)

        agent_x, agent_y = (int(v) for v in base.agent_pos)
        status = "TERMINATED" if state["term"] else "TRUNCATED" if state["trunc"] else "running"
        carrying = base.carrying
        carry_str = "None" if carrying is None else \
            f"{type(carrying).__name__}({getattr(carrying, 'color', '')})"
        max_steps = getattr(base, "max_steps", "?")

        header = (f"<div style='font-weight:700;font-size:14px;margin-bottom:2px'>"
                  f"{html.escape(type(base).__name__)} &nbsp;{_badge(status)}</div>")
        wrappers = _wrapper_names(env)
        if wrappers:                            # show the stack so it's clear it's wrapped
            chain = " ← ".join(html.escape(w) for w in wrappers)
            header += (f"<div style='color:#2d7dd2;font-size:11px;margin-bottom:2px'>"
                       f"wrappers: {chain} ← <b>{html.escape(type(base).__name__)}</b></div>")
        mission = getattr(base, "mission", None)
        if mission:
            header += (f"<div style='color:#777;font-style:italic;font-size:12px;"
                       f"margin-bottom:2px'>{html.escape(str(mission))}</div>")

        history = " ".join(state["history"][-_HISTORY_LEN:]) or "—"
        state_rows = [
            ("step", f"{getattr(base, 'step_count', state['steps'])} / {max_steps}"),
            ("last action", state["last"]),
            ("reward (last)", f"{state['last_r']:+.3f}"),
            ("reward (total)", f"{state['ret']:+.3f}"),
            ("history", history),
        ]
        agent_rows = [
            ("agent pos", f"({agent_x}, {agent_y})"),
            ("facing", _DIRS.get(int(base.agent_dir), "?")),
            ("carrying", carry_str),
        ]
        groups = _grid_objects(base)
        obj_rows = [(t, "  ".join(groups[t])) for t in sorted(groups)]
        pairs = _paired_flags(_status_flags(base))
        flags_body = "".join(
            _kv_html(b, _flag_value_html(n, p, hp)) for b, n, p, hp in pairs)
        flags_title = "status (now ← prev)" if any(hp for *_, hp in pairs) else "status"

        info.value = (
            "<div style='font-family:monospace;font-size:12.5px;line-height:1.35;"
            "padding-left:16px;min-width:330px'>"
            + header
            + _section("state", state_rows)
            + _section("agent", agent_rows)
            + _section("grid objects", obj_rows)
            + _section_html(flags_title, flags_body)
            + "</div>"
        )

    def _step(top_action, label, base_id):
        # Step the full wrapper stack so shaped/scaled reward and preprocessing apply;
        # `top_action` already accounts for any action-remapping wrapper.
        if state["term"] or state["trunc"]:
            return
        _, reward, terminated, truncated, _ = state["env"].step(top_action)
        state["steps"] += 1
        state["last_r"] = float(reward)
        state["ret"] += float(reward)
        state["last"] = f"{base_id} ({label})"
        state["history"].append(_GLYPHS.get(base_id, "?"))
        state["term"], state["trunc"] = bool(terminated), bool(truncated)
        _refresh()

    def _reset(use_seed):
        state["env"].reset(seed=seed_box.value if use_seed else None)
        state.update(steps=0, ret=0.0, last="—", last_r=0.0,
                     term=False, trunc=False, history=[])
        _refresh()

    buttons = []
    for label, base_id, top_action in _button_actions(state["env"]):
        disabled = top_action is None          # base action not exposed by this env
        button = widgets.Button(
            description=label, disabled=disabled,
            tooltip="" if not disabled else "not in this env's action subset",
            layout=widgets.Layout(width="92px"))
        if not disabled:
            button.on_click(
                lambda _b, ta=top_action, lab=label, bi=base_id: _step(ta, lab, bi))
        buttons.append(button)

    seed_box = widgets.IntText(value=seed, description="seed", layout=widgets.Layout(width="140px"))
    reset_seed = widgets.Button(description="reset (seed)", button_style="warning")
    reset_rand = widgets.Button(description="reset (random)", button_style="info")
    reset_seed.on_click(lambda _b: _reset(True))
    reset_rand.on_click(lambda _b: _reset(False))

    controls = widgets.VBox([
        widgets.HBox(buttons[:3]),                     # ⟲ left  ↑ forward  ⟳ right
        widgets.HBox(buttons[3:]),                     # pickup  toggle  drop
        widgets.HBox([seed_box, reset_seed, reset_rand]),
    ])
    _refresh()
    display(widgets.HBox([img, info], layout=widgets.Layout(align_items="flex-start")), controls)
