"""Environment factories — the single source of truth for the wrapper stacks.

Extracted from the notebook so experiments run purely from scripts. Both scripts
and any ad-hoc analysis should import these instead of re-declaring the stack.
"""

from __future__ import annotations

from envs import ComplexEnv, SimpleRoomEnv
from wrappers import (
    ActionSubsetWrapper,
    ComplexPotentialWrapper,
    ComplexShapingWrapper,
    CropOuterWallsWrapper,
    DEFAULT_COMPLEX_SHAPING,
    FrameStackWrapper,
    GrayscaleWrapper,
    ResizeObsWrapper,
    SimpleRoomShapingWrapper,
    TileInsetWrapper,
)

# ComplexEnv: 0 left, 1 right, 2 forward, 3 pickup, 4 drop, 5 toggle (drop 6=done).
COMPLEX_ACTIONS = (0, 1, 2, 3, 4, 5)
# SimpleRoom only needs turn/forward.
SIMPLE_ACTIONS = (0, 1, 2)


def make_complex_env(
    *,
    max_steps: int = 200,
    tile_size: int = 12,
    keep_fraction: float = 0.75,
    cnn_size: int = 64,
    grayscale: bool = True,
    frame_stack: int = 1,
    shaping: dict[str, float] | None = None,
    potential: dict[str, float] | None = None,
    use_enter_right_room: bool = True,
):
    """ComplexEnv + reward shaping + action subset + image pipeline.

    ``grayscale=False`` keeps RGB (key/lava/water are colour-coded). ``frame_stack>1``
    stacks the last N frames. ``potential`` (a dict of ComplexPotentialWrapper kwargs)
    selects **potential-based** shaping (dense, non-farmable) instead of the event
    shaping — this is the from-scratch default that avoids camping local optima.
    """
    env = ComplexEnv(max_steps=max_steps, tile_size=tile_size)
    if potential is not None:
        env = ComplexPotentialWrapper(env, **potential)
    else:
        kw = dict(DEFAULT_COMPLEX_SHAPING)
        if shaping:
            kw.update(shaping)
        kw.pop("use_enter_right_room", None)
        env = ComplexShapingWrapper(env, use_enter_right_room=use_enter_right_room, **kw)
    env = ActionSubsetWrapper(env, action_ids=COMPLEX_ACTIONS)
    env = CropOuterWallsWrapper(env)
    env = TileInsetWrapper(env, keep_fraction=keep_fraction)
    if grayscale:
        env = GrayscaleWrapper(env)
    env = ResizeObsWrapper(env, size=cnn_size)
    if frame_stack and frame_stack > 1:
        env = FrameStackWrapper(env, k=frame_stack)
    return env


def make_simple_env(
    *,
    max_steps: int = 100,
    tile_size: int = 12,
    keep_fraction: float = 0.75,
    cnn_size: int = 64,
    grayscale: bool = True,
    frame_stack: int = 1,
):
    """SimpleRoomEnv sanity stack (grayscale by default, matches Exp7)."""
    env = SimpleRoomEnv(max_steps=max_steps, tile_size=tile_size)
    env = SimpleRoomShapingWrapper(env, goal_scale=50.0, step_penalty=0.1)
    env = ActionSubsetWrapper(env, action_ids=SIMPLE_ACTIONS)
    env = CropOuterWallsWrapper(env)
    env = TileInsetWrapper(env, keep_fraction=keep_fraction)
    if grayscale:
        env = GrayscaleWrapper(env)
    env = ResizeObsWrapper(env, size=cnn_size)
    if frame_stack and frame_stack > 1:
        env = FrameStackWrapper(env, k=frame_stack)
    return env
