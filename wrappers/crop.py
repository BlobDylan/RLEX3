"""Crop constant outer-wall border from MiniGrid full-grid renders."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CropOuterWallsWrapper(gym.ObservationWrapper):
    """Remove the outer wall ring from a full-grid MiniGrid image.

    MiniGrid rooms are bordered by a 1-cell wall. With observation tile size
    ``T``, that is a ``T``-pixel border on each side. Cropping it drops a
    constant, uninformative frame and shrinks the CNN input
    (e.g. ``40×40`` → ``32×32`` for a ``10×10`` room with ``tile_size=4``).

    Works on RGB ``(H, W, 3)`` and grayscale ``(H, W, 1)``.
    """

    def __init__(self, env: gym.Env, wall_cells: int = 1) -> None:
        super().__init__(env)
        self.wall_cells = int(wall_cells)

        core = env.unwrapped
        tile = int(getattr(core, "_obs_tile_size", 1))
        self.pad = self.wall_cells * tile

        h, w, c = env.observation_space.shape
        if self.pad * 2 >= h or self.pad * 2 >= w:
            raise ValueError(
                f"Crop pad={self.pad} too large for observation shape {(h, w, c)}"
            )
        new_shape = (h - 2 * self.pad, w - 2 * self.pad, c)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=new_shape, dtype=np.uint8
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        p = self.pad
        return obs[p:-p, p:-p, ...]
