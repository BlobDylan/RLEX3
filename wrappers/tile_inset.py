"""Keep only the informative center of each MiniGrid tile."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TileInsetWrapper(gym.ObservationWrapper):
    """Crop a margin from every tile, keeping the center block.

    MiniGrid paints objects in the middle of each ``tile_size×tile_size`` cell;
    the outer ring is mostly empty padding. Instead of shrinking the whole
    image (which blurs content), we **discard that ring per tile** and
    reassemble a denser grid.

    Example with ``tile_size=8``, ``keep_fraction=0.75`` (≈25% linear cut):
    each tile becomes ``6×6``, so an ``8×8``-cell interior image goes
    ``64×64 → 48×48`` without resampling the agent/goal pixels.
    """

    def __init__(
        self,
        env: gym.Env,
        inset: int | None = None,
        keep_fraction: float = 0.75,
    ) -> None:
        super().__init__(env)
        core = env.unwrapped
        self.tile = int(getattr(core, "_obs_tile_size", 1))
        if self.tile < 2:
            raise ValueError(f"tile_size must be >= 2 for inset, got {self.tile}")

        if inset is None:
            inner = max(1, int(round(self.tile * float(keep_fraction))))
            inset = max(0, (self.tile - inner) // 2)
        self.inset = int(inset)
        self.inner = self.tile - 2 * self.inset
        if self.inner < 1:
            raise ValueError(
                f"inset={self.inset} leaves empty tile (tile_size={self.tile})"
            )

        h, w, c = env.observation_space.shape
        if h % self.tile or w % self.tile:
            raise ValueError(
                f"Obs shape {(h, w)} not divisible by tile_size={self.tile}"
            )
        rows, cols = h // self.tile, w // self.tile
        new_shape = (rows * self.inner, cols * self.inner, c)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=new_shape, dtype=np.uint8
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        h, w, c = obs.shape
        rows, cols = h // self.tile, w // self.tile
        t, inn, ins = self.tile, self.inner, self.inset
        # (rows, tile, cols, tile, C) → take center of each tile → pack.
        tiles = obs.reshape(rows, t, cols, t, c)
        centers = tiles[:, ins : ins + inn, :, ins : ins + inn, :]
        return centers.reshape(rows * inn, cols * inn, c)
