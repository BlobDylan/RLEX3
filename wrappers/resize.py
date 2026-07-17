"""Resize image observations to a fixed CNN input size."""

from __future__ import annotations

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ResizeObsWrapper(gym.ObservationWrapper):
    """Resize H×W image obs with nearest-neighbor (keeps crisp MiniGrid pixels)."""

    def __init__(self, env: gym.Env, size: int | tuple[int, int] = 64) -> None:
        super().__init__(env)
        if isinstance(size, int):
            self.out_h = self.out_w = int(size)
        else:
            self.out_h, self.out_w = int(size[0]), int(size[1])

        h, w, c = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.out_h, self.out_w, c), dtype=np.uint8
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # cv2 wants (W, H); nearest keeps agent/goal blocks sharp.
        out = cv2.resize(obs, (self.out_w, self.out_h), interpolation=cv2.INTER_NEAREST)
        if out.ndim == 2:  # cv2 drops singleton channel on grayscale
            out = out[:, :, None]
        return out
