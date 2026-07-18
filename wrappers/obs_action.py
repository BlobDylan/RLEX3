"""Observation / action helpers used by training factories (not assignment examples)."""

from __future__ import annotations

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class GrayscaleWrapper(gym.ObservationWrapper):
    """RGB (H, W, 3) → grayscale (H, W, 1) uint8."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        h, w, _ = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(h, w, 1), dtype=np.uint8
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return gray[:, :, None]


class ActionSubsetWrapper(gym.ActionWrapper):
    """Expose a contiguous Discrete subset of MiniGrid actions."""

    def __init__(self, env: gym.Env, action_ids: tuple[int, ...] = (0, 1, 2, 3, 4, 5)) -> None:
        super().__init__(env)
        self.action_ids = tuple(int(a) for a in action_ids)
        self.action_space = spaces.Discrete(len(self.action_ids))

    def action(self, act: int) -> int:
        return self.action_ids[int(act)]
