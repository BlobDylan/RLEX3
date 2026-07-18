"""Frame-stacking wrapper — gives a memoryless agent short-term temporal context.

ComplexEnv is partially observable: whether the agent is *carrying* a key/water
is invisible in the rendered pixels (the object is removed from the grid on
pickup and not drawn on the agent). Stacking the last ``k`` frames lets a CNN see
the pickup/consume transitions and infer inventory state.
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FrameStackWrapper(gym.Wrapper):
    """Stack the last ``k`` image observations along the channel axis.

    ``(H, W, C)`` → ``(H, W, C * k)`` uint8. The oldest frame is index 0..C-1,
    the newest is the last block.
    """

    def __init__(self, env: gym.Env, k: int = 4) -> None:
        super().__init__(env)
        self.k = max(1, int(k))
        h, w, c = env.observation_space.shape
        self._frames: deque[np.ndarray] = deque(maxlen=self.k)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(h, w, c * self.k), dtype=np.uint8
        )

    def _stacked(self) -> np.ndarray:
        return np.concatenate(list(self._frames), axis=-1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._frames.clear()
        for _ in range(self.k):
            self._frames.append(obs)
        return self._stacked(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._stacked(), reward, terminated, truncated, info
