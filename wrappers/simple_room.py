"""Reward shaping for SimpleRoomEnv (legal dense signal; no distance/geometry)."""

from __future__ import annotations

import gymnasium as gym


class SimpleRoomShapingWrapper(gym.Wrapper):
    """Scale the sparse goal reward and add a constant per-step cost.

    Inspired by the HW2 EmptyEnv shaping (``+50`` on goal, ``-0.1``/step), but
    implemented as a wrapper so the fixed env class stays untouched.

    - Goal: ``goal_scale * env_reward`` (env gives ``1.0`` on success).
    - Every step: subtract ``step_penalty`` (encourages shorter paths).
    - No distance / position features — assignment-safe.
    """

    def __init__(
        self,
        env: gym.Env,
        goal_scale: float = 50.0,
        step_penalty: float = 0.1,
    ) -> None:
        super().__init__(env)
        self.goal_scale = float(goal_scale)
        self.step_penalty = float(step_penalty)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = self.goal_scale * float(reward) - self.step_penalty
        info = dict(info)
        info["success"] = bool(terminated)
        return obs, reward, terminated, truncated, info
