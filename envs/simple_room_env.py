"""SimpleRoomEnv — extracted from the assignment notebook (FIXED env logic).

Sanity-check env: walk onto the green goal in an empty room. Same observation
contract as ``ComplexEnv``. Do not change task dynamics; only ``max_steps`` /
``tile_size`` are free knobs for training.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv as BaseMiniGridEnv


# =============================================================================
# ENVIRONMENT 1: SimpleRoomEnv (empty room, sanity check)   --- FIXED, DO NOT EDIT
# =============================================================================
class SimpleRoomEnv(BaseMiniGridEnv):
    """Empty room: walk onto the green goal. Pass/fail sanity-check env.

    Same observation contract as ``ComplexEnv`` (raw full-grid RGB, uint8 HWC) and
    the standard MiniGrid ``Discrete(7)`` action space. Sparse reward ``+1`` on the
    goal. FIXED — wrap it to add preprocessing; you may only override ``max_steps``.
    """

    def __init__(self, width=10, height=10, max_steps=100, tile_size=4,
                 render_mode="rgb_array", **kwargs):
        assert width >= 4 and height >= 4
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space, width=width, height=height,
            see_through_walls=True, max_steps=max_steps, render_mode=render_mode,
            highlight=False, **kwargs,
        )
        # --- spaces: Discrete(7) actions, raw full-grid RGB observation ---
        self.action_space = spaces.Discrete(7)
        self._obs_tile_size = int(tile_size)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(height * self._obs_tile_size, width * self._obs_tile_size, 3),
            dtype=np.uint8,
        )
        # --- per-episode layout: set by _gen_grid() ---
        self.goal_pos = None

    @staticmethod
    def _gen_mission():
        return "reach the green goal"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # goal in a random corner
        goal_x = int(self.np_random.choice([1, width - 2]))
        goal_y = int(self.np_random.choice([1, height - 2]))
        self.goal_pos = (goal_x, goal_y)
        self.put_obj(Goal(), goal_x, goal_y)

        # agent starts on any non-goal cell, facing a random direction
        while True:
            ax = int(self.np_random.integers(1, width - 1))
            ay = int(self.np_random.integers(1, height - 1))
            if (ax, ay) != self.goal_pos:
                break
        self.agent_pos = (ax, ay)
        self.agent_dir = int(self.np_random.integers(0, 4))
        self.mission = self._gen_mission()

    def gen_obs(self):
        return self.get_frame(highlight=False, tile_size=self._obs_tile_size)

    def is_on_goal(self) -> bool:
        """True iff the agent is standing on the goal cell right now (a win)."""
        return tuple(self.agent_pos) == tuple(self.goal_pos)

    def step(self, action):
        obs, _, _, truncated, info = super().step(action)
        terminated = self.is_on_goal()
        reward = 1.0 if terminated else 0.0
        return obs, reward, terminated, truncated, info
