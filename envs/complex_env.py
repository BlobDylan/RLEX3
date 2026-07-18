"""ComplexEnv — extracted from the assignment notebook (FIXED env logic).

Do not change task dynamics; only ``max_steps`` / ``tile_size`` are free knobs for training.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Door, Goal, Key, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv as BaseMiniGridEnv


# =============================================================================
# ENVIRONMENT 2: ComplexEnv (key -> door -> water -> lava -> goal)  --- FIXED, DO NOT EDIT
# =============================================================================
class ComplexEnv(BaseMiniGridEnv):
    """Key -> door -> water -> lava -> goal (image-based, sparse reward).

    The left room holds a single locked-door key; the right room holds the goal in
    a top or bottom right corner ringed by lethal ``Lava`` (the ring mirrors with
    the corner), plus several ``Ball`` "water" objects.
    To win: pick up the key, toggle the locked door open, drop the key (only one
    object can be carried), pick up a water ball, ``toggle`` it while facing a lava
    tile (extinguishes that tile and consumes the ball) -- ferrying water until the
    path is clear -- then step onto the goal. Stepping on un-extinguished lava is lethal.

    Observation: raw full-grid RGB ``(H*tile, W*tile, 3)`` uint8. Reward: sparse
    ``+1`` on the goal, ``0`` otherwise. FIXED -- wrap it to add preprocessing / reward
    shaping; you may only override ``max_steps``.


    Helper getters -- for event-based reward shaping and for evaluation / EDA. (The
    observation is the raw RGB image, so there is no tabular "state" to read here; and
    reward may NOT depend on geometry / distance, so the position getters are for
    analysis, not for shaping.)
      curr (now)            : is_carrying_key / is_carrying_water / is_door_open /
                              is_door_locked / is_on_lava / is_on_goal / is_in_right_room /
                              extinguished_lava_count
      prev (start of step)  : prev_carrying_key / prev_carrying_water / prev_door_open /
                              prev_door_locked / prev_in_right_room / prev_extinguished_lava_count
      positions (eval & EDA): key_position / water_positions / lava_positions
    Detect an event as *curr and not prev*, e.g.
    just picked up water == ``is_carrying_water() and not prev_carrying_water()``.
    Latch a milestone ("ever opened the door") yourself in your wrapper if you want one.
    """

    def __init__(self, max_steps=400, width=10, height=10, tile_size=32,
                 render_mode="rgb_array", partition_col=3,
                 agent_start_pos=(1, 1), goal_corner="random",
                 n_water=3, water_corner="random",
                 key_door_color="purple", water_color="blue", **kwargs):
        assert width >= 8 and height >= 5
        assert 2 <= partition_col < width - 2   # leave >=1 right-room column for the goal
        assert goal_corner in ("top", "bottom", "random")
        assert water_corner in ("top", "bottom", "random")
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space, width=width, height=height,
            see_through_walls=True, max_steps=max_steps, render_mode=render_mode,
            highlight=False, **kwargs,
        )
        # --- spaces: Discrete(7) actions, raw full-grid RGB observation ----------------
        self.action_space = spaces.Discrete(7)
        self._obs_tile_size = int(tile_size)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(height * self._obs_tile_size, width * self._obs_tile_size, 3),
            dtype=np.uint8,
        )

        # --- task configuration: fixed for the env's lifetime (from constructor args) --
        self.partition_col = partition_col
        self.key_color = key_door_color
        self.door_color = key_door_color
        self.water_color = water_color
        self.n_water = int(n_water)
        self.goal_corner = goal_corner           # "top" / "bottom" / "random" (resolved per episode)
        self.water_corner = water_corner         # "top" / "bottom" / "random" (resolved per episode)
        # agent start: None -> randomized per episode; otherwise reused each episode.
        self.fixed_agent_start_pos = (None if agent_start_pos is None
                                      else tuple(int(v) for v in agent_start_pos))

        # --- per-episode layout: (re)filled by _gen_grid() on every reset ---------------
        self.door_pos = None
        self.goal_pos = None
        self.lava_ring = ()                 # ring around the goal corner (mirrors it)
        self._door = None

        # --- episode progress: count of lava tiles extinguished so far this episode -----
        self._extinguished_lava_count = 0

        # --- previous-step snapshot: each value as of the START of the current step ------
        self._prev_carrying_key = False
        self._prev_carrying_water = False
        self._prev_door_open = False
        self._prev_door_locked = False
        self._prev_in_right_room = False
        self._prev_extinguished_lava_count = 0

    @staticmethod
    def _gen_mission():
        return "get the key, open the door, extinguish the lava with water, then reach the goal"

    def _resolve_corner(self, mode):
        """Resolve a 'top'/'bottom'/'random' corner mode to a concrete 'top'/'bottom'
        for this episode ('random' is re-rolled every episode via the env RNG)."""
        if mode == "random":
            return "top" if int(self.np_random.integers(0, 2)) == 0 else "bottom"
        return mode

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # partition wall splitting left (key) from right (goal + lava + water)
        for y in range(height):
            self.grid.set(self.partition_col, y, Wall())

        # locked door at a random partition row
        door_y = int(self.np_random.integers(1, height - 1))
        self.door_pos = (self.partition_col, door_y)
        self._door = Door(self.door_color, is_locked=True)
        self.grid.set(self.partition_col, door_y, self._door)

        # agent start: fixed if provided, else a random left-room cell, facing random
        if self.fixed_agent_start_pos is not None:
            self.agent_pos = self.fixed_agent_start_pos
        else:
            ax = int(self.np_random.integers(1, self.partition_col))
            ay = int(self.np_random.integers(1, height - 1))
            self.agent_pos = (ax, ay)
        self.agent_dir = int(self.np_random.integers(0, 4))

        # the single key on a left-room cell (never UNDER the agent). It MAY land on the
        # door-approach cell -- no clearance is reserved, so it can block the way.
        blocked = {tuple(self.agent_pos)}
        left_cells = [(x, y) for x in range(1, self.partition_col)
                      for y in range(1, height - 1) if (x, y) not in blocked]
        kx, ky = left_cells[int(self.np_random.integers(len(left_cells)))]
        self.grid.set(kx, ky, Key(self.key_color))

        # goal in a right-room corner (top or bottom); the lava ring mirrors with it so
        # it always cups the goal corner. The goal stays in the far-right column (x=width-2).
        goal_corner = self._resolve_corner(self.goal_corner)
        gx = width - 2
        if goal_corner == "bottom":
            self.goal_pos = (gx, height - 2)
            self.lava_ring = ((gx - 1, height - 2), (gx, height - 3), (gx - 1, height - 3))
        else:  # top: vertical mirror of the bottom ring
            self.goal_pos = (gx, 1)
            self.lava_ring = ((gx - 1, 1), (gx, 2), (gx - 1, 2))
        self.put_obj(Goal(), self.goal_pos[0], self.goal_pos[1])

        # lethal lava ringing the goal corner
        for lx, ly in self.lava_ring:
            self.grid.set(lx, ly, Lava())

        # water balls: a contiguous cluster on the top or bottom row (water_corner), starting
        # at the left edge of the right room. Cells overlapping the goal/lava are skipped, so a
        # large n_water relative to the room width may place fewer than n_water balls.
        water_corner = self._resolve_corner(self.water_corner)
        wy = 1 if water_corner == "top" else height - 2
        reserved = {tuple(self.goal_pos)} | set(self.lava_ring)
        placements = []
        for i in range(self.n_water):
            wx = self.partition_col + 1 + i
            if wx > width - 2:
                break
            if (wx, wy) not in reserved:
                placements.append((wx, wy))
        for wx, wy in placements:
            self.grid.set(wx, wy, Ball(self.water_color))

        # reset the per-episode extinguished-lava count (the prev snapshot is taken in
        # reset, once the base class has finished setting up self.carrying etc.)
        self._extinguished_lava_count = 0
        self.mission = self._gen_mission()

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)   # calls _gen_grid()
        self._snapshot_prev()                                   # prev == the fresh reset state
        info = dict(info)
        info.update(self._info_snapshot(terminated=False, extinguished_now=0))
        return obs, info

    def gen_obs(self):
        return self.get_frame(highlight=False, tile_size=self._obs_tile_size)

    # ============================ curr: instantaneous state (true only RIGHT NOW) =======
    def is_carrying_key(self) -> bool:
        """True iff a key is in hand right now."""
        return isinstance(self.carrying, Key)

    def is_carrying_water(self) -> bool:
        """True iff a water ball is in hand right now."""
        return isinstance(self.carrying, Ball)

    def is_door_open(self) -> bool:
        """True iff the door is open right now."""
        return bool(self._door is not None and self._door.is_open)

    def is_door_locked(self) -> bool:
        """True iff the door is locked right now (opening it unlocks it for good)."""
        return bool(self._door is not None and self._door.is_locked)

    def is_on_lava(self) -> bool:
        """True iff the agent is standing on a lava tile right now (a lethal lava death)."""
        return isinstance(self.grid.get(int(self.agent_pos[0]), int(self.agent_pos[1])), Lava)

    def is_on_goal(self) -> bool:
        """True iff the agent is standing on the goal cell right now (a win)."""
        return tuple(int(v) for v in self.agent_pos) == tuple(int(v) for v in self.goal_pos)

    def is_in_right_room(self) -> bool:
        """True iff the agent is past the partition wall (right room)."""
        return int(self.agent_pos[0]) > self.partition_col

    def extinguished_lava_count(self) -> int:
        """How many lava tiles have been extinguished so far this episode (only grows)."""
        return self._extinguished_lava_count

    # ============================ prev: same state, as of the START of this step ========
    def prev_carrying_key(self) -> bool: return self._prev_carrying_key
    def prev_carrying_water(self) -> bool: return self._prev_carrying_water
    def prev_door_open(self) -> bool: return self._prev_door_open
    def prev_door_locked(self) -> bool: return self._prev_door_locked
    def prev_in_right_room(self) -> bool: return self._prev_in_right_room
    def prev_extinguished_lava_count(self) -> int: return self._prev_extinguished_lava_count

    # ============================ positions / layout: for evaluation / EDA ==============
    def key_position(self) -> tuple[int, int] | None:
        """The single key's (x, y) while uncarried, else None (its color is ``key_color``)."""
        for x in range(self.width):
            for y in range(self.height):
                if isinstance(self.grid.get(x, y), Key):
                    return (x, y)
        return None

    def water_positions(self) -> list:
        """Remaining (uncarried) water balls on the grid."""
        return [(x, y) for x in range(self.width) for y in range(self.height)
                if isinstance(self.grid.get(x, y), Ball)]

    def lava_positions(self) -> list:
        """Lava tiles still present (not yet extinguished)."""
        return [(x, y) for x in range(self.width) for y in range(self.height)
                if isinstance(self.grid.get(x, y), Lava)]

    # ============================ internal bookkeeping ==================================
    def _snapshot_prev(self):
        """Capture the instantaneous state so the prev_* getters describe the value as
        of the START of the next step (i.e. the last step's value)."""
        self._prev_carrying_key = self.is_carrying_key()
        self._prev_carrying_water = self.is_carrying_water()
        self._prev_door_open = self.is_door_open()
        self._prev_door_locked = self.is_door_locked()
        self._prev_in_right_room = self.is_in_right_room()
        self._prev_extinguished_lava_count = self._extinguished_lava_count

    def _faced_lava_to_extinguish(self, action):
        """Return the (x, y) of the lava tile this action would extinguish -- i.e.
        ``toggle`` while carrying water and facing lava -- or None. (Base MiniGrid has
        no toggle behaviour for lava, so we resolve it ourselves.)"""
        if int(action) != int(self.actions.toggle) or not isinstance(self.carrying, Ball):
            return None
        fx, fy = (int(v) for v in self.front_pos)
        return (fx, fy) if isinstance(self.grid.get(fx, fy), Lava) else None

    def _resolve_goal(self, terminated):
        """Win (+1, terminate) on the goal cell; otherwise keep the base ``terminated``
        so a step onto un-extinguished lava stays lethal (death ends with no reward)."""
        if self.is_on_goal():
            return 1.0, True
        return 0.0, terminated

    def _info_snapshot(self, *, terminated, extinguished_now):
        """Ground-truth signals the env owns, surfaced through ``info`` for logging /
        evaluation."""
        return {
            "extinguished_lava_count": int(self._extinguished_lava_count),
            "n_lava": len(self.lava_ring),
            "extinguished_this_step": int(extinguished_now),
            "is_success": bool(self.is_on_goal()),
            "died_on_lava": bool(terminated and not self.is_on_goal()),
        }

    # ============================ step / reward =========================================
    def step(self, action):
        self._snapshot_prev()   # 'prev' = state as of the start of this step

        extinguish_target = self._faced_lava_to_extinguish(action)
        obs, _, terminated, truncated, info = super().step(action)

        extinguished_now = 0
        if extinguish_target is not None:
            ex, ey = extinguish_target
            self.grid.set(ex, ey, None)     # lava -> empty floor
            self.carrying = None            # the water ball is consumed
            self._extinguished_lava_count += 1
            extinguished_now = 1
            obs = self.gen_obs()            # refresh the image after the change

        reward, terminated = self._resolve_goal(terminated)

        info = dict(info)
        info.update(self._info_snapshot(terminated=terminated, extinguished_now=extinguished_now))
        return obs, reward, terminated, truncated, info
