"""Event-based reward shaping for ComplexEnv (no distance / geometry)."""

from __future__ import annotations

from typing import Any

import gymnasium as gym


# Locked Step 6a magnitudes (first-time / per-spawn-tile anti-farm rules).
# Max stage sum ≈ 5+10+5+2×3+5×3 ≈ 41; goal_scale=50 is above that but closer —
# user choice (match SimpleRoom). Retune in 6d if needed.
DEFAULT_COMPLEX_SHAPING: dict[str, float] = {
    "goal_scale": 50.0,  # env +1 → +50 (match SimpleRoom)
    "step_penalty": 0.1,  # same as SimpleRoom
    "key_pickup": 5.0,  # first time only (anti pick/drop farm)
    "door_open": 10.0,  # first time only
    "enter_right_room": 5.0,  # first time only
    "leave_right_room": 5.0,  # every return to left (discourage backtracking)
    "key_drop": 5.0,  # first drop in right room (free hand for water)
    "key_drop_locked_left": 5.0,  # every drop in left while door still locked
    "water_pickup": 2.0,  # once per *original* water tile (not drop/re-pick)
    "lava_extinguish": 10.0,  # per tile (env removes tile — not repeatable)
    "lava_death": 10.0,  # subtracted on lethal lava (terminated, not success)
}


class ComplexShapingWrapper(gym.Wrapper):
    """Scale the sparse goal reward and add legal **first-time** event bonuses.

    Anti-reward-hacking:
      - key / door / enter-right / key-drop: paid at most once per episode (latched).
      - key-drop bonus only in the **right room** (need free hand for water).
      - key-drop in left while door locked: penalized **every** time.
      - leave-right: penalized **every** time the agent crosses back to the left
        (enter bonus stays first-time only).
      - water: paid once per **original spawn tile**. Dropping and re-picking the
        same ball (or picking from a drop cell) does not pay again.
      - lava extinguish: env deletes the tile, so it cannot be farmed.
      - positions are used only to latch *which water spawns already paid* —
        never as a distance/geometry reward feature into the agent.

    Stage latches in ``info`` (Step 6b+): ``stage_key`` … ``stage_goal``.
    ``info["reward_breakdown"]`` lists which terms fired this step.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        goal_scale: float = DEFAULT_COMPLEX_SHAPING["goal_scale"],
        step_penalty: float = DEFAULT_COMPLEX_SHAPING["step_penalty"],
        key_pickup: float = DEFAULT_COMPLEX_SHAPING["key_pickup"],
        door_open: float = DEFAULT_COMPLEX_SHAPING["door_open"],
        enter_right_room: float = DEFAULT_COMPLEX_SHAPING["enter_right_room"],
        leave_right_room: float = DEFAULT_COMPLEX_SHAPING["leave_right_room"],
        key_drop: float = DEFAULT_COMPLEX_SHAPING["key_drop"],
        key_drop_locked_left: float = DEFAULT_COMPLEX_SHAPING["key_drop_locked_left"],
        water_pickup: float = DEFAULT_COMPLEX_SHAPING["water_pickup"],
        water_pickup_once: bool = True,
        lava_extinguish: float = DEFAULT_COMPLEX_SHAPING["lava_extinguish"],
        lava_death: float = DEFAULT_COMPLEX_SHAPING["lava_death"],
        lava_death_with_water: float | None = None,
        stall_grace: int = 8,
        stall_penalty: float = 0.0,
        use_enter_right_room: bool = True,
        use_leave_right_room: bool = True,
    ) -> None:
        super().__init__(env)
        self.goal_scale = float(goal_scale)
        self.step_penalty = float(step_penalty)
        self.key_pickup = float(key_pickup)
        self.door_open = float(door_open)
        self.enter_right_room = float(enter_right_room)
        self.leave_right_room = float(leave_right_room)
        self.key_drop = float(key_drop)
        self.key_drop_locked_left = float(key_drop_locked_left)
        self.water_pickup = float(water_pickup)
        self.water_pickup_once = bool(water_pickup_once)
        self.lava_extinguish = float(lava_extinguish)
        self.lava_death = float(lava_death)
        # If set, use a DIFFERENT (typically smaller) death penalty when the agent dies
        # to lava while carrying water — remove the fear that keeps it out of the lava
        # region so exploration there can stumble onto the toggle-extinguish. ``None``
        # keeps a single ``lava_death`` for all deaths (original behaviour).
        self.lava_death_with_water = (
            None if lava_death_with_water is None else float(lava_death_with_water)
        )
        # Anti-camp: once the agent's (x,y) has been unchanged for >= stall_grace steps,
        # add an ESCALATING penalty each further stationary step until it moves (0 = off).
        self.stall_grace = int(stall_grace)
        self.stall_penalty = float(stall_penalty)
        self.use_enter_right_room = bool(use_enter_right_room)
        self.use_leave_right_room = bool(use_leave_right_room)
        self._reset_latches()

    def _reset_latches(self) -> None:
        self._stage_key = False
        self._stage_door = False
        self._stage_right = False
        self._stage_water = False
        self._stage_lava = False
        self._stage_goal = False
        # Payment latches (separate from stage metrics).
        self._paid_key = False
        self._paid_door = False
        self._paid_right = False
        self._paid_key_drop = False
        self._paid_water = False
        # Original water spawn cells still eligible for a pickup bonus.
        self._awardable_water: set[tuple[int, int]] = set()
        # Anti-camp position tracking.
        self._last_pos: tuple[int, int] | None = None
        self._stationary_steps = 0

    def reset(self, **kwargs: Any):
        self._reset_latches()
        obs, info = self.env.reset(**kwargs)
        core = self.env.unwrapped
        self._awardable_water = {
            (int(x), int(y)) for x, y in core.water_positions()
        }
        info = self._augment_info(dict(info), reward_breakdown={})
        return obs, info

    def step(self, action):
        core = self.env.unwrapped
        # Snapshot water cells *before* the transition (for per-tile first pickup).
        waters_before = {(int(x), int(y)) for x, y in core.water_positions()}

        obs, reward, terminated, truncated, info = self.env.step(action)
        core = self.env.unwrapped
        info = dict(info)
        breakdown: dict[str, float] = {}

        # Sparse env reward is 1.0 on goal, else 0.0 (lava death also 0).
        base = float(reward)
        shaped = self.goal_scale * base
        if base != 0.0:
            breakdown["goal"] = shaped

        shaped -= self.step_penalty
        breakdown["step_penalty"] = -self.step_penalty

        # Escalating anti-camp penalty: if the agent's (x,y) has not changed for
        # >= stall_grace steps, subtract a penalty that grows each further stationary
        # step (ramped over ~10 steps, then held) until it moves. Turning in place counts
        # as stationary (position unchanged), so idle-spinning is punished too.
        if self.stall_penalty > 0.0:
            pos = (int(core.agent_pos[0]), int(core.agent_pos[1]))
            if self._last_pos is not None and pos == self._last_pos:
                self._stationary_steps += 1
            else:
                self._stationary_steps = 0
            self._last_pos = pos
            over = min(self._stationary_steps - self.stall_grace + 1, 10)
            if over > 0:
                stall = self.stall_penalty * float(over)
                shaped -= stall
                breakdown["stall_penalty"] = -stall

        # --- first-time / per-original-tile events ---
        if (
            not self._paid_key
            and core.is_carrying_key()
            and not core.prev_carrying_key()
        ):
            shaped += self.key_pickup
            breakdown["key_pickup"] = self.key_pickup
            self._paid_key = True
            self._stage_key = True

        if (
            not self._paid_door
            and core.is_door_open()
            and not core.prev_door_open()
        ):
            shaped += self.door_open
            breakdown["door_open"] = self.door_open
            self._paid_door = True
            self._stage_door = True

        if core.is_in_right_room() and not core.prev_in_right_room():
            self._stage_right = True
            if self.use_enter_right_room and not self._paid_right:
                shaped += self.enter_right_room
                breakdown["enter_right_room"] = self.enter_right_room
                self._paid_right = True
        elif core.is_in_right_room():
            self._stage_right = True
        elif (
            self.use_leave_right_room
            and self.leave_right_room != 0.0
            and core.prev_in_right_room()
            and not core.is_in_right_room()
        ):
            # Crossed back through the door to the left — every time (not latched).
            shaped -= self.leave_right_room
            breakdown["leave_right_room"] = -self.leave_right_room

        just_dropped_key = (
            self._paid_key
            and core.prev_carrying_key()
            and not core.is_carrying_key()
        )
        # First key-drop once the door is open (hands free → can pick water). Dropping
        # while the door is still locked is neither rewarded nor penalised — the key is
        # on the floor and can be picked back up, so a penalty only froze exploration.
        if (
            just_dropped_key
            and not self._paid_key_drop
            and self.key_drop != 0.0
            and core.is_door_open()
        ):
            shaped += self.key_drop
            breakdown["key_drop"] = self.key_drop
            self._paid_key_drop = True

        if core.is_carrying_water() and not core.prev_carrying_water():
            waters_after = {(int(x), int(y)) for x, y in core.water_positions()}
            disappeared = waters_before - waters_after
            new_originals = [p for p in disappeared if p in self._awardable_water]
            for p in new_originals:
                self._awardable_water.discard(p)  # a dropped/re-picked ball never re-pays
            if new_originals:
                self._stage_water = True
                if self.water_pickup_once:
                    # Reward only the FIRST bucket picked up this episode (anti-hoard): the
                    # per-bucket variant let the agent farm ~n_water*water_pickup by collecting
                    # every ball WITHOUT extinguishing, an easy rival to the harder lava reward.
                    if not self._paid_water:
                        shaped += self.water_pickup
                        breakdown["water_pickup"] = self.water_pickup
                        self._paid_water = True
                else:
                    paid = self.water_pickup * len(new_originals)
                    shaped += paid
                    breakdown["water_pickup"] = paid

        extinguished = int(info.get("extinguished_now", 0) or 0)
        if extinguished <= 0:
            extinguished = max(
                0,
                int(core.extinguished_lava_count())
                - int(core.prev_extinguished_lava_count()),
            )
        if extinguished > 0:
            bonus = self.lava_extinguish * float(extinguished)
            shaped += bonus
            breakdown["lava_extinguish"] = bonus
            self._stage_lava = True

        if terminated and not core.is_on_goal():
            # Contextual death penalty: dying with water in hand can be treated as less
            # bad (or free) to stop lava-fear from suppressing exploration in the lava
            # region. Water persists through a lethal step (only ``toggle`` consumes it).
            carrying_water = core.is_carrying_water() or core.prev_carrying_water()
            penalty = self.lava_death
            if carrying_water and self.lava_death_with_water is not None:
                penalty = self.lava_death_with_water
            if penalty != 0.0:
                shaped -= penalty
                breakdown["lava_death"] = -penalty

        if terminated and core.is_on_goal():
            self._stage_goal = True

        # Stage metrics stay sticky even when bonus already paid earlier.
        if core.is_carrying_key() or self._paid_key:
            self._stage_key = True
        if core.is_door_open() or self._paid_door:
            self._stage_door = True

        info = self._augment_info(info, reward_breakdown=breakdown)
        info["success"] = bool(terminated and core.is_on_goal())
        info["died_on_lava"] = bool(terminated and not core.is_on_goal())
        info["awardable_water_left"] = len(self._awardable_water)
        return obs, shaped, terminated, truncated, info

    def _augment_info(
        self, info: dict[str, Any], *, reward_breakdown: dict[str, float]
    ) -> dict[str, Any]:
        info["reward_breakdown"] = reward_breakdown
        info["stage_key"] = self._stage_key
        info["stage_door"] = self._stage_door
        info["stage_right"] = self._stage_right
        info["stage_water"] = self._stage_water
        info["stage_lava"] = self._stage_lava
        info["stage_goal"] = self._stage_goal
        return info

    def shaping_config(self) -> dict[str, float | bool]:
        """Current magnitudes (for logging / debate)."""
        return {
            "goal_scale": self.goal_scale,
            "step_penalty": self.step_penalty,
            "key_pickup": self.key_pickup,
            "door_open": self.door_open,
            "enter_right_room": self.enter_right_room,
            "leave_right_room": self.leave_right_room,
            "key_drop": self.key_drop,
            "key_drop_locked_left": self.key_drop_locked_left,
            "water_pickup": self.water_pickup,
            "lava_extinguish": self.lava_extinguish,
            "lava_death": self.lava_death,
            "use_enter_right_room": self.use_enter_right_room,
            "use_leave_right_room": self.use_leave_right_room,
            "anti_farm": "first-time milestones; key-drop once in right; key-drop locked-left every; water once per spawn; leave-right every",
            "max_stage_shaping_est": (
                self.key_pickup
                + self.door_open
                + (self.enter_right_room if self.use_enter_right_room else 0.0)
                + self.key_drop
                + self.water_pickup * 3.0  # default n_water
                + self.lava_extinguish * 3.0  # lava ring size
            ),
        }


class ComplexPotentialWrapper(gym.Wrapper):
    """Potential-based reward shaping (Ng, Harada & Russell 1999) for ComplexEnv.

    Reward = ``F + goal`` where ``F = gamma·Phi(s') - Phi(s)`` and ``Phi`` is a monotonic
    count of task milestones reached (key → door → right-room → water → each lava tile).
    Because ``F`` telescopes over a trajectory (Σ = γ^T·Phi(s_T) - Phi(s_0)), there is
    **no bankable reward to camp on** — standing still at a stage is mildly negative and
    only *progress* pays — yet the optimal policy is provably unchanged. The sparse goal
    reward (``goal_scale``) is the one lasting reward; lava death is penalised.

    Set ``gamma`` to the agent's discount so the shaping stays policy-invariant.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        gamma: float = 0.99,
        potential_scale: float = 1.0,
        goal_scale: float = 10.0,
        lava_death: float = 5.0,
    ) -> None:
        super().__init__(env)
        self.gamma = float(gamma)
        self.potential_scale = float(potential_scale)
        self.goal_scale = float(goal_scale)
        self.lava_death = float(lava_death)
        self._reset_latches()

    def _reset_latches(self) -> None:
        self._key = self._door = self._right = self._water = self._goal = False
        self._lavas = 0
        self._prev_phi = 0.0

    def _progress(self) -> int:
        return (
            int(self._key) + int(self._door) + int(self._right)
            + int(self._water) + int(self._lavas)
        )

    def _phi(self) -> float:
        return self.potential_scale * self._progress()

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        self._reset_latches()
        self._prev_phi = self._phi()
        return obs, self._augment(dict(info))

    def step(self, action):
        core = self.env.unwrapped
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Monotonic milestone latches → the potential only ever increases.
        self._key = self._key or core.is_carrying_key()
        self._door = self._door or core.is_door_open()
        self._right = self._right or core.is_in_right_room()
        self._water = self._water or core.is_carrying_water()
        self._lavas = max(self._lavas, int(core.extinguished_lava_count()))
        self._goal = self._goal or core.is_on_goal()

        phi = self._phi()
        done = bool(terminated or truncated)
        # Potential shaping on non-terminal steps (Phi(terminal) := 0 by convention,
        # but we let the sparse goal reward carry the final transition instead).
        shaped = 0.0 if done else (self.gamma * phi - self._prev_phi)
        self._prev_phi = phi

        shaped += self.goal_scale * float(reward)  # env reward is +1 exactly on the goal
        if terminated and not core.is_on_goal():
            shaped -= self.lava_death

        info = self._augment(dict(info))
        info["success"] = bool(terminated and core.is_on_goal())
        info["died_on_lava"] = bool(terminated and not core.is_on_goal())
        return obs, shaped, terminated, truncated, info

    def _augment(self, info: dict[str, Any]) -> dict[str, Any]:
        info["stage_key"] = self._key
        info["stage_door"] = self._door
        info["stage_right"] = self._right
        info["stage_water"] = self._water
        info["stage_lava"] = self._lavas > 0
        info["stage_goal"] = self._goal
        return info
