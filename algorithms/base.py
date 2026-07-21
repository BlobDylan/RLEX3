"""Shared base class for value-based, policy-based, and actor-critic agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch


from utils.device import get_torch_device, seed_everything


class BaseAlgorithm(ABC):
    """Minimal interface shared by all agents (DQN, REINFORCE, A2C, ...).

    Subclasses own their networks and update rules. This base only defines the
    common training / evaluation / checkpoint surface so experiments stay
    comparable across families.
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_actions: int,
        device: str | torch.device | None = None,
        seed: int | None = None,
    ) -> None:
        self.obs_shape = tuple(obs_shape)
        self.n_actions = int(n_actions)
        if device is None:
            self.device = get_torch_device()
        else:
            self.device = torch.device(device)
        self.seed = seed

        if seed is not None:
            seed_everything(seed, self.device)

        self.total_steps = 0
        self.total_episodes = 0

    # ------------------------------------------------------------------
    # Required by every algorithm
    # ------------------------------------------------------------------

    @abstractmethod
    def select_action(self, obs: np.ndarray, *, explore: bool = True) -> int:
        """Pick an action for ``obs``.

        Args:
            obs: Preprocessed observation from the env (numpy).
            explore: If True, allow exploration (e.g. epsilon-greedy / sampling).
                     If False, act greedily / deterministically (for evaluation).
        """

    @abstractmethod
    def train(
        self,
        env_fn,
        *,
        total_timesteps: int,
        **kwargs: Any,
    ) -> dict[str, list[float]]:
        """Run the full training loop.

        Args:
            env_fn: Zero-arg callable that returns a fresh (wrapped) env.
            total_timesteps: Environment steps to collect / train on.

        Returns:
            History dict with lists of scalars (e.g. episode returns, losses)
            suitable for plotting into ``graphs/``.
        """

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist networks / optimizer state to ``path``."""

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Restore networks / optimizer state from ``path``."""

    # ------------------------------------------------------------------
    # Shared helpers (override if needed)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        env,
        *,
        n_episodes: int = 10,
        max_steps: int | None = None,
        seed: int | None = None,
    ) -> dict[str, float]:
        """Roll out ``n_episodes`` greedily and return mean / std return + success rate.

        If the env puts ``stage_*`` / ``success`` in ``info`` (ComplexEnv), those are
        used instead of treating every ``terminated`` as a win (lava death).
        """
        stage_keys = (
            "stage_key",
            "stage_door",
            "stage_right",
            "stage_water",
            "stage_lava",
            "stage_goal",
        )
        returns: list[float] = []
        lengths: list[int] = []
        successes: list[float] = []
        stage_hits: dict[str, list[float]] = {k: [] for k in stage_keys}
        saw_stages = False

        for ep in range(n_episodes):
            obs, info = env.reset(seed=None if seed is None else seed + ep)
            done = False
            ep_return = 0.0
            ep_len = 0

            while not done:
                action = self.select_action(obs, explore=False)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += float(reward)
                ep_len += 1
                done = terminated or truncated
                if max_steps is not None and ep_len >= max_steps:
                    break

            if "success" in info:
                reached_goal = bool(info["success"])
            else:
                reached_goal = bool(terminated)

            returns.append(ep_return)
            lengths.append(ep_len)
            successes.append(1.0 if reached_goal else 0.0)
            for k in stage_keys:
                if k in info:
                    saw_stages = True
                stage_hits[k].append(1.0 if info.get(k) else 0.0)

        out: dict[str, float] = {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_length": float(np.mean(lengths)),
            "success_rate": float(np.mean(successes)),
            "n_episodes": float(n_episodes),
        }
        if saw_stages:
            for k, vals in stage_hits.items():
                out[k] = float(np.mean(vals))
        return out

    def to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """Convert a single observation to a batched float32 tensor on device.

        Always float32 (MPS does not support float64). Contiguous copy avoids
        slow/strided host→device paths.
        """
        x = np.ascontiguousarray(obs)
        x = torch.from_numpy(x).to(device=self.device, dtype=torch.float32)
        if x.ndim == len(self.obs_shape):
            x = x.unsqueeze(0)
        return x

    # ------------------------------------------------------------------
    # Shared training-history / logging helpers.
    # Used by the on-policy agents (REINFORCE, PPO) so their history dicts and
    # printed logs match DQN's, which keeps its own equivalents.
    # ------------------------------------------------------------------
    STAGE_KEYS: tuple[str, ...] = (
        "stage_key", "stage_door", "stage_right", "stage_water", "stage_lava", "stage_goal",
    )

    def _new_history(self) -> dict[str, list[float]]:
        keys = ("episode_return", "episode_length", "episode_success", "loss", "epsilon", "steps")
        history: dict[str, list[float]] = {k: [] for k in keys}
        for sk in self.STAGE_KEYS:
            history[sk] = []
        return history

    @staticmethod
    def _info_at(infos: Any, i: int) -> dict:
        """Pull the i-th env's info dict from a (Sync)VectorEnv info payload."""
        if isinstance(infos, (list, tuple)):
            return dict(infos[i] or {})
        if not isinstance(infos, dict):
            return {}
        if "final_info" in infos:  # gymnasium>=0.26 uses final_info for done envs
            fi = infos["final_info"]
            if fi is not None and i < len(fi) and fi[i] is not None:
                return dict(fi[i])
        out: dict = {}
        for k, v in infos.items():
            if k in ("final_observation", "final_info", "_final_observation", "_final_info"):
                continue
            try:
                out[k] = v[i]
            except Exception:
                continue
        return out

    @staticmethod
    def _success_flag(info: dict, terminated: bool) -> float:
        if "success" in info:
            return 1.0 if info["success"] else 0.0
        return 1.0 if terminated else 0.0

    def _record_episode(
        self, history, recent_returns, recent_success, recent_stages,
        ep_return, ep_len, success, info, log_every_episodes, saw_stages, note: str = "",
    ) -> bool:
        """Append one finished episode to ``history`` and print a periodic log line."""
        recent_returns.append(ep_return)
        recent_success.append(success)
        history["episode_return"].append(float(ep_return))
        history["episode_length"].append(float(ep_len))
        history["episode_success"].append(float(success))
        history["steps"].append(float(self.total_steps))
        for sk in self.STAGE_KEYS:
            val = 1.0 if info.get(sk) else 0.0
            if sk in info:
                saw_stages = True
            history[sk].append(val)
            recent_stages[sk].append(val)
        if log_every_episodes and self.total_episodes % max(1, log_every_episodes) == 0:
            mean_r = float(np.mean(recent_returns)) if recent_returns else 0.0
            succ = float(np.mean(recent_success)) if recent_success else 0.0
            line = (
                f"steps={self.total_steps:>8d}  episodes={self.total_episodes:>5d}  "
                f"return={ep_return:7.2f}  mean20={mean_r:7.2f}  succ20={succ:5.1%}"
            )
            if note:
                line += f"  {note}"
            if saw_stages:
                short = {
                    "stage_key": "key", "stage_door": "door", "stage_right": "right",
                    "stage_water": "water", "stage_lava": "lava", "stage_goal": "goal",
                }
                parts = [
                    f"{lbl}={(float(np.mean(recent_stages[sk])) if recent_stages[sk] else 0.0):4.0%}"
                    for sk, lbl in short.items()
                ]
                line += "  " + " ".join(parts)
            print(line, flush=True)
        return saw_stages

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"obs_shape={self.obs_shape}, n_actions={self.n_actions}, "
            f"device={self.device}, steps={self.total_steps})"
        )
