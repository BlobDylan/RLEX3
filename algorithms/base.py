"""Shared base class for value-based, policy-based, and actor-critic agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch


from .device import get_torch_device, seed_everything


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
        """Roll out ``n_episodes`` greedily and return mean / std return + success rate."""
        returns: list[float] = []
        lengths: list[int] = []
        successes: list[float] = []

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=None if seed is None else seed + ep)
            done = False
            ep_return = 0.0
            ep_len = 0
            reached_goal = False

            while not done:
                action = self.select_action(obs, explore=False)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_return += float(reward)
                ep_len += 1
                reached_goal = bool(terminated)
                done = terminated or truncated
                if max_steps is not None and ep_len >= max_steps:
                    break

            returns.append(ep_return)
            lengths.append(ep_len)
            successes.append(1.0 if reached_goal else 0.0)

        return {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_length": float(np.mean(lengths)),
            "success_rate": float(np.mean(successes)),
            "n_episodes": float(n_episodes),
        }

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

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"obs_shape={self.obs_shape}, n_actions={self.n_actions}, "
            f"device={self.device}, steps={self.total_steps})"
        )
