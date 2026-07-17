"""Deep Q-Network (DQN) — value-based baseline for MiniGrid image observations."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from .base import BaseAlgorithm


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-size circular buffer of (s, a, r, s', done) transitions."""

    def __init__(self, capacity: int, obs_shape: tuple[int, ...]) -> None:
        self.capacity = int(capacity)
        self.obs_shape = tuple(obs_shape)
        self.ptr = 0
        self.size = 0

        self.obs = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity,), dtype=np.int64)
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.dones = np.empty((capacity,), dtype=np.float32)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=batch_size)
        obs = torch.as_tensor(self.obs[idx], dtype=torch.float32, device=device)
        next_obs = torch.as_tensor(self.next_obs[idx], dtype=torch.float32, device=device)
        actions = torch.as_tensor(self.actions[idx], dtype=torch.int64, device=device)
        rewards = torch.as_tensor(self.rewards[idx], dtype=torch.float32, device=device)
        dones = torch.as_tensor(self.dones[idx], dtype=torch.float32, device=device)
        return obs, actions, rewards, next_obs, dones


# ---------------------------------------------------------------------------
# Q-network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Nature-style CNN that accepts HWC or CHW image observations."""

    def __init__(self, obs_shape: tuple[int, ...], n_actions: int) -> None:
        super().__init__()
        self.obs_shape = tuple(obs_shape)
        self.channels_last = self._is_channels_last(obs_shape)
        c = obs_shape[-1] if self.channels_last else obs_shape[0]

        self.features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    @staticmethod
    def _is_channels_last(shape: tuple[int, ...]) -> bool:
        # MiniGrid / GrayscaleWrapper emit (H, W, C) with C in {1, 3}.
        return len(shape) == 3 and shape[-1] in (1, 3) and shape[0] > 3

    def _nchw(self, x: torch.Tensor) -> torch.Tensor:
        if self.channels_last:
            x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return x / 255.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(self._nchw(x)))


# ---------------------------------------------------------------------------
# DQN agent
# ---------------------------------------------------------------------------

class DQN(BaseAlgorithm):
    """Classic DQN with experience replay and a frozen target network."""

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_actions: int,
        device: str | torch.device = "cpu",
        seed: int | None = None,
        *,
        gamma: float = 0.99,
        lr: float = 1e-4,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        learning_starts: int = 1_000,
        train_freq: int = 4,
        target_update_freq: int = 1_000,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 50_000,
    ) -> None:
        super().__init__(obs_shape, n_actions, device=device, seed=seed)

        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.target_update_freq = target_update_freq
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps

        self.q_net = QNetwork(obs_shape, n_actions).to(self.device)
        self.target_net = QNetwork(obs_shape, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, obs_shape)

        self._updates = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def epsilon(self) -> float:
        """Linearly decayed epsilon for the current ``total_steps``."""
        if self.eps_decay_steps <= 0:
            return self.eps_end
        t = min(1.0, self.total_steps / self.eps_decay_steps)
        return self.eps_start + t * (self.eps_end - self.eps_start)

    def select_action(self, obs: np.ndarray, *, explore: bool = True) -> int:
        if explore and np.random.rand() < self.epsilon():
            return int(np.random.randint(0, self.n_actions))

        with torch.no_grad():
            q = self.q_net(self.to_tensor(obs))
            return int(q.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self) -> float | None:
        """One gradient step on a minibatch. Returns loss, or None if skipped."""
        if len(self.buffer) < max(self.batch_size, self.learning_starts):
            return None

        obs, actions, rewards, next_obs, dones = self.buffer.sample(
            self.batch_size, self.device
        )

        with torch.no_grad():
            next_q = self.target_net(next_obs).max(dim=1).values
            targets = rewards + (1.0 - dones) * self.gamma * next_q

        q_values = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self._updates += 1
        if self._updates % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

    def train(
        self,
        env_fn: Callable[[], Any],
        *,
        total_timesteps: int,
        log_every: int = 1_000,
        **kwargs: Any,
    ) -> dict[str, list[float]]:
        """Interact with ``env_fn()``, store transitions, and update Q."""
        env = env_fn()
        history: dict[str, list[float]] = {
            "episode_return": [],
            "episode_length": [],
            "loss": [],
            "epsilon": [],
            "steps": [],
        }

        obs, _ = env.reset(seed=self.seed)
        ep_return = 0.0
        ep_len = 0
        recent_returns: deque[float] = deque(maxlen=20)

        while self.total_steps < total_timesteps:
            action = self.select_action(obs, explore=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            self.buffer.add(obs, action, float(reward), next_obs, done)
            obs = next_obs
            ep_return += float(reward)
            ep_len += 1
            self.total_steps += 1

            loss = None
            if self.total_steps >= self.learning_starts and self.total_steps % self.train_freq == 0:
                loss = self.update()
                if loss is not None:
                    history["loss"].append(loss)

            if done:
                self.total_episodes += 1
                recent_returns.append(ep_return)
                history["episode_return"].append(ep_return)
                history["episode_length"].append(float(ep_len))
                history["epsilon"].append(self.epsilon())
                history["steps"].append(float(self.total_steps))

                if log_every and self.total_episodes % max(1, log_every // 50) == 0:
                    mean_r = float(np.mean(recent_returns)) if recent_returns else 0.0
                    print(
                        f"steps={self.total_steps:>7d}  "
                        f"episodes={self.total_episodes:>5d}  "
                        f"return={ep_return:7.2f}  "
                        f"mean20={mean_r:7.2f}  "
                        f"eps={self.epsilon():.3f}"
                    )

                obs, _ = env.reset()
                ep_return = 0.0
                ep_len = 0

        env.close()
        return history

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
                "total_episodes": self.total_episodes,
                "updates": self._updates,
                "obs_shape": self.obs_shape,
                "n_actions": self.n_actions,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = int(ckpt.get("total_steps", 0))
        self.total_episodes = int(ckpt.get("total_episodes", 0))
        self._updates = int(ckpt.get("updates", 0))
