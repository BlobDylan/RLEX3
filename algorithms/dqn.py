"""Deep Q-Network (DQN) — value-based baseline for MiniGrid image observations."""

from __future__ import annotations

from collections import Counter, deque
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
    """Fixed-size circular buffer of (s, a, r, s', terminated) transitions."""

    def __init__(self, capacity: int, obs_shape: tuple[int, ...]) -> None:
        self.capacity = int(capacity)
        self.obs_shape = tuple(obs_shape)
        self.ptr = 0
        self.size = 0

        self.obs = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity,), dtype=np.int64)
        self.rewards = np.empty((capacity,), dtype=np.float32)
        # Bootstrapping mask: True only on real terminals (goal), NOT timeouts.
        self.terminateds = np.empty((capacity,), dtype=np.float32)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
    ) -> None:
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.terminateds[self.ptr] = float(terminated)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=batch_size)
        # Contiguous float32/int64 on device — MPS-friendly (no float64).
        obs = torch.from_numpy(np.ascontiguousarray(self.obs[idx])).to(
            device=device, dtype=torch.float32
        )
        next_obs = torch.from_numpy(np.ascontiguousarray(self.next_obs[idx])).to(
            device=device, dtype=torch.float32
        )
        actions = torch.from_numpy(np.ascontiguousarray(self.actions[idx])).to(
            device=device, dtype=torch.int64
        )
        rewards = torch.from_numpy(np.ascontiguousarray(self.rewards[idx])).to(
            device=device, dtype=torch.float32
        )
        terminateds = torch.from_numpy(
            np.ascontiguousarray(self.terminateds[idx])
        ).to(device=device, dtype=torch.float32)
        return obs, actions, rewards, next_obs, terminateds


# ---------------------------------------------------------------------------
# Q-network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Small CNN for MiniGrid frames (HWC or CHW). Gentler than Nature-Atari strides."""

    def __init__(self, obs_shape: tuple[int, ...], n_actions: int) -> None:
        super().__init__()
        self.obs_shape = tuple(obs_shape)
        self.channels_last = self._is_channels_last(obs_shape)
        c = obs_shape[-1] if self.channels_last else obs_shape[0]

        # Strided convs; then bilinear resize to a fixed map (MPS-safe).
        # AdaptiveAvgPool2d(H_out) on MPS requires H % H_out == 0 — fails on
        # cropped 32×32 → … → 8×8 pooled to 5×5. Interpolate avoids that and
        # keeps a spatial feature map (unlike global pool).
        self.features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.feat_hw = (5, 5)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * self.feat_hw[0] * self.feat_hw[1], 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
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
        x = self.features(self._nchw(x))
        x = F.interpolate(x, size=self.feat_hw, mode="bilinear", align_corners=False)
        return self.head(x)

# ---------------------------------------------------------------------------
# DQN agent
# ---------------------------------------------------------------------------

class DQN(BaseAlgorithm):
    """DQN / Double DQN with replay and a target network (soft or hard updates)."""

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_actions: int,
        device: str | torch.device | None = None,
        seed: int | None = None,
        *,
        gamma: float = 0.99,
        lr: float = 2.5e-4,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        learning_starts: int = 1_000,
        train_freq: int = 1,
        target_update_freq: int = 1_000,
        tau: float = 0.005,
        double_dqn: bool = True,
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
        self.tau = float(tau)
        self.double_dqn = bool(double_dqn)
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

    def diagnose_greedy(
        self,
        env,
        *,
        max_steps: int = 100,
        seed: int | None = 0,
    ) -> dict[str, Any]:
        """Roll out one greedy episode and summarize the action distribution."""
        obs, _ = env.reset(seed=seed)
        q0 = self.q_values(obs)
        actions: list[int] = []
        total_reward = 0.0
        terminated = truncated = False

        for _ in range(max_steps):
            action = self.select_action(obs, explore=False)
            actions.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        counts = Counter(actions)
        hist = {a: counts.get(a, 0) for a in range(self.n_actions)}
        return {
            "q_values_start": q0.tolist(),
            "actions": actions[:40],  # preview
            "action_counts": hist,
            "steps": len(actions),
            "return": total_reward,
            "success": bool(terminated),
        }

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        """Return Q(s, ·) for a single observation (numpy)."""
        with torch.no_grad():
            q = self.q_net(self.to_tensor(obs))
            return q.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def _sync_target(self) -> None:
        if self.tau >= 1.0:
            if self._updates % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())
            return
        with torch.no_grad():
            for tp, p in zip(self.target_net.parameters(), self.q_net.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

    def update(self) -> float | None:
        """One gradient step on a minibatch. Returns loss, or None if skipped."""
        if len(self.buffer) < max(self.batch_size, self.learning_starts):
            return None

        obs, actions, rewards, next_obs, terminateds = self.buffer.sample(
            self.batch_size, self.device
        )

        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.q_net(next_obs).argmax(dim=1)
                next_q = self.target_net(next_obs).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                next_q = self.target_net(next_obs).max(dim=1).values
            targets = rewards + (1.0 - terminateds) * self.gamma * next_q

        q_values = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self._updates += 1
        self._sync_target()

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
            "episode_success": [],
            "loss": [],
            "epsilon": [],
            "steps": [],
        }

        obs, _ = env.reset(seed=self.seed)
        ep_return = 0.0
        ep_len = 0
        recent_returns: deque[float] = deque(maxlen=20)
        recent_success: deque[float] = deque(maxlen=20)

        while self.total_steps < total_timesteps:
            action = self.select_action(obs, explore=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            episode_done = bool(terminated or truncated)
            success = 1.0 if terminated else 0.0

            # Bootstrap only on true terminals (goal), not on max_steps truncations.
            self.buffer.add(obs, action, float(reward), next_obs, bool(terminated))
            obs = next_obs
            ep_return += float(reward)
            ep_len += 1
            self.total_steps += 1

            if self.total_steps >= self.learning_starts and self.total_steps % self.train_freq == 0:
                loss = self.update()
                if loss is not None:
                    history["loss"].append(loss)

            if episode_done:
                self.total_episodes += 1
                recent_returns.append(ep_return)
                recent_success.append(success)
                history["episode_return"].append(ep_return)
                history["episode_length"].append(float(ep_len))
                history["episode_success"].append(success)
                history["epsilon"].append(self.epsilon())
                history["steps"].append(float(self.total_steps))

                if log_every and self.total_episodes % max(1, log_every // 50) == 0:
                    mean_r = float(np.mean(recent_returns)) if recent_returns else 0.0
                    succ20 = float(np.mean(recent_success)) if recent_success else 0.0
                    print(
                        f"steps={self.total_steps:>7d}  "
                        f"episodes={self.total_episodes:>5d}  "
                        f"return={ep_return:7.2f}  "
                        f"mean20={mean_r:7.2f}  "
                        f"succ20={succ20:5.1%}  "
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
                "double_dqn": self.double_dqn,
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
