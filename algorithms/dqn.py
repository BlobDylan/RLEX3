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
        # CUDA: pin_memory + non_blocking overlaps H2D with compute when possible.
        non_blocking = device.type == "cuda"

        def _to(arr: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
            t = torch.from_numpy(np.ascontiguousarray(arr))
            if non_blocking:
                t = t.pin_memory()
            return t.to(device=device, dtype=dtype, non_blocking=non_blocking)

        obs = _to(self.obs[idx], torch.float32)
        next_obs = _to(self.next_obs[idx], torch.float32)
        actions = _to(self.actions[idx], torch.int64)
        rewards = _to(self.rewards[idx], torch.float32)
        terminateds = _to(self.terminateds[idx], torch.float32)
        return obs, actions, rewards, next_obs, terminateds


# ---------------------------------------------------------------------------
# Q-network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """CNN for MiniGrid frames (HWC or CHW). Gentler than Nature-Atari strides.

    ``width_mult`` scales conv/head channels (1 ≈ 0.47M params; params ~ width²).
    ``n_extra_conv`` adds that many extra 3×3 stride-1 layers at the top width.
    ``fc_mult`` scales only the MLP head width (on top of ``width_mult``).
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_actions: int,
        *,
        width_mult: int = 1,
        n_extra_conv: int = 0,
        fc_mult: int = 1,
    ) -> None:
        super().__init__()
        self.obs_shape = tuple(obs_shape)
        self.width_mult = max(1, int(width_mult))
        self.n_extra_conv = max(0, int(n_extra_conv))
        self.fc_mult = max(1, int(fc_mult))
        self.channels_last = self._is_channels_last(obs_shape)
        c = obs_shape[-1] if self.channels_last else obs_shape[0]

        w = self.width_mult
        c1, c2, c3 = 32 * w, 64 * w, 64 * w
        hdim = 256 * w * self.fc_mult

        # Strided convs; then bilinear resize to a fixed map (MPS-safe).
        # AdaptiveAvgPool2d(H_out) on MPS requires H % H_out == 0 — fails on
        # cropped 32×32 → … → 8×8 pooled to 5×5. Interpolate avoids that and
        # keeps a spatial feature map (unlike global pool).
        feat_layers: list[nn.Module] = [
            nn.Conv2d(c, c1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ]
        for _ in range(self.n_extra_conv):
            feat_layers.extend(
                [
                    nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                ]
            )
        self.features = nn.Sequential(*feat_layers)
        self.feat_hw = (5, 5)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * self.feat_hw[0] * self.feat_hw[1], hdim),
            nn.ReLU(),
            nn.Linear(hdim, n_actions),
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
        gradient_steps: int = 1,
        target_update_freq: int = 1_000,
        tau: float = 0.005,
        double_dqn: bool = True,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 50_000,
        log_loss_every: int = 50,
        width_mult: int = 1,
        n_extra_conv: int = 0,
        fc_mult: int = 1,
    ) -> None:
        super().__init__(obs_shape, n_actions, device=device, seed=seed)

        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.gradient_steps = max(1, int(gradient_steps))
        self.target_update_freq = target_update_freq
        self.tau = float(tau)
        self.double_dqn = bool(double_dqn)
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        # loss.item() forces a device sync — only do it every N updates.
        self.log_loss_every = max(0, int(log_loss_every))
        self.width_mult = max(1, int(width_mult))
        self.n_extra_conv = max(0, int(n_extra_conv))
        self.fc_mult = max(1, int(fc_mult))

        _arch = dict(
            width_mult=self.width_mult,
            n_extra_conv=self.n_extra_conv,
            fc_mult=self.fc_mult,
        )
        self.q_net = QNetwork(obs_shape, n_actions, **_arch).to(self.device)
        self.target_net = QNetwork(obs_shape, n_actions, **_arch).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, obs_shape)

        self._updates = 0

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.q_net.parameters())

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

    def select_actions(self, obss: np.ndarray, *, explore: bool = True) -> np.ndarray:
        """Batched ε-greedy — one GPU forward for ``n_envs`` observations.

        Args:
            obss: Array shaped ``(n_envs, *obs_shape)``.
        """
        n = int(obss.shape[0])
        if explore:
            eps = self.epsilon()
            rand_mask = np.random.rand(n) < eps
        else:
            rand_mask = np.zeros(n, dtype=bool)

        actions = np.empty(n, dtype=np.int64)
        if rand_mask.any():
            actions[rand_mask] = np.random.randint(0, self.n_actions, size=int(rand_mask.sum()))

        need_q = ~rand_mask
        if need_q.any():
            with torch.no_grad():
                batch = np.ascontiguousarray(obss[need_q])
                x = torch.from_numpy(batch).to(device=self.device, dtype=torch.float32)
                greedy = self.q_net(x).argmax(dim=1).cpu().numpy()
            actions[need_q] = greedy
        return actions

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

    def update(self, *, return_loss: bool | None = None) -> float | None:
        """One gradient step on a minibatch. Returns loss, or None if skipped.

        ``return_loss`` defaults to logging every ``log_loss_every`` updates so we
        avoid a host↔device sync (``loss.item()``) on every step — critical on MPS.
        """
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

        if return_loss is None:
            if self.log_loss_every <= 0:
                return_loss = False
            else:
                return_loss = (self._updates % self.log_loss_every) == 0
        if return_loss:
            return float(loss.item())
        return None

    # ComplexEnv stage latches (from ComplexShapingWrapper info); ignored if absent.
    _STAGE_KEYS: tuple[str, ...] = (
        "stage_key",
        "stage_door",
        "stage_right",
        "stage_water",
        "stage_lava",
        "stage_goal",
    )

    def train(
        self,
        env_fn: Callable[[], Any],
        *,
        total_timesteps: int,
        log_every: int | None = None,
        log_every_episodes: int | None = None,
        n_envs: int = 1,
        **kwargs: Any,
    ) -> dict[str, list[float]]:
        """Interact with ``env_fn()``, store transitions, and update Q.

        Args:
            log_every_episodes: Print a progress line every N finished episodes
                (default **10**).
            log_every: Legacy alias. If ``log_every_episodes`` is omitted and
                ``log_every >= 50``, uses ``log_every // 50`` episodes; if
                ``log_every < 50``, treats it as an episode interval.
            n_envs: If >1, run ``SyncVectorEnv`` with batched action selection
                (much better GPU amortization). Episode metrics are still logged
                per finished sub-env episode.
        """
        if log_every_episodes is None:
            if log_every is None:
                log_every_episodes = 10
            elif int(log_every) >= 50:
                log_every_episodes = max(1, int(log_every) // 50)
            else:
                log_every_episodes = max(1, int(log_every))
        else:
            log_every_episodes = max(1, int(log_every_episodes))

        n_envs = max(1, int(n_envs))
        if n_envs == 1:
            return self._train_single(
                env_fn,
                total_timesteps=total_timesteps,
                log_every_episodes=log_every_episodes,
            )
        return self._train_vectorized(
            env_fn,
            total_timesteps=total_timesteps,
            log_every_episodes=log_every_episodes,
            n_envs=n_envs,
        )

    def _maybe_learn(self, history: dict[str, list[float]]) -> None:
        """Run ``gradient_steps`` updates when ``train_freq`` says so."""
        if self.total_steps < self.learning_starts:
            return
        if self.total_steps % self.train_freq != 0:
            return
        for _ in range(self.gradient_steps):
            loss = self.update()
            if loss is not None:
                history["loss"].append(loss)

    def _learn_after_rollout(
        self,
        history: dict[str, list[float]],
        *,
        steps_collected: int,
        steps_since_update: int,
    ) -> int:
        """Vec-env: one learn burst after enough env steps (not n_envs/train_freq bursts).

        SyncVectorEnv steps are already serial on CPU; firing multiple SGD updates
        per vector step (old while-loop) made ``n_envs>1`` *slower* on MPS.
        """
        steps_since_update += steps_collected
        if self.total_steps < self.learning_starts:
            return steps_since_update
        if steps_since_update >= self.train_freq:
            for _ in range(self.gradient_steps):
                loss = self.update()
                if loss is not None:
                    history["loss"].append(loss)
            steps_since_update = 0
        return steps_since_update

    def _train_single(
        self,
        env_fn: Callable[[], Any],
        *,
        total_timesteps: int,
        log_every_episodes: int,
    ) -> dict[str, list[float]]:
        env = env_fn()
        history = self._empty_history()
        obs, info = env.reset(seed=self.seed)
        ep_return = 0.0
        ep_len = 0
        recent_returns: deque[float] = deque(maxlen=20)
        recent_success: deque[float] = deque(maxlen=20)
        recent_stages: dict[str, deque[float]] = {
            sk: deque(maxlen=20) for sk in self._STAGE_KEYS
        }
        saw_stages = False

        while self.total_steps < total_timesteps:
            action = self.select_action(obs, explore=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_done = bool(terminated or truncated)
            success = self._success_from_info(info, terminated)

            self.buffer.add(obs, action, float(reward), next_obs, bool(terminated))
            obs = next_obs
            ep_return += float(reward)
            ep_len += 1
            self.total_steps += 1
            self._maybe_learn(history)

            if episode_done:
                self.total_episodes += 1
                saw_stages = self._log_episode(
                    history,
                    recent_returns,
                    recent_success,
                    recent_stages,
                    ep_return,
                    ep_len,
                    success,
                    info,
                    saw_stages,
                    log_every_episodes,
                )
                obs, info = env.reset()
                ep_return = 0.0
                ep_len = 0

        env.close()
        return history

    def _train_vectorized(
        self,
        env_fn: Callable[[], Any],
        *,
        total_timesteps: int,
        log_every_episodes: int,
        n_envs: int,
    ) -> dict[str, list[float]]:
        import gymnasium as gym

        envs = gym.vector.SyncVectorEnv([env_fn for _ in range(n_envs)])
        history = self._empty_history()
        seed = self.seed
        reset_seeds = None if seed is None else [seed + i for i in range(n_envs)]
        obss, infos = envs.reset(seed=reset_seeds)
        ep_returns = np.zeros(n_envs, dtype=np.float64)
        ep_lens = np.zeros(n_envs, dtype=np.int64)
        recent_returns: deque[float] = deque(maxlen=20)
        recent_success: deque[float] = deque(maxlen=20)
        recent_stages: dict[str, deque[float]] = {
            sk: deque(maxlen=20) for sk in self._STAGE_KEYS
        }
        saw_stages = False
        steps_since_update = 0

        while self.total_steps < total_timesteps:
            actions = self.select_actions(obss, explore=True)
            next_obss, rewards, terminateds, truncateds, infos = envs.step(actions)
            dones = np.logical_or(terminateds, truncateds)

            for i in range(n_envs):
                if self.total_steps >= total_timesteps:
                    break
                next_obs_i = next_obss[i]
                term_i = bool(terminateds[i])
                if dones[i]:
                    final = None
                    if isinstance(infos, dict) and "final_observation" in infos:
                        fo = infos["final_observation"]
                        if fo is not None and fo[i] is not None:
                            final = fo[i]
                    if final is not None:
                        next_obs_i = final
                self.buffer.add(
                    obss[i],
                    int(actions[i]),
                    float(rewards[i]),
                    next_obs_i,
                    term_i,
                )
                ep_returns[i] += float(rewards[i])
                ep_lens[i] += 1
                self.total_steps += 1

                if dones[i]:
                    info_i = self._info_at(infos, i)
                    success = self._success_from_info(info_i, term_i)
                    self.total_episodes += 1
                    saw_stages = self._log_episode(
                        history,
                        recent_returns,
                        recent_success,
                        recent_stages,
                        float(ep_returns[i]),
                        int(ep_lens[i]),
                        success,
                        info_i,
                        saw_stages,
                        log_every_episodes,
                    )
                    ep_returns[i] = 0.0
                    ep_lens[i] = 0

            steps_since_update = self._learn_after_rollout(
                history,
                steps_collected=n_envs,
                steps_since_update=steps_since_update,
            )
            obss = next_obss

        envs.close()
        return history

    def _empty_history(self) -> dict[str, list[float]]:
        history: dict[str, list[float]] = {
            "episode_return": [],
            "episode_length": [],
            "episode_success": [],
            "loss": [],
            "epsilon": [],
            "steps": [],
        }
        for sk in self._STAGE_KEYS:
            history[sk] = []
        return history

    @staticmethod
    def _success_from_info(info: dict, terminated: bool) -> float:
        if "success" in info:
            return 1.0 if info["success"] else 0.0
        return 1.0 if terminated else 0.0

    @staticmethod
    def _info_at(infos: Any, i: int) -> dict:
        """Pull the i-th env's info from a VectorEnv info payload."""
        if isinstance(infos, (list, tuple)):
            return dict(infos[i] or {})
        if not isinstance(infos, dict):
            return {}
        # gymnasium>=0.26 may use "final_info" for done envs
        if "final_info" in infos:
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

    def _log_episode(
        self,
        history: dict[str, list[float]],
        recent_returns: deque[float],
        recent_success: deque[float],
        recent_stages: dict[str, deque[float]],
        ep_return: float,
        ep_len: int,
        success: float,
        info: dict,
        saw_stages: bool,
        log_every_episodes: int,
    ) -> bool:
        recent_returns.append(ep_return)
        recent_success.append(success)
        history["episode_return"].append(ep_return)
        history["episode_length"].append(float(ep_len))
        history["episode_success"].append(success)
        history["epsilon"].append(self.epsilon())
        history["steps"].append(float(self.total_steps))

        for sk in self._STAGE_KEYS:
            val = 1.0 if info.get(sk) else 0.0
            if sk in info:
                saw_stages = True
            history[sk].append(val)
            recent_stages[sk].append(val)

        if log_every_episodes and self.total_episodes % max(1, log_every_episodes) == 0:
            mean_r = float(np.mean(recent_returns)) if recent_returns else 0.0
            succ20 = float(np.mean(recent_success)) if recent_success else 0.0
            line = (
                f"steps={self.total_steps:>7d}  "
                f"episodes={self.total_episodes:>5d}  "
                f"return={ep_return:7.2f}  "
                f"mean20={mean_r:7.2f}  "
                f"succ20={succ20:5.1%}  "
                f"eps={self.epsilon():.3f}"
            )
            if saw_stages:
                parts = []
                short = {
                    "stage_key": "key",
                    "stage_door": "door",
                    "stage_right": "right",
                    "stage_water": "water",
                    "stage_lava": "lava",
                    "stage_goal": "goal",
                }
                for sk, label in short.items():
                    rate = float(np.mean(recent_stages[sk])) if recent_stages[sk] else 0.0
                    parts.append(f"{label}={rate:4.0%}")
                line += "  " + " ".join(parts)
            print(line)
        return saw_stages

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
                "width_mult": self.width_mult,
                "n_extra_conv": self.n_extra_conv,
                "fc_mult": self.fc_mult,
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
        if "width_mult" in ckpt:
            self.width_mult = int(ckpt["width_mult"])
        if "n_extra_conv" in ckpt:
            self.n_extra_conv = int(ckpt["n_extra_conv"])
        if "fc_mult" in ckpt:
            self.fc_mult = int(ckpt["fc_mult"])

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"obs_shape={self.obs_shape}, n_actions={self.n_actions}, "
            f"device={self.device}, width_mult={self.width_mult}, "
            f"n_extra_conv={self.n_extra_conv}, fc_mult={self.fc_mult}, "
            f"params={self.n_parameters():,})"
        )