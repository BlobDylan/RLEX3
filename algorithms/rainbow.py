"""Rainbow DQN — Double + Dueling + Noisy + n-step + PER + C51 (Hessel et al.).

Built for the same MiniGrid image stack as ``DQN``. Prefer this when rare
events (door/water) need better credit assignment and exploration than ε-greedy.
"""

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
from .dqn import DQN, QNetwork


# ---------------------------------------------------------------------------
# Noisy linear (factorised Gaussian noise)
# ---------------------------------------------------------------------------

class NoisyLinear(nn.Module):
    """Linear layer with learnable factorised noise (Fortunato et al.)."""

    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self.register_buffer("weight_eps", torch.empty(out_features, in_features))
        self.register_buffer("bias_eps", torch.empty(out_features))

        mu_range = 1.0 / np.sqrt(in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(sigma0 / np.sqrt(in_features))
        self.bias_sigma.data.fill_(sigma0 / np.sqrt(out_features))
        self.reset_noise()

    @staticmethod
    def _scale_noise(size: int, device: torch.device) -> torch.Tensor:
        x = torch.randn(size, device=device)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        eps_in = self._scale_noise(self.in_features, self.weight_mu.device)
        eps_out = self._scale_noise(self.out_features, self.weight_mu.device)
        self.weight_eps.copy_(eps_out.ger(eps_in))
        self.bias_eps.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_eps
            bias = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


# ---------------------------------------------------------------------------
# Prioritized replay
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """Proportional PER (Schaul et al.) over the same transition layout as DQN."""

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        *,
        alpha: float = 0.6,
        eps: float = 1e-6,
    ) -> None:
        self.capacity = int(capacity)
        self.obs_shape = tuple(obs_shape)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.ptr = 0
        self.size = 0
        self.max_priority = 1.0

        self.obs = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity,), dtype=np.int64)
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.terminateds = np.empty((capacity,), dtype=np.float32)
        # Bootstrap discount γ^k for this transition's n-step length k.
        self.boot_gamma = np.empty((capacity,), dtype=np.float32)
        self.priorities = np.zeros((capacity,), dtype=np.float64)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        boot_gamma: float = 1.0,
    ) -> None:
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.terminateds[self.ptr] = float(terminated)
        self.boot_gamma[self.ptr] = float(boot_gamma)
        self.priorities[self.ptr] = self.max_priority
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int, device: torch.device, *, beta: float
    ) -> tuple[torch.Tensor, ...]:
        beta = float(np.clip(beta, 0.0, 1.0))
        prios = self.priorities[: self.size] ** self.alpha
        probs = prios / prios.sum()
        idx = np.random.choice(self.size, size=batch_size, p=probs)
        weights = (self.size * probs[idx]) ** (-beta)
        weights = weights / weights.max()

        non_blocking = device.type == "cuda"

        def _to(arr: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
            t = torch.from_numpy(np.ascontiguousarray(arr))
            if non_blocking:
                t = t.pin_memory()
            return t.to(device=device, dtype=dtype, non_blocking=non_blocking)

        return (
            _to(self.obs[idx], torch.float32),
            _to(self.actions[idx], torch.int64),
            _to(self.rewards[idx], torch.float32),
            _to(self.next_obs[idx], torch.float32),
            _to(self.terminateds[idx], torch.float32),
            _to(self.boot_gamma[idx], torch.float32),
            _to(weights.astype(np.float32), torch.float32),
            torch.from_numpy(idx.astype(np.int64)).to(device),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        prios = np.abs(td_errors) + self.eps
        self.priorities[indices] = prios
        self.max_priority = max(self.max_priority, float(prios.max()))


# ---------------------------------------------------------------------------
# Rainbow Q-network (dueling + noisy + distributional)
# ---------------------------------------------------------------------------

class RainbowQNetwork(nn.Module):
    """CNN trunk + dueling noisy heads over a categorical return distribution."""

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_actions: int,
        *,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 50.0,
        width_mult: int = 2,
        n_extra_conv: int = 0,
        fc_mult: int = 1,
        noisy: bool = True,
    ) -> None:
        super().__init__()
        self.n_actions = int(n_actions)
        self.n_atoms = int(n_atoms)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.noisy = bool(noisy)
        self.width_mult = max(1, int(width_mult))
        self.n_extra_conv = max(0, int(n_extra_conv))
        self.fc_mult = max(1, int(fc_mult))

        # Reuse the same CNN trunk construction as DQN via a throwaway QNetwork.
        trunk = QNetwork(
            obs_shape,
            n_actions,
            width_mult=self.width_mult,
            n_extra_conv=self.n_extra_conv,
            fc_mult=self.fc_mult,
        )
        self.features = trunk.features
        self.feat_hw = trunk.feat_hw
        self.channels_last = trunk.channels_last
        c3 = 64 * self.width_mult
        feat_dim = c3 * self.feat_hw[0] * self.feat_hw[1]
        hdim = 256 * self.width_mult * self.fc_mult

        Linear = NoisyLinear if self.noisy else nn.Linear
        self.value = nn.Sequential(
            Linear(feat_dim, hdim),
            nn.ReLU(),
            Linear(hdim, self.n_atoms),
        )
        self.advantage = nn.Sequential(
            Linear(feat_dim, hdim),
            nn.ReLU(),
            Linear(hdim, self.n_actions * self.n_atoms),
        )

        support = torch.linspace(self.v_min, self.v_max, self.n_atoms)
        self.register_buffer("support", support)

    def _nchw(self, x: torch.Tensor) -> torch.Tensor:
        if self.channels_last:
            x = x.permute(0, 3, 1, 2)
        return x / 255.0

    def reset_noise(self) -> None:
        if not self.noisy:
            return
        for mod in self.modules():
            if isinstance(mod, NoisyLinear):
                mod.reset_noise()

    def forward_atoms(self, x: torch.Tensor) -> torch.Tensor:
        """Return log-probabilities over atoms: ``(B, n_actions, n_atoms)``."""
        z = self.features(self._nchw(x))
        z = F.interpolate(z, size=self.feat_hw, mode="bilinear", align_corners=False)
        z = torch.flatten(z, 1)
        v = self.value(z).view(-1, 1, self.n_atoms)
        a = self.advantage(z).view(-1, self.n_actions, self.n_atoms)
        q_atoms = v + a - a.mean(dim=1, keepdim=True)
        return F.log_softmax(q_atoms, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expected Q-values ``(B, n_actions)`` (for greedy / Double DQN)."""
        log_prob = self.forward_atoms(x)
        return torch.exp(log_prob).mul(self.support).sum(dim=-1)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class RainbowDQN(DQN):
    """Rainbow DQN agent (inherits DQN train loop / logging / diagnose)."""

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_actions: int,
        device: str | torch.device | None = None,
        seed: int | None = None,
        *,
        gamma: float = 0.95,
        lr: float = 1e-4,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        learning_starts: int = 2_000,
        train_freq: int = 2,
        gradient_steps: int = 1,
        target_update_freq: int = 500,
        tau: float = 1.0,
        n_step: int = 3,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 50.0,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_steps: int = 100_000,
        noisy: bool = True,
        eps_start: float = 1.0,
        eps_end: float = 0.0,
        eps_decay_steps: int = 1,  # unused when noisy=True
        log_loss_every: int = 100,
        width_mult: int = 2,
        n_extra_conv: int = 0,
        fc_mult: int = 1,
        **_: Any,
    ) -> None:
        # Skip DQN.__init__ net/buffer setup; call BaseAlgorithm then configure.
        BaseAlgorithm.__init__(self, obs_shape, n_actions, device=device, seed=seed)

        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.learning_starts = int(learning_starts)
        self.train_freq = int(train_freq)
        self.gradient_steps = max(1, int(gradient_steps))
        self.target_update_freq = int(target_update_freq)
        self.tau = float(tau)
        self.double_dqn = True
        self.n_step = max(1, int(n_step))
        self.n_atoms = int(n_atoms)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.per_beta_start = float(per_beta_start)
        self.per_beta_steps = max(1, int(per_beta_steps))
        self.noisy = bool(noisy)
        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)
        self.eps_decay_steps = int(eps_decay_steps)
        self.log_loss_every = max(0, int(log_loss_every))
        self.width_mult = max(1, int(width_mult))
        self.n_extra_conv = max(0, int(n_extra_conv))
        self.fc_mult = max(1, int(fc_mult))

        arch = dict(
            n_atoms=self.n_atoms,
            v_min=self.v_min,
            v_max=self.v_max,
            width_mult=self.width_mult,
            n_extra_conv=self.n_extra_conv,
            fc_mult=self.fc_mult,
            noisy=self.noisy,
        )
        self.q_net = RainbowQNetwork(obs_shape, n_actions, **arch).to(self.device)
        self.target_net = RainbowQNetwork(obs_shape, n_actions, **arch).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = Adam(self.q_net.parameters(), lr=lr)
        self.buffer = PrioritizedReplayBuffer(
            buffer_size, obs_shape, alpha=per_alpha
        )
        self._n_step_queue: deque[tuple] = deque(maxlen=self.n_step)
        self._updates = 0
        self.delta_z = (self.v_max - self.v_min) / max(1, self.n_atoms - 1)

    def per_beta(self) -> float:
        t = min(1.0, self.total_steps / self.per_beta_steps)
        return self.per_beta_start + t * (1.0 - self.per_beta_start)

    def epsilon(self) -> float:
        """Noisy nets replace ε-greedy; report 0 so logs stay meaningful."""
        if self.noisy:
            return 0.0
        return super().epsilon()

    def reset_noise(self) -> None:
        self.q_net.reset_noise()
        self.target_net.reset_noise()

    def select_action(self, obs: np.ndarray, *, explore: bool = True) -> int:
        if self.noisy and explore:
            self.q_net.reset_noise()
        elif (not self.noisy) and explore and np.random.rand() < self.epsilon():
            return int(np.random.randint(0, self.n_actions))
        with torch.no_grad():
            # eval() disables noise sampling → mean weights for greedy diagnose
            was_training = self.q_net.training
            if not explore:
                self.q_net.eval()
            else:
                self.q_net.train()
            q = self.q_net(self.to_tensor(obs))
            action = int(q.argmax(dim=1).item())
            self.q_net.train(was_training)
            return action

    def select_actions(self, obss: np.ndarray, *, explore: bool = True) -> np.ndarray:
        n = int(obss.shape[0])
        if self.noisy and explore:
            self.q_net.reset_noise()
            with torch.no_grad():
                self.q_net.train()
                x = torch.from_numpy(np.ascontiguousarray(obss)).to(
                    device=self.device, dtype=torch.float32
                )
                return self.q_net(x).argmax(dim=1).cpu().numpy().astype(np.int64)

        if not explore:
            with torch.no_grad():
                self.q_net.eval()
                x = torch.from_numpy(np.ascontiguousarray(obss)).to(
                    device=self.device, dtype=torch.float32
                )
                out = self.q_net(x).argmax(dim=1).cpu().numpy().astype(np.int64)
                self.q_net.train()
                return out
        return super().select_actions(obss, explore=True)

    # ------------------------------------------------------------------
    # n-step storage
    # ------------------------------------------------------------------

    def _emit_n_step(self) -> None:
        """Emit one n-step (or shorter, at episode end) transition from the queue head."""
        assert self._n_step_queue
        R = 0.0
        terminated_any = False
        next_obs = self._n_step_queue[0][3]
        k = 0
        for i, (_o, _a, r, no, term) in enumerate(self._n_step_queue):
            R += (self.gamma**i) * float(r)
            next_obs = no
            k = i + 1
            if term:
                terminated_any = True
                break
        obs0, a0, _, _, _ = self._n_step_queue[0]
        boot = 0.0 if terminated_any else (self.gamma**k)
        self.buffer.add(
            obs0, int(a0), float(R), next_obs, bool(terminated_any), boot_gamma=boot
        )
        self._n_step_queue.popleft()

    def _store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self._n_step_queue.append(
            (np.array(obs, copy=True), int(action), float(reward), np.array(next_obs, copy=True), bool(terminated))
        )
        episode_done = bool(terminated or truncated)
        if episode_done:
            while self._n_step_queue:
                self._emit_n_step()
        elif len(self._n_step_queue) >= self.n_step:
            self._emit_n_step()

    # ------------------------------------------------------------------
    # Learning (C51 + Double + PER)
    # ------------------------------------------------------------------

    def update(self, *, return_loss: bool | None = None) -> float | None:
        if len(self.buffer) < max(self.batch_size, self.learning_starts):
            return None

        (
            obs,
            actions,
            rewards,
            next_obs,
            terminateds,
            boot_gamma,
            weights,
            indices,
        ) = self.buffer.sample(self.batch_size, self.device, beta=self.per_beta())

        self.q_net.reset_noise()
        self.target_net.reset_noise()

        with torch.no_grad():
            # Double DQN action selection on online net
            next_q = self.q_net(next_obs)
            next_actions = next_q.argmax(dim=1)
            next_log_prob = self.target_net.forward_atoms(next_obs)
            next_dist = torch.exp(
                next_log_prob[torch.arange(self.batch_size, device=self.device), next_actions]
            )  # (B, n_atoms)

            # `rewards` already hold n-step returns; bootstrap with stored γ^k
            support = self.q_net.support
            Tz = rewards.unsqueeze(1) + boot_gamma.unsqueeze(1) * support
            Tz = Tz.clamp(self.v_min, self.v_max)
            b = (Tz - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            # Avoid zero mass when b lands exactly on an atom (l == u).
            l[(u > 0) & (l == u)] -= 1
            u[(l < (self.n_atoms - 1)) & (l == u)] += 1
            l = l.clamp(0, self.n_atoms - 1)
            u = u.clamp(0, self.n_atoms - 1)

            target = torch.zeros_like(next_dist)
            offset = (
                torch.arange(self.batch_size, device=self.device)
                .unsqueeze(1)
                .expand(self.batch_size, self.n_atoms)
                * self.n_atoms
            )
            target.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            target.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        log_prob = self.q_net.forward_atoms(obs)
        log_pa = log_prob[torch.arange(self.batch_size, device=self.device), actions]
        # Cross-entropy; TD error proxy = KL for PER
        per_sample = -(target * log_pa).sum(dim=1)
        loss = (per_sample * weights).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        td = per_sample.detach().abs().cpu().numpy()
        self.buffer.update_priorities(indices.cpu().numpy(), td)

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
        # n-step + PER path is single-env for now (vec would need per-env queues).
        if n_envs != 1:
            print("RainbowDQN: n_envs>1 not supported yet — using n_envs=1")
        if log_every_episodes is None:
            if log_every is None:
                log_every_episodes = 10
            elif int(log_every) >= 50:
                log_every_episodes = max(1, int(log_every) // 50)
            else:
                log_every_episodes = max(1, int(log_every))
        else:
            log_every_episodes = max(1, int(log_every_episodes))
        return self._train_single(
            env_fn,
            total_timesteps=total_timesteps,
            log_every_episodes=log_every_episodes,
        )

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
        self._n_step_queue.clear()

        while self.total_steps < total_timesteps:
            action = self.select_action(obs, explore=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_done = bool(terminated or truncated)
            success = self._success_from_info(info, terminated)

            self._store_transition(
                obs, action, float(reward), next_obs, bool(terminated), bool(truncated)
            )
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
                "width_mult": self.width_mult,
                "n_extra_conv": self.n_extra_conv,
                "fc_mult": self.fc_mult,
                "n_step": self.n_step,
                "n_atoms": self.n_atoms,
                "v_min": self.v_min,
                "v_max": self.v_max,
                "noisy": self.noisy,
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

    def __repr__(self) -> str:
        return (
            f"RainbowDQN(obs_shape={self.obs_shape}, n_actions={self.n_actions}, "
            f"device={self.device}, width_mult={self.width_mult}, "
            f"n_step={self.n_step}, atoms={self.n_atoms}, noisy={self.noisy}, "
            f"params={self.n_parameters():,})"
        )
