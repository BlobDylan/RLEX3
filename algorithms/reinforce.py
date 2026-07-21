"""REINFORCE (Monte-Carlo policy gradient) from scratch, for MiniGrid pixels.

Policy-based method: a CNN policy net outputs action logits; we sample during
training (exploration), act greedily at eval. Updates use complete-episode
discounted returns with a batch-mean/std baseline and an entropy bonus. Collection
is vectorised (``n_envs`` parallel envs) and an update fires once enough complete
episodes are gathered. Consistent surface with ``DQN`` (``select_action``, ``train``
returning a plottable history dict, ``save``/``load``).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from .base import BaseAlgorithm
from .networks import CNNEncoder


class PolicyNetwork(nn.Module):
    """CNN encoder -> action-logit head."""

    def __init__(self, obs_shape, n_actions, *, width_mult=1, n_extra_conv=0, fc_mult=1) -> None:
        super().__init__()
        self.encoder = CNNEncoder(
            obs_shape, width_mult=width_mult, n_extra_conv=n_extra_conv, fc_mult=fc_mult
        )
        self.pi = nn.Linear(self.encoder.out_dim, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pi(self.encoder(x))  # logits


class REINFORCE(BaseAlgorithm):
    """Monte-Carlo policy gradient with entropy regularisation."""

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_actions: int,
        device: str | torch.device | None = None,
        seed: int | None = None,
        *,
        gamma: float = 0.99,
        lr: float = 3e-4,
        entropy_coef: float = 0.05,
        normalize_returns: bool = True,
        episodes_per_update: int = 16,
        max_grad_norm: float = 5.0,
        width_mult: int = 1,
        n_extra_conv: int = 0,
        fc_mult: int = 1,
    ) -> None:
        super().__init__(obs_shape, n_actions, device=device, seed=seed)
        self.gamma = float(gamma)
        self.entropy_coef = float(entropy_coef)
        self.normalize_returns = bool(normalize_returns)
        self.episodes_per_update = max(1, int(episodes_per_update))
        self.max_grad_norm = float(max_grad_norm)

        self.policy = PolicyNetwork(
            obs_shape, n_actions, width_mult=width_mult, n_extra_conv=n_extra_conv, fc_mult=fc_mult
        ).to(self.device)
        self.optimizer = Adam(self.policy.parameters(), lr=lr)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.policy.parameters())

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------
    def select_action(self, obs: np.ndarray, *, explore: bool = True) -> int:
        with torch.no_grad():
            logits = self.policy(self.to_tensor(obs))
            if explore:
                a = torch.distributions.Categorical(logits=logits).sample()
            else:
                a = logits.argmax(dim=1)
        return int(a.item())

    def select_actions(self, obss: np.ndarray, *, explore: bool = True) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(obss)).to(self.device, torch.float32)
            logits = self.policy(x)
            if explore:
                a = torch.distributions.Categorical(logits=logits).sample()
            else:
                a = logits.argmax(dim=1)
        return a.cpu().numpy()

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def _update(self, obs_batch, act_batch, ret_batch, history) -> None:
        obs = torch.from_numpy(np.asarray(obs_batch)).to(self.device, torch.float32)
        actions = torch.as_tensor(act_batch, dtype=torch.int64, device=self.device)
        returns = torch.as_tensor(ret_batch, dtype=torch.float32, device=self.device)
        if self.normalize_returns and returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        logits = self.policy(obs)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        pg_loss = -(logp * returns).mean()
        loss = pg_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        history["loss"].append(float(loss.item()))

    def train(
        self,
        env_fn: Callable[[], Any],
        *,
        total_timesteps: int,
        log_every_episodes: int = 10,
        n_envs: int = 8,
        callback: Callable[[int, dict], None] | None = None,
        **kwargs: Any,
    ) -> dict[str, list[float]]:
        import gymnasium as gym

        n_envs = max(1, int(n_envs))
        envs = gym.vector.SyncVectorEnv([env_fn for _ in range(n_envs)])
        history = self._new_history()
        recent_returns: deque[float] = deque(maxlen=20)
        recent_success: deque[float] = deque(maxlen=20)
        recent_stages = {sk: deque(maxlen=20) for sk in self.STAGE_KEYS}
        saw_stages = False

        obs_buf: list[list] = [[] for _ in range(n_envs)]
        act_buf: list[list] = [[] for _ in range(n_envs)]
        rew_buf: list[list] = [[] for _ in range(n_envs)]
        batch_obs: list = []
        batch_act: list = []
        batch_ret: list = []
        n_complete = 0

        seeds = None if self.seed is None else [self.seed + i for i in range(n_envs)]
        obss, _ = envs.reset(seed=seeds)

        while self.total_steps < total_timesteps:
            actions = self.select_actions(obss, explore=True)
            next_obss, rewards, terminateds, truncateds, infos = envs.step(actions)
            dones = np.logical_or(terminateds, truncateds)

            for i in range(n_envs):
                obs_buf[i].append(obss[i])
                act_buf[i].append(int(actions[i]))
                rew_buf[i].append(float(rewards[i]))
                self.total_steps += 1

                if dones[i]:
                    # Discounted returns for this completed episode.
                    rets = [0.0] * len(rew_buf[i])
                    g = 0.0
                    for t in reversed(range(len(rew_buf[i]))):
                        g = rew_buf[i][t] + self.gamma * g
                        rets[t] = g
                    batch_obs.extend(obs_buf[i])
                    batch_act.extend(act_buf[i])
                    batch_ret.extend(rets)
                    n_complete += 1

                    info_i = self._info_at(infos, i)
                    self.total_episodes += 1
                    saw_stages = self._record_episode(
                        history, recent_returns, recent_success, recent_stages,
                        sum(rew_buf[i]), len(rew_buf[i]),
                        self._success_flag(info_i, bool(terminateds[i])), info_i,
                        log_every_episodes, saw_stages,
                    )
                    obs_buf[i], act_buf[i], rew_buf[i] = [], [], []

            obss = next_obss

            if n_complete >= self.episodes_per_update:
                self._update(batch_obs, batch_act, batch_ret, history)
                batch_obs, batch_act, batch_ret = [], [], []
                n_complete = 0

            if callback is not None:
                callback(self.total_steps, history)

        envs.close()
        return history

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
                "total_episodes": self.total_episodes,
                "obs_shape": self.obs_shape,
                "n_actions": self.n_actions,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = int(ckpt.get("total_steps", 0))
        self.total_episodes = int(ckpt.get("total_episodes", 0))
