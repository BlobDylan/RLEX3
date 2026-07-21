"""PPO (Proximal Policy Optimization) from scratch, for MiniGrid pixels.

Actor-critic method: a shared CNN encoder feeds a policy head (action logits) and
a value head (state value). Training collects fixed-length vectorised rollouts,
computes GAE(λ) advantages + returns, then runs several epochs of minibatch SGD on
the clipped surrogate objective + value loss + entropy bonus. Same surface as
``DQN``/``REINFORCE`` (``select_action``, ``train`` returning a plottable history,
``save``/``load``).
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
from .networks import CNNEncoder


class ActorCritic(nn.Module):
    """Shared CNN encoder -> {policy logits, state value}."""

    def __init__(self, obs_shape, n_actions, *, width_mult=1, n_extra_conv=0, fc_mult=1) -> None:
        super().__init__()
        self.encoder = CNNEncoder(
            obs_shape, width_mult=width_mult, n_extra_conv=n_extra_conv, fc_mult=fc_mult
        )
        self.pi = nn.Linear(self.encoder.out_dim, n_actions)
        self.v = nn.Linear(self.encoder.out_dim, 1)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return self.pi(z), self.v(z).squeeze(-1)


class PPO(BaseAlgorithm):
    """Clipped-objective PPO with GAE advantages."""

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_actions: int,
        device: str | torch.device | None = None,
        seed: int | None = None,
        *,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        lr: float = 2.5e-4,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rollout_steps: int = 128,
        update_epochs: int = 4,
        num_minibatches: int = 4,
        n_envs: int = 8,
        anneal_lr: bool = True,
        width_mult: int = 1,
        n_extra_conv: int = 0,
        fc_mult: int = 1,
    ) -> None:
        super().__init__(obs_shape, n_actions, device=device, seed=seed)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_coef = float(clip_coef)
        self.ent_coef = float(ent_coef)
        self.vf_coef = float(vf_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.rollout_steps = int(rollout_steps)
        self.update_epochs = int(update_epochs)
        self.num_minibatches = int(num_minibatches)
        self.n_envs = max(1, int(n_envs))
        # Linearly decay lr -> 0 over training (CleanRL-style): lets the policy COMMIT to
        # deterministic actions late so the greedy/argmax policy matches the stochastic one.
        self.anneal_lr = bool(anneal_lr)
        self.base_lr = float(lr)

        self.net = ActorCritic(
            obs_shape, n_actions, width_mult=width_mult, n_extra_conv=n_extra_conv, fc_mult=fc_mult
        ).to(self.device)
        self.optimizer = Adam(self.net.parameters(), lr=lr, eps=1e-5)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------
    def select_action(self, obs: np.ndarray, *, explore: bool = True) -> int:
        with torch.no_grad():
            logits, _ = self.net(self.to_tensor(obs))
            if explore:
                a = torch.distributions.Categorical(logits=logits).sample()
            else:
                a = logits.argmax(dim=1)
        return int(a.item())

    def _policy_value(self, obss_t: torch.Tensor):
        """Return sampled actions, their log-probs, and values for a batch (numpy-in)."""
        logits, value = self.net(obss_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def train(
        self,
        env_fn: Callable[[], Any],
        *,
        total_timesteps: int,
        log_every_episodes: int = 10,
        n_envs: int | None = None,
        callback: Callable[[int, dict], None] | None = None,
        **kwargs: Any,
    ) -> dict[str, list[float]]:
        import gymnasium as gym

        n_envs = self.n_envs if n_envs is None else max(1, int(n_envs))
        T = self.rollout_steps
        envs = gym.vector.SyncVectorEnv([env_fn for _ in range(n_envs)])
        history = self._new_history()
        recent_returns: deque[float] = deque(maxlen=20)
        recent_success: deque[float] = deque(maxlen=20)
        recent_stages = {sk: deque(maxlen=20) for sk in self.STAGE_KEYS}
        saw_stages = False

        # Rollout storage (T, n_envs, ...).
        obs_shape = self.obs_shape
        obs_store = np.zeros((T, n_envs, *obs_shape), dtype=np.float32)
        act_store = np.zeros((T, n_envs), dtype=np.int64)
        logp_store = np.zeros((T, n_envs), dtype=np.float32)
        val_store = np.zeros((T, n_envs), dtype=np.float32)
        rew_store = np.zeros((T, n_envs), dtype=np.float32)
        done_store = np.zeros((T, n_envs), dtype=np.float32)

        ep_returns = np.zeros(n_envs, dtype=np.float64)
        ep_lens = np.zeros(n_envs, dtype=np.int64)

        seeds = None if self.seed is None else [self.seed + i for i in range(n_envs)]
        obss, _ = envs.reset(seed=seeds)

        while self.total_steps < total_timesteps:
            # ---- collect a rollout of T steps ----
            for t in range(T):
                obs_t = torch.from_numpy(np.ascontiguousarray(obss)).to(self.device, torch.float32)
                with torch.no_grad():
                    action, logp, value = self._policy_value(obs_t)
                actions = action.cpu().numpy()

                next_obss, rewards, terminateds, truncateds, infos = envs.step(actions)
                dones = np.logical_or(terminateds, truncateds)

                obs_store[t] = obss
                act_store[t] = actions
                logp_store[t] = logp.cpu().numpy()
                val_store[t] = value.cpu().numpy()
                rew_store[t] = rewards
                done_store[t] = dones.astype(np.float32)

                ep_returns += rewards
                ep_lens += 1
                self.total_steps += n_envs

                for i in range(n_envs):
                    if dones[i]:
                        info_i = self._info_at(infos, i)
                        self.total_episodes += 1
                        saw_stages = self._record_episode(
                            history, recent_returns, recent_success, recent_stages,
                            float(ep_returns[i]), int(ep_lens[i]),
                            self._success_flag(info_i, bool(terminateds[i])), info_i,
                            log_every_episodes, saw_stages,
                        )
                        ep_returns[i] = 0.0
                        ep_lens[i] = 0

                obss = next_obss

            # ---- bootstrap value + GAE ----
            with torch.no_grad():
                last_obs = torch.from_numpy(np.ascontiguousarray(obss)).to(self.device, torch.float32)
                _, last_value = self.net(last_obs)
                last_value = last_value.cpu().numpy()

            adv_store = np.zeros((T, n_envs), dtype=np.float32)
            last_gae = np.zeros(n_envs, dtype=np.float32)
            for t in reversed(range(T)):
                next_nonterminal = 1.0 - done_store[t]
                next_value = last_value if t == T - 1 else val_store[t + 1]
                delta = rew_store[t] + self.gamma * next_value * next_nonterminal - val_store[t]
                last_gae = delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae
                adv_store[t] = last_gae
            ret_store = adv_store + val_store

            if self.anneal_lr:
                frac = max(0.0, 1.0 - self.total_steps / max(1, total_timesteps))
                for group in self.optimizer.param_groups:
                    group["lr"] = self.base_lr * frac

            self._update(obs_store, act_store, logp_store, adv_store, ret_store, history)

            if callback is not None:
                callback(self.total_steps, history)

        envs.close()
        return history

    def _update(self, obs_store, act_store, logp_store, adv_store, ret_store, history) -> None:
        b_obs = torch.from_numpy(obs_store.reshape(-1, *self.obs_shape)).to(self.device, torch.float32)
        b_act = torch.from_numpy(act_store.reshape(-1)).to(self.device, torch.int64)
        b_logp = torch.from_numpy(logp_store.reshape(-1)).to(self.device, torch.float32)
        b_adv = torch.from_numpy(adv_store.reshape(-1)).to(self.device, torch.float32)
        b_ret = torch.from_numpy(ret_store.reshape(-1)).to(self.device, torch.float32)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        n = b_obs.shape[0]
        mb_size = max(1, n // self.num_minibatches)
        idx = np.arange(n)
        last_loss = 0.0
        for _ in range(self.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, mb_size):
                mb = idx[start:start + mb_size]
                logits, value = self.net(b_obs[mb])
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(b_act[mb])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - b_logp[mb])
                adv = b_adv[mb]
                pg1 = -adv * ratio
                pg2 = -adv * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                v_loss = F.mse_loss(value, b_ret[mb])
                loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()
                last_loss = float(loss.item())
        history["loss"].append(last_loss)

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "net": self.net.state_dict(),
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
        self.net.load_state_dict(ckpt["net"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = int(ckpt.get("total_steps", 0))
        self.total_episodes = int(ckpt.get("total_episodes", 0))
