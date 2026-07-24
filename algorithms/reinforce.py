"""REINFORCE with a learned baseline (Monte-Carlo policy gradient) from scratch.

Policy-based method: a shared CNN encoder feeds a policy head (action logits) and a
value head used **only as a baseline**. Updates use complete-episode discounted MC
returns; the advantage is ``return - V(s)`` (the value net is fit to the same MC returns
by regression). Crucially there is **no bootstrapping** — ``V`` never appears in the
return target — which keeps this a REINFORCE-family method, distinct from the
actor-critic PPO (GAE bootstrap + clipped surrogate). The learned baseline is the
variance reduction a value-free MC estimator lacks; without it the policy stays
state-independent and its greedy argmax can't navigate.

Collection is vectorised (``n_envs`` parallel envs); an update fires once enough
complete episodes are gathered. Same surface as ``DQN`` (``select_action`` / ``train``
returning a plottable history / ``save`` / ``load``).
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
from .networks import CNNEncoder, orthogonal_init_


class PolicyNetwork(nn.Module):
    """Shared CNN encoder -> {action logits, state-value baseline}.

    ``orthogonal=True`` applies ``sqrt(2)`` orthogonal init to the trunk, a tiny ``0.01``
    gain on the logit head (near-uniform initial policy) and ``1.0`` on the value head —
    the same scheme that lets PPO sharpen to a *deterministic* optimum, which is what
    closes REINFORCE's greedy-vs-stochastic gap and avoids early single-action collapse.
    """

class PolicyNetwork(nn.Module):
    """CNN encoder -> action-logit head.

    ``orthogonal=True`` applies ``sqrt(2)`` orthogonal init to the trunk and a tiny
    ``0.01`` gain on the logit head (near-uniform initial policy) — the same scheme that
    lets PPO sharpen to a *deterministic* optimum, which is what closes REINFORCE's
    greedy-vs-stochastic gap and avoids early single-action collapse.
    """

    def __init__(self, obs_shape, n_actions, *, width_mult=1, n_extra_conv=0, fc_mult=1,
                 orthogonal=True) -> None:
        super().__init__()
        self.encoder = CNNEncoder(
            obs_shape, width_mult=width_mult, n_extra_conv=n_extra_conv, fc_mult=fc_mult,
            orthogonal=orthogonal,
        )
        self.pi = nn.Linear(self.encoder.out_dim, n_actions)
        if orthogonal:
            orthogonal_init_(self.pi, gain=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pi(self.encoder(x))  # logits


class ValueNetwork(nn.Module):
    """Separate CNN encoder -> scalar state-value baseline.

    Kept **separate** from the policy (its own trunk): sharing the encoder lets the
    value regression's large gradients dominate and corrupt the policy features, which
    empirically kept the policy state-independent. A dedicated baseline net avoids that.
    """

    def __init__(self, obs_shape, *, width_mult=1, n_extra_conv=0, fc_mult=1,
                 orthogonal=True) -> None:
        super().__init__()
        self.encoder = CNNEncoder(
            obs_shape, width_mult=width_mult, n_extra_conv=n_extra_conv, fc_mult=fc_mult,
            orthogonal=orthogonal,
        )
        self.v = nn.Linear(self.encoder.out_dim, 1)
        if orthogonal:
            orthogonal_init_(self.v, gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.v(self.encoder(x)).squeeze(-1)


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
        value_lr: float = 1e-3,
        entropy_coef: float = 0.05,
        ent_coef_final: float | None = None,
        ent_anneal_steps: int = 0,
        normalize_returns: bool = True,
        episodes_per_update: int = 16,
        update_epochs: int = 1,
        num_minibatches: int = 1,
        reward_scale: float = 0.1,
        max_grad_norm: float = 5.0,
        width_mult: int = 1,
        n_extra_conv: int = 0,
        fc_mult: int = 1,
        orthogonal: bool = True,
    ) -> None:
        super().__init__(obs_shape, n_actions, device=device, seed=seed)
        self.gamma = float(gamma)
        self.entropy_coef = float(entropy_coef)
        # Entropy anneal (PPO analogue of ε-decay): full exploration pressure while the
        # policy discovers the task, decaying to a small floor for greedy consolidation.
        self.ent_coef_final = float(entropy_coef if ent_coef_final is None else ent_coef_final)
        self.ent_anneal_steps = int(ent_anneal_steps)
        self.normalize_returns = bool(normalize_returns)
        self.episodes_per_update = max(1, int(episodes_per_update))
        # Several gradient passes over each collected batch (advantages fixed from the
        # baseline). One weak PG step per batch can't overcome majority-action bias
        # (“forward” dominates), leaving the greedy argmax degenerate; multiple strong
        # passes are what let the policy make “turn” the argmax at the states that need it.
        self.update_epochs = max(1, int(update_epochs))
        self.num_minibatches = max(1, int(num_minibatches))
        # Scale MC returns so the value-baseline MSE stays ~O(1); raw shaped returns (~±50)
        # otherwise blow up the value gradient. Reported episode returns stay RAW.
        self.reward_scale = float(reward_scale)
        self.max_grad_norm = float(max_grad_norm)

        self.policy = PolicyNetwork(
            obs_shape, n_actions, width_mult=width_mult, n_extra_conv=n_extra_conv,
            fc_mult=fc_mult, orthogonal=orthogonal,
        ).to(self.device)
        self.value_net = ValueNetwork(
            obs_shape, width_mult=width_mult, n_extra_conv=n_extra_conv,
            fc_mult=fc_mult, orthogonal=orthogonal,
        ).to(self.device)
        self.optimizer = Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = Adam(self.value_net.parameters(), lr=value_lr)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.policy.parameters()) + sum(
            p.numel() for p in self.value_net.parameters()
        )

    def current_ent_coef(self) -> float:
        """Linearly annealed entropy coefficient for the current ``total_steps``."""
        if self.ent_anneal_steps <= 0:
            return self.entropy_coef
        t = min(1.0, self.total_steps / self.ent_anneal_steps)
        return self.entropy_coef + t * (self.ent_coef_final - self.entropy_coef)

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
        returns = torch.as_tensor(ret_batch, dtype=torch.float32, device=self.device) * self.reward_scale

        # Advantage = MC return - V(s), computed ONCE from the current baseline (no
        # bootstrap). Fixed as the target for the epoch passes below.
        with torch.no_grad():
            advantages = returns - self.value_net(obs)
            if self.normalize_returns and advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ent_coef = self.current_ent_coef()
        n = obs.shape[0]
        mb_size = max(1, n // self.num_minibatches)
        idx = np.arange(n)
        last_pg = 0.0
        for _ in range(self.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, mb_size):
                mb = torch.from_numpy(idx[start:start + mb_size]).to(self.device)

                # policy step: plain PG (no importance ratio / clipping -> still REINFORCE)
                logits = self.policy(obs[mb])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(actions[mb])
                entropy = dist.entropy().mean()
                pg_loss = -(logp * advantages[mb]).mean() - ent_coef * entropy
                self.optimizer.zero_grad(set_to_none=True)
                pg_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # baseline step: fit V(s) to the MC returns by regression
                value_loss = F.mse_loss(self.value_net(obs[mb]), returns[mb])
                self.value_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.value_optimizer.step()
                last_pg = float(pg_loss.item())
        history["loss"].append(last_pg)

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
        from gymnasium.vector import AutoresetMode

        n_envs = max(1, int(n_envs))
        # SAME_STEP autoreset exposes the true final obs of a done episode in
        # ``infos['final_obs']`` (the returned obs is already the reset), so a *timed-out*
        # episode can bootstrap its cut-off tail from V(final_obs) instead of being scored
        # as if the future were worthless.
        envs = gym.vector.SyncVectorEnv(
            [env_fn for _ in range(n_envs)], autoreset_mode=AutoresetMode.SAME_STEP
        )
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

            # Bootstrap value (RAW units) for the cut-off tail of TIMED-OUT episodes only;
            # true terminals get 0. V is trained on scaled returns, so divide back out.
            tail = np.zeros(n_envs, dtype=np.float32)
            trunc_only = np.logical_and(truncateds, np.logical_not(terminateds))
            if trunc_only.any():
                fo = infos.get("final_obs", infos.get("final_observation"))
                if fo is not None:
                    idx = np.nonzero(trunc_only)[0]
                    fobs = np.stack([np.asarray(fo[j]) for j in idx])
                    with torch.no_grad():
                        fx = torch.from_numpy(fobs).to(self.device, torch.float32)
                        v = self.value_net(fx).cpu().numpy()
                    tail[idx] = self.gamma * v / max(self.reward_scale, 1e-8)

            for i in range(n_envs):
                obs_buf[i].append(obss[i])
                act_buf[i].append(int(actions[i]))
                rew_buf[i].append(float(rewards[i]))
                self.total_steps += 1

                if dones[i]:
                    # Discounted return-to-go; the tail seed bootstraps a timed-out future.
                    rets = [0.0] * len(rew_buf[i])
                    g = float(tail[i])
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
                "value_net": self.value_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "value_optimizer": self.value_optimizer.state_dict(),
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
        if "value_net" in ckpt:
            self.value_net.load_state_dict(ckpt["value_net"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "value_optimizer" in ckpt:
            self.value_optimizer.load_state_dict(ckpt["value_optimizer"])
        self.total_steps = int(ckpt.get("total_steps", 0))
        self.total_episodes = int(ckpt.get("total_episodes", 0))
