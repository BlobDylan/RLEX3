"""Random Network Distillation (RND) intrinsic exploration — Burda et al. 2018.

A fixed, randomly-initialised *target* network encodes each observation; a
*predictor* network is trained to match it. Prediction error is large on novel
states and small on familiar ones, yielding an intrinsic reward that pushes the
agent toward unseen regions of the environment (e.g. through the open door into
the never-visited right room, then onward to water / lava / goal). Pure RL — no
demonstrations.

Usage inside DQN (single value head): store ``reward = extrinsic·reward_scale +
rnd_coef · intrinsic`` so the novelty bonus lives on the same O(1) scale as the
scaled extrinsic reward; train the predictor on sampled observations each update.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam


class RunningMeanStd:
    """Scalar running mean/variance (Welford, parallel update)."""

    def __init__(self) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size == 0:
            return
        bm, bv, bc = float(x.mean()), float(x.var()), int(x.size)
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        m_a = self.var * self.count
        m_b = bv * bc
        self.var = (m_a + m_b + delta * delta * self.count * bc / tot) / tot
        self.count = tot

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var)) + 1e-8


class _RNDEncoder(nn.Module):
    """Small conv encoder → feature vector (used for both target and predictor)."""

    def __init__(self, obs_shape: tuple[int, ...], out_dim: int = 128) -> None:
        super().__init__()
        self.channels_last = (
            len(obs_shape) == 3 and obs_shape[-1] <= 16 and obs_shape[0] > obs_shape[-1]
        )
        if self.channels_last:
            h, w, c = obs_shape
        else:
            c, h, w = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, c, h, w)).shape[1]
        self.head = nn.Linear(flat, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channels_last:
            x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = x / 255.0
        return self.head(self.conv(x))


class RNDExploration:
    """Intrinsic novelty reward = normalized predictor error on a state."""

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        device: torch.device,
        *,
        lr: float = 1e-4,
        out_dim: int = 128,
    ) -> None:
        self.device = device
        self.target = _RNDEncoder(obs_shape, out_dim).to(device)
        self.predictor = _RNDEncoder(obs_shape, out_dim).to(device)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.target.eval()
        self.optimizer = Adam(self.predictor.parameters(), lr=lr)
        self.rms = RunningMeanStd()

    @torch.no_grad()
    def intrinsic(self, obs_batch: np.ndarray) -> np.ndarray:
        """Normalized intrinsic reward for a batch ``(N, *obs_shape)``. Updates rms."""
        x = torch.as_tensor(np.ascontiguousarray(obs_batch), dtype=torch.float32, device=self.device)
        err = ((self.predictor(x) - self.target(x)) ** 2).mean(dim=1)
        e = err.detach().cpu().numpy()
        self.rms.update(e)
        return e / self.rms.std

    def train(self, obs_batch: torch.Tensor) -> float:
        """One predictor update toward the frozen target on a float32 obs batch."""
        pred = self.predictor(obs_batch)
        with torch.no_grad():
            tgt = self.target(obs_batch)
        loss = ((pred - tgt) ** 2).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach())
