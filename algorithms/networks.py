"""Shared CNN encoder for the pixel-based agents (REINFORCE, PPO).

Mirrors the convolutional trunk of ``dqn.QNetwork`` so every algorithm sees the
*same* representation of the image — which keeps the cross-algorithm comparison
fair. DQN keeps its own ``QNetwork`` (proven / in-flight); this module is used by
the on-policy agents. Input is the raw ``(H, W, C)`` uint8-scale RGB frame; the
encoder handles HWC->CHW and ``/255`` normalisation internally.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _is_channels_last(shape: tuple[int, ...]) -> bool:
    """HWC when the last dim is a small channel count and leading dims are spatial."""
    if len(shape) != 3:
        return False
    h, w, c = shape
    return c <= 16 and h > c and w > c


def orthogonal_init_(layer: nn.Module, gain: float = 2.0 ** 0.5, bias: float = 0.0) -> nn.Module:
    """Orthogonal weight init with zeroed bias — the standard PPO init. ``gain=sqrt(2)``
    for ReLU trunks; small gains (e.g. 0.01) on a policy head keep the initial policy
    near-uniform, which stabilises and speeds early on-policy convergence."""
    nn.init.orthogonal_(layer.weight, gain)
    if getattr(layer, "bias", None) is not None:
        nn.init.constant_(layer.bias, bias)
    return layer


class CNNEncoder(nn.Module):
    """Conv trunk -> fixed spatial map -> flat feature vector of size ``out_dim``.

    ``out_dim == 256 * width_mult * fc_mult``. Matches DQN's trunk: three strided
    convs (+ ``n_extra_conv`` same-resolution convs), a bilinear resize to an 8x8
    grid-aligned map (MPS-safe, avoids adaptive-pool constraints), then one FC layer.

    ``orthogonal=True`` applies ``sqrt(2)`` orthogonal init to every conv/linear (PPO
    convention). Left ``False`` by default so DQN/REINFORCE keep PyTorch's default init.
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        *,
        width_mult: int = 1,
        n_extra_conv: int = 0,
        fc_mult: int = 1,
        feat_hw: tuple[int, int] = (8, 8),
        orthogonal: bool = False,
    ) -> None:
        super().__init__()
        self.channels_last = _is_channels_last(obs_shape)
        c = obs_shape[-1] if self.channels_last else obs_shape[0]
        w = max(1, int(width_mult))
        c1, c2, c3 = 32 * w, 64 * w, 64 * w
        self.out_dim = 256 * w * max(1, int(fc_mult))
        self.feat_hw = (int(feat_hw[0]), int(feat_hw[1]))

        layers: list[nn.Module] = [
            nn.Conv2d(c, c1, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        ]
        for _ in range(max(0, int(n_extra_conv))):
            layers += [nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1), nn.ReLU()]
        self.features = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * self.feat_hw[0] * self.feat_hw[1], self.out_dim),
            nn.ReLU(),
        )
        if orthogonal:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    orthogonal_init_(m)

    def _nchw(self, x: torch.Tensor) -> torch.Tensor:
        if self.channels_last:
            x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return x / 255.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(self._nchw(x))
        x = F.interpolate(x, size=self.feat_hw, mode="bilinear", align_corners=False)
        return self.fc(x)
