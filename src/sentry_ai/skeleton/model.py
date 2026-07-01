"""Compact temporal-convolution autoencoder over a pose window. The whole point
of "anomaly-first": train ONLY on normal motion so the net learns to reconstruct
it; an unusual window reconstructs badly → high error → anomaly.

Small on purpose (a few hundred KB): two strided 1-D convs down to a narrow
latent, mirrored back up. 1-D conv over time is cheap on CPU and exports cleanly
to ONNX/OpenVINO for the edge — no recurrence, no attention needed at this size.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from sentry_ai.skeleton.features import FEAT_DIM


class PoseAutoencoder(nn.Module):
    """(B, T, F) → reconstructed (B, T, F). Anomaly = per-window/-frame MSE."""

    def __init__(self, feat_dim: int = FEAT_DIM, hidden: int = 64, latent: int = 16) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden = hidden
        self.latent = latent
        self.enc = nn.Sequential(
            nn.Conv1d(feat_dim, hidden, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, latent, 3, padding=1),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Conv1d(latent, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, feat_dim, 5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        z = self.enc(x.transpose(1, 2))  # (B, F, T) → (B, latent, T/2)
        y = self.dec(z)  # (B, F, ~T)
        # stride-2 ↓ then ×2 ↑ can be off by one for odd T — crop/pad to match.
        if y.shape[-1] > t:
            y = y[..., :t]
        elif y.shape[-1] < t:
            y = nn.functional.pad(y, (0, t - y.shape[-1]))
        out: torch.Tensor = y.transpose(1, 2)
        return out  # (B, T, F)


@torch.no_grad()
def frame_errors(model: PoseAutoencoder, x: torch.Tensor) -> torch.Tensor:
    """Per-timestep reconstruction error: (B, T, F) → (B, T), mean over features."""
    model.eval()
    recon = model(x)
    err: torch.Tensor = ((recon - x) ** 2).mean(dim=2)
    return err


@torch.no_grad()
def window_errors(model: PoseAutoencoder, x: torch.Tensor) -> NDArray[np.float32]:
    """Per-window reconstruction error: (B, T, F) → (B,) numpy, mean over T,F."""
    return frame_errors(model, x).mean(dim=1).cpu().numpy().astype(np.float32)
