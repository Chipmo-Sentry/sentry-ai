"""Train the pose anomaly autoencoder on NORMAL motion + pick an anomaly
threshold. Device-agnostic (CPU here; the same code runs on a free Colab GPU for
the real PoseLift run — just pass device='cuda')."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from sentry_ai.eval.pose_runner import PoseClip
from sentry_ai.skeleton.features import FEAT_DIM
from sentry_ai.skeleton.model import PoseAutoencoder, window_errors
from sentry_ai.skeleton.windows import dataset_windows, stack_features


@dataclass(slots=True)
class TrainConfig:
    length: int = 32  # ~2 s of pose at 15 fps
    stride: int = 8
    epochs: int = 40
    lr: float = 1e-3
    batch_size: int = 64
    hidden: int = 64
    latent: int = 16
    threshold_pct: float = 95.0  # normal-window error percentile → anomaly cutoff
    seed: int = 0


def train_autoencoder(
    clips: list[PoseClip],
    cfg: TrainConfig | None = None,
    *,
    device: str = "cpu",
    log: Callable[[str], None] = lambda _s: None,
) -> tuple[PoseAutoencoder, dict[str, Any]]:
    """Fit on label-0 (normal) windows; return the model + a meta dict carrying
    everything inference needs (arch dims, window spec, anomaly threshold)."""
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    windows = dataset_windows(clips, cfg.length, cfg.stride)
    normal = [w for w in windows if w.label == 0]
    if not normal:
        raise ValueError("no normal (label-0) windows to train on")
    x = torch.from_numpy(stack_features(normal)).to(device)  # (N, T, F)
    n = int(x.shape[0])

    model = PoseAutoencoder(FEAT_DIM, cfg.hidden, cfg.latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()
    report_every = max(1, cfg.epochs // 10)
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, cfg.batch_size):
            xb = x[perm[i : i + cfg.batch_size]]
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        if epoch % report_every == 0 or epoch == cfg.epochs - 1:
            log(f"epoch {epoch + 1}/{cfg.epochs}  loss={total / n:.6f}")

    threshold = float(np.percentile(window_errors(model, x), cfg.threshold_pct))
    meta: dict[str, Any] = {
        "feat_dim": FEAT_DIM,
        "hidden": cfg.hidden,
        "latent": cfg.latent,
        "length": cfg.length,
        "stride": cfg.stride,
        "threshold": threshold,
        "threshold_pct": cfg.threshold_pct,
        "n_normal_windows": n,
        "n_total_windows": len(windows),
    }
    return model, meta


def save_checkpoint(model: PoseAutoencoder, meta: dict[str, Any], path: str | Path) -> None:
    torch.save({"state_dict": model.state_dict(), "meta": meta}, Path(path))


def export_onnx(model: PoseAutoencoder, meta: dict[str, Any], path: str | Path) -> None:
    """Export to ONNX for the edge (OpenVINO consumes ONNX directly). Dynamic
    batch so the agent can score several people at once.

    Needs the export-only `onnx` package (not shipped — training/eval don't use
    it). Raises a clear RuntimeError if it's missing rather than a deep traceback."""
    try:
        import onnx  # type: ignore[import-not-found]  # noqa: F401 — export-only dep
    except ModuleNotFoundError as e:
        raise RuntimeError("ONNX export needs the 'onnx' package: uv pip install onnx") from e
    model.eval()
    dummy = torch.zeros(1, int(meta["length"]), int(meta["feat_dim"]))
    torch.onnx.export(
        model,
        (dummy,),
        str(path),
        input_names=["pose_window"],
        output_names=["recon"],
        dynamic_axes={"pose_window": {0: "batch"}, "recon": {0: "batch"}},
        opset_version=17,
        # Legacy TorchScript exporter: the torch 2.x dynamo path needs onnxscript,
        # an extra dep we don't ship. The legacy path produces a standard ONNX that
        # OpenVINO consumes directly — all we need for the edge.
        dynamo=False,
    )
