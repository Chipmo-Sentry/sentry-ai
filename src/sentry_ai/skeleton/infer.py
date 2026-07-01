"""Score clips with the trained anomaly model and reuse the eval.pose reporting,
so the learned model is measured with the EXACT metric (frame-level ROC-AUC) the
rule-based BehaviorScorer baseline already reports — apples to apples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from sentry_ai.eval.pose_runner import ClipResult, PoseClip, build_pose_report
from sentry_ai.skeleton.model import PoseAutoencoder, frame_errors
from sentry_ai.skeleton.windows import clip_windows, stack_features


def load_checkpoint(
    path: str | Path, *, device: str = "cpu"
) -> tuple[PoseAutoencoder, dict[str, Any]]:
    ckpt = torch.load(Path(path), map_location=device, weights_only=True)
    meta = ckpt["meta"]
    model = PoseAutoencoder(int(meta["feat_dim"]), int(meta["hidden"]), int(meta["latent"])).to(
        device
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, meta


def score_clip(
    clip: PoseClip, model: PoseAutoencoder, meta: dict[str, Any], *, device: str = "cpu"
) -> NDArray[np.float32]:
    """Per-frame anomaly score (0..100) — reconstruction error scaled so the
    training threshold lands at 50 (≈ the engine's HIGH/CRITICAL boundary). Each
    frame takes the max over the windows covering it; uncovered frames stay 0.
    Scaling is monotonic, so ROC-AUC is unaffected — it's only for readability."""
    scores = np.zeros(clip.n_frames, dtype=np.float32)
    windows = clip_windows(clip, int(meta["length"]), int(meta["stride"]))
    if not windows:
        return scores
    x = torch.from_numpy(stack_features(windows)).to(device)
    fe = frame_errors(model, x).cpu().numpy()  # (N, T)
    denom = float(meta["threshold"]) if float(meta["threshold"]) > 1e-9 else 1.0
    for w, errs in zip(windows, fe, strict=True):
        for i, fidx in enumerate(w.frame_indices):
            if 0 <= fidx < clip.n_frames:
                s = min(100.0, 50.0 * float(errs[i]) / denom)
                if s > scores[fidx]:
                    scores[fidx] = s
    return scores


def clip_result(
    clip: PoseClip, model: PoseAutoencoder, meta: dict[str, Any], *, device: str = "cpu"
) -> ClipResult:
    fs = score_clip(clip, model, meta, device=device)
    if clip.frame_labels is not None and len(clip.frame_labels) == clip.n_frames:
        labels = np.asarray(clip.frame_labels, dtype=np.int_)
    else:
        labels = np.zeros(clip.n_frames, dtype=np.int_)
    return ClipResult(
        name=clip.name,
        label=clip.label,
        frame_scores=fs,
        peak=float(fs.max()) if clip.n_frames else 0.0,
        frame_labels=labels,
        fired=[],
    )


def evaluate(
    clips: list[PoseClip], model: PoseAutoencoder, meta: dict[str, Any], *, device: str = "cpu"
) -> dict[str, Any]:
    """Full learned-model report in the same shape as eval.pose's baseline."""
    results = [clip_result(c, model, meta, device=device) for c in clips]
    return build_pose_report(results)
