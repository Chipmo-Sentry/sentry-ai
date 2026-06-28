"""Slice per-person pose tracks into fixed-length feature windows. Pure numpy.

The model scores a short motion WINDOW (e.g. ~2 s of pose), not a single frame —
behaviour lives in movement over time. We slide a window of `length` frames with
`stride` along each person's track and turn each into a (length, FEAT_DIM) array.

A window's binary label is 1 if ANY frame it covers is anomalous (PoseLift's
per-frame .npy). Training uses only label-0 (normal) windows (anomaly-first);
eval uses every window and maps scores back to real frame indices for the same
frame-level ROC-AUC the rule baseline reports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sentry_ai.eval.pose_runner import PersonTrack, PoseClip
from sentry_ai.skeleton.features import FEAT_DIM, frame_features

# A track shorter than this many real frames carries too little motion to be a
# meaningful window; skip it. Between this and `length` we edge-pad one window.
MIN_FRAMES = 4


@dataclass(slots=True)
class Window:
    features: NDArray[np.float32]  # (length, FEAT_DIM)
    frame_indices: list[int]  # real clip frame idx per timestep (-1 = edge pad)
    label: int  # 1 if any covered (real) frame is anomalous
    person_id: str
    clip_name: str


def _window_label(clip: PoseClip, idxs: list[int]) -> int:
    fl = clip.frame_labels
    if fl is None:
        return 0
    return int(any(0 <= i < len(fl) and fl[i] == 1 for i in idxs))


def track_windows(track: PersonTrack, clip: PoseClip, length: int, stride: int) -> list[Window]:
    frames = track.frames  # already sorted by frame_idx (loader guarantees)
    if len(frames) < MIN_FRAMES:
        return []
    feats = [(fk.frame_idx, frame_features(fk.kp)) for fk in frames]

    # Short track (MIN_FRAMES..length): one edge-padded window so brief tracks
    # still contribute. Pad timesteps carry frame_idx -1 → ignored by labels/eval.
    if len(feats) < length:
        pad = length - len(feats)
        idxs = [fi for fi, _ in feats] + [-1] * pad
        arr = np.stack([f for _, f in feats] + [feats[-1][1]] * pad).astype(np.float32)
        return [Window(arr, idxs, _window_label(clip, idxs), track.person_id, clip.name)]

    out: list[Window] = []
    for start in range(0, len(feats) - length + 1, stride):
        chunk = feats[start : start + length]
        idxs = [fi for fi, _ in chunk]
        arr = np.stack([f for _, f in chunk]).astype(np.float32)
        out.append(Window(arr, idxs, _window_label(clip, idxs), track.person_id, clip.name))
    return out


def clip_windows(clip: PoseClip, length: int, stride: int) -> list[Window]:
    out: list[Window] = []
    for track in clip.persons:
        out.extend(track_windows(track, clip, length, stride))
    return out


def dataset_windows(clips: list[PoseClip], length: int, stride: int) -> list[Window]:
    out: list[Window] = []
    for clip in clips:
        out.extend(clip_windows(clip, length, stride))
    return out


def stack_features(windows: list[Window]) -> NDArray[np.float32]:
    """(N, length, FEAT_DIM) tensor-ready array from a list of windows."""
    if not windows:
        return np.zeros((0, 0, FEAT_DIM), dtype=np.float32)
    return np.stack([w.features for w in windows]).astype(np.float32)
