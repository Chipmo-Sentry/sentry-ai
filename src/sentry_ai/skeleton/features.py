"""Pose → feature vector for the anomaly model. Pure numpy (no torch), so the
normalisation is unit-tested without a model.

A raw COCO-17 keypoint is in PIXEL coords, so the same gesture at the left vs
right of the frame, near vs far, looks completely different to a model. We make
each pose translation- and scale-invariant before it ever reaches the network:

  * translation — subtract the hip midpoint (the body centre), so position in the
    frame doesn't matter.
  * scale       — divide by the torso length (shoulders→hips), so a shopper close
    to the camera and one far away map to the same skeleton.

Missing keypoints (YOLO conf 0, zeroed upstream) collapse to the centre (0,0)
after normalisation — a neutral "no signal" the model learns to ignore, rather
than a spurious pixel coordinate.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

COCO_17 = 17
FEAT_DIM = COCO_17 * 2  # flattened (x, y) per joint

# COCO-17 indices used for the body-frame normalisation.
_L_SHO, _R_SHO = 5, 6
_L_HIP, _R_HIP = 11, 12

_MIN_SCALE = 1e-3  # floor so a degenerate (flat) skeleton can't divide by ~0


def _scale(xy: NDArray[np.float32], valid: NDArray[np.bool_]) -> float:
    """Body size estimate: torso length if shoulders+hips are present, else the
    vertical span of valid joints, else 1.0 (already-normalised / degenerate)."""
    if valid[_L_SHO] and valid[_R_SHO] and valid[_L_HIP] and valid[_R_HIP]:
        sho = (xy[_L_SHO] + xy[_R_SHO]) / 2.0
        hip = (xy[_L_HIP] + xy[_R_HIP]) / 2.0
        d = float(np.linalg.norm(sho - hip))
        if d > _MIN_SCALE:
            return d
    if valid.any():
        span = float(xy[valid, 1].max() - xy[valid, 1].min())
        if span > _MIN_SCALE:
            return span
    return 1.0


def normalize_pose(kp: NDArray[np.float32]) -> NDArray[np.float32]:
    """(17, 3)[x, y, conf] (or (17, 2)) → (17, 2) body-centred, scale-normalised.

    Invalid joints (conf <= 0) are returned as (0, 0) — i.e. the body centre."""
    kp = np.asarray(kp, dtype=np.float32)
    xy = kp[:, :2].astype(np.float32).copy()
    conf = kp[:, 2] if kp.shape[1] >= 3 else np.ones(COCO_17, dtype=np.float32)
    valid = conf > 0.0
    if valid[_L_HIP] and valid[_R_HIP]:
        center = (xy[_L_HIP] + xy[_R_HIP]) / 2.0
    elif valid.any():
        center = xy[valid].mean(axis=0)
    else:
        return np.zeros((COCO_17, 2), dtype=np.float32)
    scale = _scale(xy, valid)
    out = (xy - center) / scale
    out[~valid] = 0.0
    norm: NDArray[np.float32] = out.astype(np.float32)
    return norm


def frame_features(kp: NDArray[np.float32]) -> NDArray[np.float32]:
    """One pose → a flat (34,) feature vector for the model."""
    return normalize_pose(kp).reshape(-1)
