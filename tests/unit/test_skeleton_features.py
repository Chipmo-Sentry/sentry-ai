"""Pose normalisation must be translation- and scale-invariant and must treat
missing joints as 'no signal'. Pure numpy — no torch, no model."""

from __future__ import annotations

import numpy as np

from sentry_ai.skeleton.features import FEAT_DIM, frame_features, normalize_pose


def _pose() -> np.ndarray:
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 200, size=(17, 2)).astype(np.float32)
    return np.concatenate([xy, np.ones((17, 1), np.float32)], axis=1)


def test_translation_invariant() -> None:
    kp = _pose()
    shifted = kp.copy()
    shifted[:, :2] += np.array([137.0, -42.0], dtype=np.float32)
    assert np.allclose(normalize_pose(kp), normalize_pose(shifted), atol=1e-5)


def test_scale_invariant() -> None:
    kp = _pose()
    scaled = kp.copy()
    scaled[:, :2] *= 2.5  # zoom in (scales center + torso together)
    assert np.allclose(normalize_pose(kp), normalize_pose(scaled), atol=1e-5)


def test_missing_keypoint_collapses_to_center() -> None:
    kp = _pose()
    kp[9, 2] = 0.0  # left wrist confidence 0 → invalid
    out = normalize_pose(kp)
    assert tuple(out[9]) == (0.0, 0.0)


def test_all_missing_returns_zeros() -> None:
    kp = _pose()
    kp[:, 2] = 0.0
    assert np.array_equal(normalize_pose(kp), np.zeros((17, 2), np.float32))


def test_frame_features_shape() -> None:
    assert frame_features(_pose()).shape == (FEAT_DIM,)


def test_accepts_xy_only() -> None:
    kp = _pose()[:, :2]  # (17, 2), no confidence column
    assert normalize_pose(kp).shape == (17, 2)
