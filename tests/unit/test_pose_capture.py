"""Фаз 0 (ADR-0030): the camera worker rolls a per-track skeleton history and
hands the breaching track's window to the alert as training data."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sentry_ai.live_worker.camera_worker import CameraWorker
from sentry_ai.live_worker.tracker import TrackedDetection


def _worker() -> CameraWorker:
    # __init__ doesn't start threads or touch the network; a stub emitter is fine.
    return CameraWorker("cam1", "rtsp://x", emitter=SimpleNamespace())  # type: ignore[arg-type]


def _td(tid: int, *, with_kp: bool = True) -> TrackedDetection:
    kp = (np.ones((17, 3), dtype=np.float32) * 5.0) if with_kp else None
    return TrackedDetection(box=(10.0, 20.0, 40.0, 220.0), score=0.9, tracker_id=tid, keypoints=kp)


def test_capture_builds_ordered_sequence() -> None:
    w = _worker()
    w._frames_total = 1
    w._capture_pose([_td(7)])
    w._frames_total = 2
    w._capture_pose([_td(7)])
    seq = w._build_pose_sequence(7)
    assert len(seq) == 2
    assert [s["frame_idx"] for s in seq] == [1, 2]  # oldest → newest
    assert len(seq[0]["keypoints"]) == 17
    assert seq[0]["box"] == [10.0, 20.0, 40.0, 220.0]


def test_build_sequence_unknown_track_is_empty() -> None:
    assert _worker()._build_pose_sequence(999) == []


def test_capture_skips_persons_without_keypoints() -> None:
    w = _worker()
    w._frames_total = 1
    w._capture_pose([_td(1, with_kp=False)])
    assert w._build_pose_sequence(1) == []


def test_capture_separates_tracks() -> None:
    w = _worker()
    w._frames_total = 1
    w._capture_pose([_td(1), _td(2)])
    assert len(w._build_pose_sequence(1)) == 1
    assert len(w._build_pose_sequence(2)) == 1
