"""Windowing: counts, edge-padding of short tracks, and anomaly labelling."""

from __future__ import annotations

import numpy as np

from sentry_ai.eval.pose_runner import FrameKP, PersonTrack, PoseClip
from sentry_ai.skeleton.features import FEAT_DIM
from sentry_ai.skeleton.windows import MIN_FRAMES, clip_windows, track_windows


def _kp() -> np.ndarray:
    xy = np.arange(34, dtype=np.float32).reshape(17, 2)
    return np.concatenate([xy, np.ones((17, 1), np.float32)], axis=1)


def _track(pid: str, n: int) -> PersonTrack:
    return PersonTrack(pid, [FrameKP(i, _kp()) for i in range(n)])


def _clip(n: int, labels: np.ndarray | None = None) -> PoseClip:
    return PoseClip("c", [], n, labels)


def test_sliding_window_count() -> None:
    clip = _clip(20)
    wins = track_windows(_track("p", 20), clip, length=8, stride=4)
    # floor((20-8)/4)+1 = 4
    assert len(wins) == 4
    assert all(w.features.shape == (8, FEAT_DIM) for w in wins)
    assert wins[0].frame_indices == [0, 1, 2, 3, 4, 5, 6, 7]
    assert wins[1].frame_indices[0] == 4


def test_short_track_edge_padded_to_one_window() -> None:
    clip = _clip(6)
    wins = track_windows(_track("p", 6), clip, length=10, stride=4)
    assert len(wins) == 1
    w = wins[0]
    assert w.features.shape == (10, FEAT_DIM)
    assert w.frame_indices[:6] == [0, 1, 2, 3, 4, 5]
    assert w.frame_indices[6:] == [-1, -1, -1, -1]  # pad timesteps ignored downstream


def test_too_short_track_skipped() -> None:
    clip = _clip(MIN_FRAMES - 1)
    assert track_windows(_track("p", MIN_FRAMES - 1), clip, length=8, stride=4) == []


def test_window_label_from_frame_labels() -> None:
    labels = np.zeros(20, dtype=np.int_)
    labels[10:13] = 1  # anomalous frames 10,11,12
    clip = _clip(20, labels)
    wins = track_windows(_track("p", 20), clip, length=8, stride=4)
    # window starting at 8 covers frames 8..15 → overlaps the anomaly → label 1
    by_start = {w.frame_indices[0]: w.label for w in wins}
    assert by_start[8] == 1
    assert by_start[0] == 0  # frames 0..7, no anomaly


def test_clip_windows_spans_all_persons() -> None:
    clip = _clip(16)
    clip.persons.extend([_track("a", 16), _track("b", 16)])
    wins = clip_windows(clip, length=8, stride=8)
    assert {w.person_id for w in wins} == {"a", "b"}
