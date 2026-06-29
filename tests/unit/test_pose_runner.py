"""Unit tests for the pose-only Stage-1 eval path (PoseLift baseline).

No GPU, no model, no real dataset — synthetic clips + a round-tripped pickle
exercise the loader, the replay/alignment, ROC-AUC, and the report builder.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from sentry_ai.eval.pose_runner import (
    ClipResult,
    FrameKP,
    PersonTrack,
    PoseClip,
    _person_height,
    build_pose_report,
    replay_clip,
    roc_auc,
)
from sentry_ai.eval.poselift import _to_kp17, load_clip, load_split


def _standing_pose() -> np.ndarray:
    """A plausible upright COCO-17 pose (x, y, conf), conf all high."""
    xy = [
        (50, 10),
        (47, 8),
        (53, 8),
        (44, 9),
        (56, 9),  # head
        (40, 30),
        (60, 30),
        (35, 55),
        (65, 55),
        (32, 78),
        (68, 78),  # arms
        (43, 75),
        (57, 75),
        (42, 110),
        (58, 110),
        (41, 150),
        (59, 150),  # legs
    ]
    return np.array([[x, y, 1.0] for x, y in xy], dtype=np.float32)


# --- ROC-AUC ---------------------------------------------------------------


def test_roc_auc_perfect_separation():
    assert roc_auc(np.array([1, 2, 3, 4.0]), np.array([0, 0, 1, 1])) == 1.0


def test_roc_auc_reversed():
    assert roc_auc(np.array([1, 2, 3, 4.0]), np.array([1, 1, 0, 0])) == 0.0


def test_roc_auc_ties_are_half():
    # Two tied pairs split across classes → no separating power.
    assert roc_auc(np.array([1, 1, 2, 2.0]), np.array([0, 1, 0, 1])) == 0.5


def test_roc_auc_single_class_is_none():
    assert roc_auc(np.array([1, 2, 3.0]), np.array([1, 1, 1])) is None
    assert roc_auc(np.array([1, 2, 3.0]), np.array([0, 0, 0])) is None


# --- keypoint coercion -----------------------------------------------------


def test_to_kp17_from_flat_51():
    kp = _to_kp17(np.arange(51, dtype=np.float32))
    assert kp is not None and kp.shape == (17, 3)


def test_to_kp17_from_flat_34_pads_conf():
    kp = _to_kp17(np.arange(34, dtype=np.float32))
    assert kp is not None and kp.shape == (17, 3)
    assert np.all(kp[:, 2] == 1.0)


def test_to_kp17_passthrough_and_reject():
    assert _to_kp17(np.zeros((17, 3), dtype=np.float32)).shape == (17, 3)
    assert _to_kp17(np.zeros((17, 2), dtype=np.float32)).shape == (17, 3)
    assert _to_kp17(np.zeros(10, dtype=np.float32)) is None


# --- person height ---------------------------------------------------------


def test_person_height_prefers_bbox():
    fk = FrameKP(0, _standing_pose(), bbox=(0, 0, 30, 200))
    assert _person_height(fk) == 200.0


def test_person_height_falls_back_to_kp_span():
    fk = FrameKP(0, _standing_pose(), bbox=None)
    # span = ymax(150) - ymin(8) = 142
    assert _person_height(fk) == pytest.approx(142.0)


# --- PoseClip label --------------------------------------------------------


def test_clip_label_from_frame_labels():
    p = PersonTrack("1", [FrameKP(i, _standing_pose()) for i in range(5)])
    theft = PoseClip("a", [p], n_frames=5, frame_labels=np.array([0, 0, 1, 0, 0]))
    benign = PoseClip("b", [p], n_frames=5, frame_labels=np.zeros(5, dtype=int))
    assert theft.label == "theft"
    assert benign.label == "benign"


# --- replay alignment ------------------------------------------------------


def test_replay_clip_alignment():
    track = PersonTrack(
        "1", [FrameKP(i, _standing_pose(), bbox=(0, 0, 30, 200)) for i in range(20)]
    )
    clip = PoseClip("c", [track], n_frames=20, frame_labels=np.zeros(20, dtype=int))
    res = replay_clip(clip, fps=15.0)
    assert res.frame_scores.shape == (20,)
    assert res.frame_labels.shape == (20,)
    assert res.peak == pytest.approx(float(res.frame_scores.max()))
    assert res.label == "benign"


# --- report builder (deterministic, bypasses the engine) -------------------


def _cr(name: str, label: str, peak: float, n: int = 10) -> ClipResult:
    scores = np.full(n, peak, dtype=np.float32)
    labels = np.ones(n, dtype=int) if label == "theft" else np.zeros(n, dtype=int)
    return ClipResult(name=name, label=label, frame_scores=scores, peak=peak, frame_labels=labels)


def test_build_pose_report_perfect_separation():
    results = [
        _cr("t1", "theft", 60.0),
        _cr("t2", "theft", 80.0),
        _cr("b1", "benign", 5.0),
        _cr("b2", "benign", 9.0),
    ]
    report = build_pose_report(results)
    assert report["n_clips"] == 4
    assert report["n_theft"] == 2
    assert report["peak_auc"] == 1.0
    # At the CRITICAL cutoff (50) the two high-peak theft clips separate cleanly.
    at50 = report["at_engine_cutoff"]["50"]
    assert at50["precision"] == 1.0
    assert at50["recall"] == 1.0
    assert report["best_f1"]["f1"] == 1.0


def test_build_pose_report_keys_present():
    report = build_pose_report([_cr("t", "theft", 30.0), _cr("b", "benign", 4.0)])
    for key in ("n_clips", "peak_auc", "frame_auc", "best_f1", "at_engine_cutoff", "sweep", "rows"):
        assert key in report


# --- loader round-trip (no GDrive needed) ----------------------------------


def test_load_clip_roundtrip(tmp_path: Path):
    """A real-format {frame_idx: {person_id: [bbox, keypoints]}} pickle + .npy
    loads cleanly: frame→person nesting is transposed, bbox xyxy→xywh."""
    pose = _standing_pose()  # (17, 3)
    bbox = [10, 20, 40, 220]  # [x1, y1, x2, y2]
    data: dict = {i: {"1": [bbox, pose]} for i in range(6)}
    data[0]["2"] = [bbox, pose]  # a second person on frame 0
    pkl = tmp_path / "cam1_001.pkl"
    with pkl.open("wb") as f:
        pickle.dump(data, f)
    np.save(tmp_path / "cam1_001.npy", np.array([0, 0, 1, 1, 0, 0]))

    clip = load_clip(pkl)
    assert clip.name == "cam1_001"
    assert clip.n_frames == 6  # from the .npy length
    assert len(clip.persons) == 2  # transposed: persons "1" and "2"
    assert clip.label == "theft"  # the .npy has anomalous frames
    assert all(fk.kp.shape == (17, 3) for fk in clip.persons[0].frames)
    # bbox converted [x1,y1,x2,y2] → (x, y, w, h)
    assert clip.persons[0].frames[0].bbox == (10.0, 20.0, 30.0, 200.0)


def test_load_split_separates_by_label_presence(tmp_path: Path):
    """A clip with a matching .npy → TEST (labelled); without → TRAIN (normal).
    Labels match by normalised (cam, vid), so zero-padding differences are fine."""
    pose = _standing_pose()
    frame = {"1": [[0, 0, 30, 200], pose]}
    for name in ("1_222", "1_240"):  # two clips
        with (tmp_path / f"{name}.pkl").open("wb") as f:
            pickle.dump(dict.fromkeys(range(5), frame), f)
    # Only 1_222 gets a label file — and it's zero-padded differently (01_0222).
    np.save(tmp_path / "01_0222.npy", np.array([0, 1, 1, 0, 0]))

    train, test = load_split(tmp_path)
    assert {c.name for c in test} == {"1_222"}  # labelled → test
    assert {c.name for c in train} == {"1_240"}  # no label → train (normal)
    assert test[0].frame_labels is not None
    assert train[0].frame_labels is None
