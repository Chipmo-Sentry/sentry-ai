"""End-to-end (CPU, synthetic, fast): train the anomaly autoencoder on NORMAL
pose motion and confirm it (a) reconstructs normal better than anomalous and
(b) scores anomalous frames higher — i.e. frame-level AUC well above chance.

Synthetic design: a fixed upright skeleton with small per-joint jitter is
"normal"; an anomalous frame yanks the wrists far out (a different body SHAPE,
which survives the body-centred normalisation, unlike pure translation)."""

from __future__ import annotations

import numpy as np

from sentry_ai.eval.pose_runner import FrameKP, PersonTrack, PoseClip
from sentry_ai.skeleton.infer import clip_result, evaluate, load_checkpoint, score_clip
from sentry_ai.skeleton.train import TrainConfig, save_checkpoint, train_autoencoder

_BASE = np.array(
    [
        (50, 10),
        (47, 8),
        (53, 8),
        (44, 9),
        (56, 9),
        (40, 30),
        (60, 30),
        (35, 55),
        (65, 55),
        (32, 78),
        (68, 78),
        (43, 75),
        (57, 75),
        (42, 110),
        (58, 110),
        (41, 150),
        (59, 150),
    ],
    dtype=np.float32,
)
_L_WRI, _R_WRI = 9, 10


def _pose(rng: np.random.Generator, *, anomalous: bool = False) -> np.ndarray:
    xy = _BASE + rng.normal(0, 0.6, size=(17, 2)).astype(np.float32)
    if anomalous:
        xy[_L_WRI] += np.array([40.0, -45.0], np.float32)  # wrist flung up/out
        xy[_R_WRI] += np.array([-40.0, -45.0], np.float32)
    return np.concatenate([xy, np.ones((17, 1), np.float32)], axis=1)


def _normal_clip(name: str, n: int, rng: np.random.Generator) -> PoseClip:
    frames = [FrameKP(i, _pose(rng)) for i in range(n)]
    return PoseClip(name, [PersonTrack("p", frames)], n, np.zeros(n, np.int_))


def _anom_clip(name: str, n: int, rng: np.random.Generator) -> PoseClip:
    labels = np.zeros(n, np.int_)
    lo, hi = n // 3, 2 * n // 3
    labels[lo:hi] = 1
    frames = [FrameKP(i, _pose(rng, anomalous=lo <= i < hi)) for i in range(n)]
    return PoseClip(name, [PersonTrack("p", frames)], n, labels)


def _cfg() -> TrainConfig:
    return TrainConfig(length=16, stride=4, epochs=80, batch_size=32, seed=0)


def test_train_then_eval_beats_chance() -> None:
    rng = np.random.default_rng(0)
    train_clips = [_normal_clip(f"n{i}", 60, rng) for i in range(6)]
    model, meta = train_autoencoder(train_clips, _cfg(), device="cpu")
    assert meta["threshold"] > 0.0

    eval_clips = [_normal_clip(f"en{i}", 60, rng) for i in range(3)]
    eval_clips += [_anom_clip(f"ea{i}", 60, rng) for i in range(3)]
    report = evaluate(eval_clips, model, meta, device="cpu")
    assert report["frame_auc"] is not None
    assert report["frame_auc"] > 0.7  # clearly separates anomalous frames


def test_anomalous_frames_score_higher_within_clip() -> None:
    rng = np.random.default_rng(1)
    model, meta = train_autoencoder(
        [_normal_clip(f"n{i}", 60, rng) for i in range(6)], _cfg(), device="cpu"
    )
    clip = _anom_clip("mix", 60, rng)
    scores = score_clip(clip, model, meta, device="cpu")
    labels = clip.frame_labels
    assert scores.shape == (clip.n_frames,)
    assert scores[labels == 1].mean() > scores[labels == 0].mean()


def test_checkpoint_roundtrip(tmp_path) -> None:  # noqa: ANN001 — pytest fixture
    rng = np.random.default_rng(2)
    clips = [_normal_clip(f"n{i}", 50, rng) for i in range(4)]
    model, meta = train_autoencoder(clips, _cfg(), device="cpu")
    path = tmp_path / "m.pt"
    save_checkpoint(model, meta, path)
    loaded, loaded_meta = load_checkpoint(path, device="cpu")
    assert loaded_meta["threshold"] == meta["threshold"]
    clip = _anom_clip("c", 50, rng)
    a = clip_result(clip, model, meta).frame_scores
    b = clip_result(clip, loaded, loaded_meta).frame_scores
    assert np.allclose(a, b, atol=1e-5)
