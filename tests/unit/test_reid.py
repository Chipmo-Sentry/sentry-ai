"""Cross-camera re-ID registry + histogram embedder (ADR-0022/0023)."""

from __future__ import annotations

import numpy as np

from sentry_ai.live_worker.camera_worker import _reid_quality_ok
from sentry_ai.live_worker.reid import (
    HistogramEmbedder,
    StorePersonRegistry,
    _torso_box,
    cosine,
    make_embedder,
)


def _solid(color: tuple[int, int, int], h: int = 40, w: int = 20) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = color
    return img


def test_make_embedder_histogram_default() -> None:
    assert isinstance(make_embedder("histogram"), HistogramEmbedder)
    assert isinstance(make_embedder(""), HistogramEmbedder)


def test_make_embedder_unknown_falls_back() -> None:
    assert isinstance(make_embedder("does-not-exist"), HistogramEmbedder)


def test_make_embedder_osnet_falls_back_when_unavailable() -> None:
    # torchreid isn't installed in CI → must fall back to a working embedder.
    emb = make_embedder("osnet")
    vec = emb.embed(_solid((0, 0, 200)), (0, 0, 20, 40))
    assert vec is not None and abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5


def test_embedder_normalized_and_distinguishes_colors() -> None:
    emb = HistogramEmbedder(bins=8)
    red = emb.embed(_solid((0, 0, 200)), (0, 0, 20, 40))
    blue = emb.embed(_solid((200, 0, 0)), (0, 0, 20, 40))
    assert red is not None and blue is not None
    assert abs(float(np.linalg.norm(red)) - 1.0) < 1e-5
    # Different dominant colors → low similarity.
    assert cosine(red, blue) < 0.5


def test_embedder_same_color_high_similarity() -> None:
    emb = HistogramEmbedder(bins=8)
    a = emb.embed(_solid((0, 180, 0)), (0, 0, 20, 40))
    b = emb.embed(_solid((0, 180, 0)), (0, 0, 20, 40))
    assert a is not None and b is not None
    assert cosine(a, b) > 0.99


def test_embedder_handles_empty_box() -> None:
    emb = HistogramEmbedder()
    assert emb.embed(_solid((1, 1, 1)), (5, 5, 5, 5)) is None  # zero-area
    assert emb.embed(_solid((1, 1, 1)), (-10, -10, -1, -1)) is None


def test_registry_matches_same_person_across_cameras() -> None:
    reg = StorePersonRegistry(match_threshold=0.6)
    emb = HistogramEmbedder(bins=8)
    v = emb.embed(_solid((10, 200, 30)), (0, 0, 20, 40))
    assert v is not None
    # Seen on cam A, then again (same look) on cam B → SAME store-person id.
    id_a = reg.match_or_create(v, "camA", now=1000.0)
    id_b = reg.match_or_create(v, "camB", now=1001.0)
    assert id_a == id_b
    assert reg.camera_count(id_a) == 2
    assert len(reg) == 1


def test_registry_separates_distinct_people() -> None:
    reg = StorePersonRegistry(match_threshold=0.6)
    emb = HistogramEmbedder(bins=8)
    red = emb.embed(_solid((0, 0, 200)), (0, 0, 20, 40))
    blue = emb.embed(_solid((200, 0, 0)), (0, 0, 20, 40))
    assert red is not None and blue is not None
    id1 = reg.match_or_create(red, "camA", now=1000.0)
    id2 = reg.match_or_create(blue, "camA", now=1000.0)
    assert id1 != id2
    assert len(reg) == 2


def test_registry_accumulates_score_across_cameras() -> None:
    # Decay disabled (score_halflife_sec=0) so this tests pure cross-camera
    # accumulation without the time-decay confounding the exact total.
    reg = StorePersonRegistry(match_threshold=0.6, score_halflife_sec=0.0)
    emb = HistogramEmbedder(bins=8)
    v = emb.embed(_solid((10, 200, 30)), (0, 0, 20, 40))
    assert v is not None
    pid = reg.match_or_create(v, "camA", now=1000.0)
    reg.add_score(pid, 20.0, now=1000.0)
    # Same person reappears on cam B and adds more — score accumulates.
    reg.match_or_create(v, "camB", now=1001.0)
    total = reg.add_score(pid, 15.0, now=1001.0)
    assert total == 35.0


def test_registry_score_decays_over_time() -> None:
    """The cross-camera score halves each half-life with no new evidence, so it
    can't ratchet to a permanent 100 (the all-green-boxes-show-100% bug)."""
    reg = StorePersonRegistry(match_threshold=0.6, score_halflife_sec=10.0)
    emb = HistogramEmbedder(bins=8)
    v = emb.embed(_solid((10, 200, 30)), (0, 0, 20, 40))
    assert v is not None
    pid = reg.match_or_create(v, "camA", now=0.0)
    reg.add_score(pid, 100.0, now=0.0)  # score = 100 at t=0
    # One half-life later, no new evidence → ~50.
    total = reg.add_score(pid, 0.0, now=10.0)
    assert 49.0 < total < 51.0
    # Two half-lives → ~25. The old suspicion fades instead of pinning at 100.
    total2 = reg.add_score(pid, 0.0, now=20.0)
    assert 24.0 < total2 < 26.0


# === Spatial-temporal gating ===


def test_mutual_exclusion_splits_simultaneous_cameras() -> None:
    """Same look on two cameras AT THE SAME TIME → different people (one body can't
    be in two places). This is the fix for the identical-clothing false-merge."""
    reg = StorePersonRegistry(match_threshold=0.6, coexist_sec=1.0)
    emb = HistogramEmbedder(bins=8)
    v = emb.embed(_solid((10, 10, 10)), (0, 0, 40, 80))  # near-identical "black" look
    assert v is not None
    id_a = reg.match_or_create(v, "camA", now=1000.0)
    id_b = reg.match_or_create(v, "camB", now=1000.2)  # within coexist window
    assert id_a != id_b
    assert len(reg) == 2


def test_same_camera_reentry_keeps_id() -> None:
    """Leave a camera and come back (no competing candidate) → SAME id, even on
    weak appearance evidence (same-camera lenient threshold)."""
    reg = StorePersonRegistry(coexist_sec=1.0, same_cam_threshold=0.45)
    emb = HistogramEmbedder(bins=8)
    v = emb.embed(_solid((10, 200, 30)), (0, 0, 40, 80))
    assert v is not None
    id1 = reg.match_or_create(v, "camA", now=1000.0)
    id2 = reg.match_or_create(v, "camA", now=1004.0)  # stepped out and back
    assert id1 == id2
    assert len(reg) == 1


def test_cross_camera_transition_after_gap_matches() -> None:
    """Move from camA to camB after the coexist window → same id (a real transition)."""
    reg = StorePersonRegistry(match_threshold=0.6, coexist_sec=1.0)
    emb = HistogramEmbedder(bins=8)
    v = emb.embed(_solid((10, 200, 30)), (0, 0, 40, 80))
    assert v is not None
    id_a = reg.match_or_create(v, "camA", now=1000.0)
    id_b = reg.match_or_create(v, "camB", now=1003.0)  # walked over (> coexist)
    assert id_a == id_b
    assert len(reg) == 1


def test_registry_prunes_outside_window() -> None:
    reg = StorePersonRegistry(match_threshold=0.6, window_sec=60.0)
    emb = HistogramEmbedder(bins=8)
    v = emb.embed(_solid((10, 200, 30)), (0, 0, 20, 40))
    assert v is not None
    reg.match_or_create(v, "camA", now=1000.0)
    # A later match past the window prunes the stale person first → new id, len 1.
    reg.match_or_create(v, "camA", now=2000.0)
    assert len(reg) == 1


# === Torso cropping (via keypoints) ===


def test_embed_accepts_keypoints_and_focuses_on_torso() -> None:
    """A red torso on a green background should embed close to pure-red, not green —
    proving the crop follows the keypoints rather than the whole box."""
    frame = _solid((0, 200, 0), h=100, w=60)  # green background (BGR)
    frame[20:60, 18:42] = (0, 0, 200)  # red torso patch
    # COCO-17 keypoints: shoulders (5,6) at y≈22, hips (11,12) at y≈56, inside the patch.
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[5] = (20.0, 22.0, 0.9)
    kp[6] = (40.0, 22.0, 0.9)
    kp[11] = (20.0, 56.0, 0.9)
    kp[12] = (40.0, 56.0, 0.9)

    emb = HistogramEmbedder(bins=8)
    v = emb.embed(frame, (10.0, 0.0, 50.0, 100.0), kp)
    red = emb.embed(_solid((0, 0, 200), h=40, w=24), (0.0, 0.0, 24.0, 40.0))
    green = emb.embed(_solid((0, 200, 0), h=40, w=24), (0.0, 0.0, 24.0, 40.0))
    assert v is not None and red is not None and green is not None
    assert cosine(v, red) > cosine(v, green)


def test_embed_without_keypoints_still_returns_vector() -> None:
    emb = HistogramEmbedder(bins=8)
    v = emb.embed(_solid((0, 0, 200), h=80, w=40), (0.0, 0.0, 40.0, 80.0))
    assert v is not None and abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_torso_box_uses_keypoints_when_available() -> None:
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[5] = (20.0, 22.0, 0.9)
    kp[6] = (40.0, 22.0, 0.9)
    kp[11] = (20.0, 56.0, 0.9)
    kp[12] = (40.0, 56.0, 0.9)
    tx1, ty1, tx2, ty2 = _torso_box((10.0, 0.0, 50.0, 100.0), kp)
    # Should bracket the shoulder/hip span (with padding), well inside the full box.
    assert 10.0 < tx1 < 20.0 and 40.0 < tx2 < 50.0
    assert 0.0 < ty1 < 22.0 and 56.0 < ty2 < 100.0


def test_torso_box_falls_back_without_confident_keypoints() -> None:
    # All keypoints low-confidence → proportion fallback (upper ~60% of the body).
    kp = np.zeros((17, 3), dtype=np.float32)
    box = (0.0, 0.0, 40.0, 100.0)
    tx1, ty1, tx2, ty2 = _torso_box(box, kp)
    assert ty2 < 70.0  # upper body only, not the legs


# === Re-ID quality gate (camera_worker) ===


def test_quality_gate_rejects_small_box() -> None:
    assert _reid_quality_ok((0.0, 0.0, 20.0, 30.0), frame_h=720) is False


def test_quality_gate_accepts_large_box() -> None:
    # Large enough to identify — gate is box-size only (keypoints handled by the
    # torso-crop fallback), so overhead/fisheye people with hidden hips still pass.
    assert _reid_quality_ok((0.0, 0.0, 80.0, 200.0), frame_h=720) is True
