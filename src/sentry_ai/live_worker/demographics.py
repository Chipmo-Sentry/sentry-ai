"""Per-track visitor gender/age classification for /insights demographics (docs/30 F5).

Feeds the OPTIONAL `TrackPayload.gender` / `TrackPayload.age_band` fields the
backend footfall aggregator counts into hourly (gender, age_band) slices. The
backend dedups per (camera, person_id) **first-wins**, so this module emits
NOTHING for a track until its label is stable (>= min_votes agreeing samples) —
a premature "unknown" would permanently mis-bucket the visitor.

Model stack — deliberately AGPL-clean (docs/24 flags Ultralytics AGPL as a
launch blocker; the same review applies here):
  - Face detection: YuNet (opencv_zoo, **MIT**), run via cv2.FaceDetectorYN.
  - Age + gender: Levi-Hassner GoogleNets from onnx/models (**Apache-2.0**),
    run via cv2.dnn.
  - NOT InsightFace buffalo_l — its pretrained weights are licensed for
    non-commercial research only, unusable for a commercial SaaS.

Everything runs on **CPU** through OpenCV (no onnxruntime, no new deps), so the
GPU budget next to YOLO+VLM is untouched. Cost is bounded three ways: a
per-track attempt cadence (every N analyzed frames), a per-frame attempt cap,
and a per-track vote lock (a person's attributes don't change — after max_votes
the track is never classified again).

Privacy: head/face crops exist only as locals inside classify(); they are never
written to disk, logged, or transmitted. Only the closed-vocabulary labels
leave this module.

Model files are downloaded once into `DEMOGRAPHICS_MODEL_DIR` with pinned
SHA-256 hashes (Phase-0 supply-chain rule) and loaded from there afterwards.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from sentry_ai.logging_setup import get_logger

log = get_logger("sentry_ai.live_worker.demographics")

Box = tuple[float, float, float, float]
Gender = Literal["male", "female"]
AgeBand = Literal["child", "youth", "adult", "senior"]

# Pinned model artifacts (URL + SHA-256, verified 2026-07-10).
_MODELS: dict[str, tuple[str, str]] = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "age_googlenet.onnx": (
        "https://github.com/onnx/models/raw/main/validated/vision/"
        "body_analysis/age_gender/models/age_googlenet.onnx",
        "fa2a3228e425056aa2b080b3afd3cf607327c86616e952602ed67b5fc16ab356",
    ),
    "gender_googlenet.onnx": (
        "https://github.com/onnx/models/raw/main/validated/vision/"
        "body_analysis/age_gender/models/gender_googlenet.onnx",
        "af24a4eaa9eaf70913cc9a337a0387c86f11549cbd9bbc16bffeefcdcf88cbf4",
    ),
}

# Adience buckets emitted by age_googlenet, and their /insights band. "youth"
# covers the teen bucket; 25+ is "adult" until the 60+ bucket ("senior").
_AGE_BUCKET_BAND: tuple[AgeBand, ...] = (
    "child",  # (0-2)
    "child",  # (4-6)
    "child",  # (8-12)
    "youth",  # (15-20)
    "adult",  # (25-32)
    "adult",  # (38-43)
    "adult",  # (48-53)
    "senior",  # (60-100)
)
_BANDS: tuple[AgeBand, ...] = ("child", "youth", "adult", "senior")
_GENDERS: tuple[Gender, ...] = ("male", "female")  # gender_googlenet output order

# Head-region heuristic: the face lives in the top fraction of the person box.
# Full box width (heads tilt/turn); a little horizontal pad for edge crops.
# 0.45 (was 0.35): overhead/angled store cameras put the face lower in the box,
# so a taller head region catches faces the tighter crop missed.
_HEAD_FRAC = 0.45
_HEAD_PAD_FRAC = 0.10
# YuNet input is capped at this max dimension — detection runs on the small
# image, classification crops from the full-res one.
_DET_MAX_DIM = 320
# 0.6 (was 0.7): store faces are often small/angled/dim — a slightly lower
# detection floor recovers many visitors who were left "unknown". The
# probability-weighted vote average absorbs the occasional weaker read.
_FACE_SCORE_MIN = 0.6
# Margin added around the detected face before the 224x224 classifier resize.
_FACE_MARGIN_FRAC = 0.15
_CLS_INPUT = (224, 224)
_CLS_MEAN = (104.0, 117.0, 123.0)  # BGR mean, per onnx/models age_gender docs

# Forget a track's vote state this long after it was last seen.
_STATE_TTL_SEC = 60.0
# Throttle for the demographics.stats log line.
_STATS_LOG_SEC = 30.0


def ensure_models(model_dir: Path) -> dict[str, Path]:
    """Download-once the three ONNX files into `model_dir`, SHA-256 verified.

    Returns name → path. Raises on download/hash failure so the caller can
    disable demographics rather than run unpinned weights.
    """
    import httpx  # noqa: PLC0415 — only needed on the first-ever run

    model_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for name, (url, sha) in _MODELS.items():
        dest = model_dir / name
        if dest.exists() and _sha256(dest) == sha:
            out[name] = dest
            continue
        log.info("demographics.model_download", model=name, url=url)
        tmp = dest.with_suffix(".part")
        with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        got = _sha256(tmp)
        if got != sha:
            tmp.unlink(missing_ok=True)
            msg = f"SHA-256 mismatch for {name}: expected {sha}, got {got}"
            raise RuntimeError(msg)
        tmp.replace(dest)
        out[name] = dest
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def band_probs_from_age(age_probs: NDArray[np.float32]) -> tuple[float, float, float, float]:
    """Collapse the 8 Adience bucket probabilities into the 4 /insights bands."""
    sums = dict.fromkeys(_BANDS, 0.0)
    for i, band in enumerate(_AGE_BUCKET_BAND):
        if i < age_probs.shape[0]:
            sums[band] += float(age_probs[i])
    return (sums["child"], sums["youth"], sums["adult"], sums["senior"])


@dataclass(frozen=True, slots=True)
class FaceVote:
    """One successful classification sample for a track."""

    gender_probs: tuple[float, float]  # (male, female)
    band_probs: tuple[float, float, float, float]  # (child, youth, adult, senior)
    face_score: float


class Classifier(Protocol):
    """Anything that can turn (frame, person box) into a FaceVote."""

    def classify(self, frame_bgr: NDArray[np.uint8], box: Box) -> FaceVote | None: ...


class FaceAttrClassifier:
    """YuNet face detection + GoogleNet age/gender on the head region of a person box.

    Returns None when no usable face is visible (turned away, too small, too
    far) — the caller simply retries on its next cadence tick.
    """

    def __init__(self, model_dir: Path, min_face_px: int = 24) -> None:
        paths = ensure_models(model_dir)
        self.min_face_px = max(1, min_face_px)
        # Nominal size; setInputSize is called per detect with the real crop size.
        self._detector = cv2.FaceDetectorYN.create(
            str(paths["face_detection_yunet_2023mar.onnx"]),
            "",
            (320, 320),
            _FACE_SCORE_MIN,
            0.3,
            50,
        )
        self._age_net = cv2.dnn.readNetFromONNX(str(paths["age_googlenet.onnx"]))
        self._gender_net = cv2.dnn.readNetFromONNX(str(paths["gender_googlenet.onnx"]))
        log.info("demographics.models_loaded", dir=str(model_dir), min_face_px=self.min_face_px)

    def classify(self, frame_bgr: NDArray[np.uint8], box: Box) -> FaceVote | None:
        head = self._head_crop(frame_bgr, box)
        if head is None:
            return None
        face = self._detect_face(head)
        if face is None:
            return None
        fx1, fy1, fx2, fy2, score = face
        crop = head[fy1:fy2, fx1:fx2]
        if crop.size == 0:
            return None
        blob = cv2.dnn.blobFromImage(crop, 1.0, _CLS_INPUT, _CLS_MEAN, swapRB=False)
        self._gender_net.setInput(blob)
        gender = self._gender_net.forward().flatten()
        self._age_net.setInput(blob)
        age = self._age_net.forward().flatten()
        return FaceVote(
            gender_probs=(float(gender[0]), float(gender[1])),
            band_probs=band_probs_from_age(age.astype(np.float32)),
            face_score=score,
        )

    def _head_crop(self, frame_bgr: NDArray[np.uint8], box: Box) -> NDArray[np.uint8] | None:
        """Top-of-box head region, clipped to the frame. None if degenerate."""
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = box
        bw = x2 - x1
        pad = bw * _HEAD_PAD_FRAC
        cx1 = max(0, int(x1 - pad))
        cx2 = min(w, int(x2 + pad))
        cy1 = max(0, int(y1))
        cy2 = min(h, int(y1 + max(1.0, (y2 - y1)) * _HEAD_FRAC))
        if cx2 - cx1 < self.min_face_px or cy2 - cy1 < self.min_face_px:
            return None
        crop = frame_bgr[cy1:cy2, cx1:cx2]
        return crop if crop.size else None

    def _detect_face(self, head_bgr: NDArray[np.uint8]) -> tuple[int, int, int, int, float] | None:
        """Best face in the head crop → (x1, y1, x2, y2, score) in crop pixels.

        Detection runs on a <=_DET_MAX_DIM copy for speed; coordinates are
        mapped back so the classifier crops from the full-res region.
        """
        ch, cw = head_bgr.shape[:2]
        scale = min(1.0, _DET_MAX_DIM / max(ch, cw))
        det_img = (
            cv2.resize(head_bgr, (max(1, int(cw * scale)), max(1, int(ch * scale))))
            if scale < 1.0
            else head_bgr
        )
        dh, dw = det_img.shape[:2]
        self._detector.setInputSize((dw, dh))
        _, faces = self._detector.detect(det_img)
        if faces is None or len(faces) == 0:
            return None
        best = max(faces, key=lambda f: float(f[14]))
        score = float(best[14])
        x, y, fw, fh = (float(v) / scale for v in best[:4])
        if min(fw, fh) < self.min_face_px:
            return None
        # Margin around the face; clip to the head crop.
        mx, my = fw * _FACE_MARGIN_FRAC, fh * _FACE_MARGIN_FRAC
        x1 = max(0, int(x - mx))
        y1 = max(0, int(y - my))
        x2 = min(cw, int(x + fw + mx))
        y2 = min(ch, int(y + fh + my))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2, score)


class _TrackLike(Protocol):
    """The slice of TrackedDetection this module needs (keeps imports light)."""

    @property
    def tracker_id(self) -> int: ...
    @property
    def box(self) -> Box: ...


@dataclass(slots=True)
class _TrackState:
    gender_sum: list[float] = field(default_factory=lambda: [0.0, 0.0])
    band_sum: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    votes: int = 0
    last_attempt_frame: int = -(10**9)  # first sighting is immediately eligible
    last_seen: float = 0.0


class TrackDemographics:
    """Per-track cadence gating + weighted-vote cache over a `Classifier`.

    observe() is called once per analyzed frame from the camera worker thread
    (single-threaded per instance — no locking needed) and returns
    tracker_id → (gender, age_band), both None until the track's label is
    stable. Labels are probability-weighted sums over the collected votes, so
    one blurry misread doesn't flip a person's bucket.
    """

    def __init__(
        self,
        classifier: Classifier,
        every_n: int = 10,
        min_votes: int = 2,
        max_votes: int = 5,
        max_per_frame: int = 2,
    ) -> None:
        self._classifier = classifier
        self.every_n = max(1, every_n)
        self.min_votes = max(1, min_votes)
        self.max_votes = max(self.min_votes, max_votes)
        self.max_per_frame = max(1, max_per_frame)
        self._states: dict[int, _TrackState] = {}
        # Rolling cost/coverage counters for the throttled stats log.
        self._attempts = 0
        self._hits = 0
        self._ms_sum = 0.0
        self._last_stats_log = time.monotonic()

    def observe(
        self,
        frame_bgr: NDArray[np.uint8],
        tracked: Sequence[_TrackLike],
        frame_idx: int,
    ) -> dict[int, tuple[Gender | None, AgeBand | None]]:
        """Classify due tracks (bounded) and return current labels for ALL tracks."""
        now = time.monotonic()
        out: dict[int, tuple[Gender | None, AgeBand | None]] = {}
        # Oldest-attempt-first so a busy frame round-robins fairly across tracks.
        due: list[tuple[int, _TrackState, Box]] = []
        for t in tracked:
            st = self._states.get(t.tracker_id)
            if st is None:
                st = _TrackState()
                self._states[t.tracker_id] = st
            st.last_seen = now
            if st.votes < self.max_votes and frame_idx - st.last_attempt_frame >= self.every_n:
                due.append((t.tracker_id, st, t.box))
        due.sort(key=lambda d: d[1].last_attempt_frame)

        for _tid, st, box in due[: self.max_per_frame]:
            st.last_attempt_frame = frame_idx
            t0 = time.perf_counter()
            vote = self._classifier.classify(frame_bgr, box)
            self._ms_sum += (time.perf_counter() - t0) * 1000.0
            self._attempts += 1
            if vote is not None:
                self._hits += 1
                st.gender_sum[0] += vote.gender_probs[0]
                st.gender_sum[1] += vote.gender_probs[1]
                for i in range(4):
                    st.band_sum[i] += vote.band_probs[i]
                st.votes += 1

        for t in tracked:
            out[t.tracker_id] = self._labels(self._states[t.tracker_id])

        self._log_stats(now)
        return out

    def _labels(self, st: _TrackState) -> tuple[Gender | None, AgeBand | None]:
        if st.votes < self.min_votes:
            return (None, None)
        gender = _GENDERS[0] if st.gender_sum[0] >= st.gender_sum[1] else _GENDERS[1]
        band = _BANDS[max(range(4), key=lambda i: st.band_sum[i])]
        return (gender, band)

    def prune(self, now: float | None = None) -> None:
        """Drop vote state for tracks idle past the TTL (called from the worker's
        periodic cleanup, alongside its other per-track side-tables)."""
        ts = time.monotonic() if now is None else now
        for tid in [k for k, s in self._states.items() if ts - s.last_seen > _STATE_TTL_SEC]:
            del self._states[tid]

    def _log_stats(self, now: float) -> None:
        if now - self._last_stats_log < _STATS_LOG_SEC or self._attempts == 0:
            return
        self._last_stats_log = now
        labeled = sum(1 for s in self._states.values() if s.votes >= self.min_votes)
        log.info(
            "demographics.stats",
            attempts=self._attempts,
            face_hits=self._hits,
            avg_ms=round(self._ms_sum / self._attempts, 1),
            labeled_tracks=labeled,
            tracked_states=len(self._states),
        )
        self._attempts = 0
        self._hits = 0
        self._ms_sum = 0.0


def make_track_demographics() -> TrackDemographics | None:
    """Build the per-camera vote cache from settings, or None when disabled or
    the models can't be provisioned (node keeps running without demographics)."""
    from sentry_ai.settings import get_settings  # noqa: PLC0415 — avoid import cycle

    s = get_settings()
    if not s.demographics_enabled:
        return None
    try:
        classifier = FaceAttrClassifier(
            Path(s.demographics_model_dir), min_face_px=s.demographics_min_face_px
        )
    except Exception as e:  # noqa: BLE001 — download/load failure must not kill the worker
        log.warning("demographics.unavailable", error=str(e))
        return None
    return TrackDemographics(
        classifier,
        every_n=s.demographics_every_n,
        min_votes=s.demographics_min_votes,
        max_votes=s.demographics_max_votes,
        max_per_frame=s.demographics_max_per_frame,
    )
