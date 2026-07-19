"""Cross-camera person re-identification + per-store score accumulation.

ADR-0022 (accumulate one person's risk across cameras) + ADR-0023 (a store's
cameras all run on ONE node → the registry is an in-process, store-scoped
structure; no central DB needed). It matches appearance embeddings across a
store's cameras within a time window and accumulates each person's risk so a
thief building suspicion across aisles is caught at any camera.

The `Embedder` is pluggable and crops to the **torso** (shoulder→hip) before
embedding — using pose keypoints when available, a body-proportion fallback
otherwise — so background, legs, and neighbouring people don't pollute the
appearance vector.

- `HistogramEmbedder` (default) is dependency-light: a two-band Hue+Saturation
  HSV histogram (lighting-tolerant, far stronger than a raw RGB histogram but
  still confused by identical clothing).
- `OSNetEmbedder` (set `REID_MODEL=osnet`, needs the optional `torchreid` extra)
  is a learned model that distinguishes people in similar clothing — the right
  choice for production accuracy.

`StorePersonRegistry` is deliberately a small API so a shared backend (Redis /
pgvector) can replace the in-memory impl for the rare store too large for a
single GPU.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from sentry_ai.logging_setup import get_logger

log = get_logger("sentry_ai.live_worker.reid")

Box = tuple[float, float, float, float]

# COCO-17 keypoint indices used to localise the torso.
_L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP = 5, 6, 11, 12
_MIN_KP_CONF = 0.3

# Hue carries clothing-colour identity; saturation is secondary and noisier under
# lighting changes, so it is down-weighted in the histogram embedding.
_SAT_WEIGHT = 0.5

# Cross-camera score decay (half-life, seconds). The store score accumulates only
# positive per-frame increments, so WITHOUT decay it is a lifetime ratchet: anyone
# present long enough — even just looking around occasionally — climbs to and pins
# at 100, making store_risk_pct meaningless (all green boxes show "100%"). Decaying
# it toward 0 makes it reflect RECENT cross-camera suspicion: it halves every
# STORE_SCORE_HALFLIFE_SEC with no new evidence, so calm fades but sustained
# suspicious activity stays high.
STORE_SCORE_HALFLIFE_SEC = 90.0


def _torso_box(box: Box, keypoints: NDArray[np.float32] | None) -> Box:
    """Return the torso region (shoulder→hip) for a person box.

    Uses the four shoulder/hip keypoints when at least three are confidently
    detected; otherwise falls back to a body-proportion estimate (central width,
    upper ~60% of the body). Coordinates are in frame pixels, unclipped.
    """
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    fallback: Box = (x1 + 0.12 * bw, y1 + 0.12 * bh, x2 - 0.12 * bw, y1 + 0.60 * bh)

    if keypoints is None:
        return fallback
    shape = getattr(keypoints, "shape", None)
    if shape is None or len(shape) < 2 or shape[0] <= _R_HIP:
        return fallback

    has_conf = shape[1] >= 3
    pts: list[tuple[float, float]] = []
    for idx in (_L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP):
        kp = keypoints[idx]
        x, y = float(kp[0]), float(kp[1])
        conf = float(kp[2]) if has_conf else 1.0
        if conf >= _MIN_KP_CONF and (x > 0.0 or y > 0.0):
            pts.append((x, y))
    if len(pts) < 3:
        return fallback

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    tx1, tx2 = min(xs), max(xs)
    ty1, ty2 = min(ys), max(ys)
    pad_x = 0.15 * max(1.0, tx2 - tx1)
    pad_y = 0.10 * max(1.0, ty2 - ty1)
    return (tx1 - pad_x, ty1 - pad_y, tx2 + pad_x, ty2 + pad_y)


def _torso_crop(
    frame_bgr: NDArray[np.uint8], box: Box, keypoints: NDArray[np.float32] | None
) -> NDArray[np.uint8] | None:
    """Slice the torso region out of the frame, clipped to bounds. None if empty."""
    h, w = frame_bgr.shape[:2]
    tx1, ty1, tx2, ty2 = _torso_box(box, keypoints)
    cx1, cy1 = max(0, int(tx1)), max(0, int(ty1))
    cx2, cy2 = min(w, int(tx2)), min(h, int(ty2))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    crop = frame_bgr[cy1:cy2, cx1:cx2]
    return crop if crop.size else None


class Embedder(Protocol):
    """Produces an L2-normalized appearance vector for a person's torso."""

    def embed(
        self,
        frame_bgr: NDArray[np.uint8],
        box: Box,
        keypoints: NDArray[np.float32] | None = None,
    ) -> NDArray[np.float32] | None: ...


class HistogramEmbedder:
    """Two-band Hue+Saturation HSV histogram of the torso (numpy + cv2).

    Splits the torso crop into an upper and lower band (so a shirt and a
    belt/trousers contribute separately) and histograms Hue + Saturation per
    band. Value (brightness) is deliberately dropped to stay lighting-tolerant.
    Stronger than a raw RGB histogram, but still confuses identical clothing —
    set `REID_MODEL=osnet` for production accuracy.
    """

    def __init__(self, bins: int = 8) -> None:
        self.bins = max(2, bins)

    def embed(
        self,
        frame_bgr: NDArray[np.uint8],
        box: Box,
        keypoints: NDArray[np.float32] | None = None,
    ) -> NDArray[np.float32] | None:
        import cv2  # noqa: PLC0415 — cv2 is a hard live-worker dep, imported lazily

        crop = _torso_crop(frame_bgr, box, keypoints)
        if crop is None:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        ch = hsv.shape[0]
        mid = max(1, ch // 2)
        bands = (hsv[:mid], hsv[mid:] if mid < ch else hsv[:mid])

        parts: list[NDArray[np.float32]] = []
        for band in bands:
            hue = cv2.calcHist([band], [0], None, [self.bins * 2], [0, 180]).flatten()
            sat = cv2.calcHist([band], [1], None, [self.bins], [0, 256]).flatten()
            parts.append(hue.astype(np.float32))
            parts.append((sat * _SAT_WEIGHT).astype(np.float32))

        vec = np.concatenate(parts).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            return None
        return vec / norm


class OSNetEmbedder:
    """Learned re-ID embedding via torchreid's OSNet (#4).

    Far more robust than the colour histogram — distinguishes people in similar
    clothing. Requires the optional `torchreid` + `torch` deps and (ideally) a
    GPU. Construction raises if torchreid is unavailable, so `make_embedder`
    falls back to the histogram. Produces an L2-normalized feature vector of the
    torso crop.
    """

    def __init__(self, model_name: str = "osnet_x0_25", device: str | None = None) -> None:
        # Lazy heavy imports — raise so the factory can fall back gracefully.
        import torch  # noqa: PLC0415
        from torchreid.utils import FeatureExtractor  # noqa: PLC0415

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._extractor = FeatureExtractor(model_name=model_name, device=dev)
        log.info("reid.osnet_loaded", model=model_name, device=dev)

    def embed(
        self,
        frame_bgr: NDArray[np.uint8],
        box: Box,
        keypoints: NDArray[np.float32] | None = None,
    ) -> NDArray[np.float32] | None:
        import cv2  # noqa: PLC0415

        crop = _torso_crop(frame_bgr, box, keypoints)
        if crop is None:
            return None
        # torchreid expects RGB HWC; FeatureExtractor accepts a list of ndarrays.
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        raw = self._extractor([rgb]).cpu().numpy()[0]
        feats: NDArray[np.float32] = np.asarray(raw, dtype=np.float32)
        norm = float(np.linalg.norm(feats))
        if norm == 0.0:
            return None
        return feats / norm


def make_embedder(name: str = "histogram") -> Embedder:
    """Resolve a re-ID embedder by name, falling back to the histogram if the
    requested learned model can't be loaded (missing deps / no GPU)."""
    key = (name or "histogram").lower()
    if key in ("histogram", "hist", ""):
        return HistogramEmbedder()
    if key in ("osnet", "torchreid"):
        try:
            return OSNetEmbedder()
        except Exception as e:  # noqa: BLE001 — any import/load failure → fallback
            log.warning("reid.osnet_unavailable_fallback_histogram", error=str(e))
            return HistogramEmbedder()
    log.warning("reid.unknown_embedder_fallback_histogram", requested=key)
    return HistogramEmbedder()


def cosine(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Cosine similarity of two same-length vectors (0 if either is degenerate)."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0 or a.shape != b.shape:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass(slots=True)
class StorePerson:
    person_id: int
    embedding: NDArray[np.float32]
    gallery: list[NDArray[np.float32]] = field(default_factory=list)
    score: float = 0.0
    # Separate from last_seen: match_or_create() refreshes last_seen every frame,
    # so decay must time off its OWN clock (last score mutation), not last_seen.
    score_ts: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    last_camera: str = ""  # most-recent camera (drives spatial-temporal gating)
    cameras: set[str] = field(default_factory=set)
    # Staff identification (live_worker/staff.py): once a badge match locks on ANY
    # camera, the store-global person stays staff — turning their back or walking
    # to another camera doesn't reset it. Cleared naturally when the person is
    # pruned after window_sec of absence (badge re-detects on return).
    is_staff: bool = False


class StorePersonRegistry:
    """In-process, store-scoped re-ID + score accumulation. Thread-safe.

    Matching combines **appearance** with **spatial-temporal** logic, which is the
    real robustness lever when appearance is ambiguous (e.g. everyone in similar
    clothing on overhead/fisheye cameras):

    - **Mutual exclusion:** one physical person can't be in two places at once, so
      a person currently active on a *different* camera (seen within ``coexist_sec``)
      is excluded as a candidate. This stops two simultaneously-visible people on
      different cameras from collapsing into one id.
    - **Same-camera re-entry:** someone who just stepped out of *this* camera's view
      is matched back on a lower appearance threshold (``same_cam_threshold``) —
      there's no competing candidate, so weak appearance evidence is enough.
    - **Gallery:** each person keeps a small gallery of recent embeddings; a
      candidate is scored by the best (max) similarity over it — robust to drift.

    Assumes cameras have **mostly non-overlapping** fields of view (true for most
    stores). One instance per store (the node holds a dict[store_id → registry]).
    """

    def __init__(
        self,
        match_threshold: float = 0.6,
        window_sec: float = 1800.0,
        ema: float = 0.9,
        gallery_size: int = 8,
        coexist_sec: float = 1.0,
        same_cam_threshold: float = 0.45,
        score_halflife_sec: float = STORE_SCORE_HALFLIFE_SEC,
    ) -> None:
        self.match_threshold = match_threshold
        self.window_sec = window_sec
        self.ema = ema
        self.gallery_size = max(1, gallery_size)
        # Half-life for the cross-camera score decay (0 → disable, pure ratchet).
        self.score_halflife_sec = max(0.0, score_halflife_sec)
        # Spatial-temporal params.
        self.coexist_sec = coexist_sec  # active-elsewhere window → mutual exclusion
        self.same_cam_threshold = same_cam_threshold  # lenient re-entry threshold
        self._people: dict[int, StorePerson] = {}
        self._next_id = 1
        self._lock = threading.Lock()

    def _appearance_sim(self, embedding: NDArray[np.float32], p: StorePerson) -> float:
        """Best similarity of an embedding to a person (centroid + gallery)."""
        sim = cosine(embedding, p.embedding)
        for g in p.gallery:
            s = cosine(embedding, g)
            if s > sim:
                sim = s
        return sim

    def match_or_create(
        self, embedding: NDArray[np.float32], camera_id: str, now: float | None = None
    ) -> int:
        """Return the store-global person id for this embedding, creating one if no
        eligible existing person is a good enough match."""
        ts = time.time() if now is None else now
        with self._lock:
            self._prune(ts)
            best_id = -1
            best_sim = -1.0
            for pid, p in self._people.items():
                # Mutual exclusion: a person active on a DIFFERENT camera right now
                # is physically there — this detection can't be them.
                if (
                    p.last_camera
                    and p.last_camera != camera_id
                    and (ts - p.last_seen) < self.coexist_sec
                ):
                    continue
                sim = self._appearance_sim(embedding, p)
                # Same-camera re-entry tolerates weaker appearance evidence.
                threshold = (
                    self.same_cam_threshold if p.last_camera == camera_id else self.match_threshold
                )
                if sim >= threshold and sim > best_sim:
                    best_sim = sim
                    best_id = pid
            if best_id != -1:
                p = self._people[best_id]
                # EMA-update the centroid (tracks slow drift) + refresh the gallery.
                p.embedding = (self.ema * p.embedding + (1.0 - self.ema) * embedding).astype(
                    np.float32
                )
                p.gallery.append(embedding.astype(np.float32))
                if len(p.gallery) > self.gallery_size:
                    p.gallery.pop(0)
                p.last_seen = ts
                p.last_camera = camera_id
                p.cameras.add(camera_id)
                return best_id
            pid = self._next_id
            self._next_id += 1
            self._people[pid] = StorePerson(
                person_id=pid,
                embedding=embedding.astype(np.float32),
                gallery=[embedding.astype(np.float32)],
                last_seen=ts,
                last_camera=camera_id,
                cameras={camera_id},
            )
            return pid

    def add_score(self, person_id: int, delta: float, now: float | None = None) -> float:
        """Add to a person's cross-camera score (time-decayed), return the new total.

        The score halves every ``score_halflife_sec`` of elapsed wall-time before
        the new increment is added, so it tracks RECENT suspicion instead of
        ratcheting to 100 over a person's whole visit. ``score_halflife_sec=0``
        disables decay (legacy pure-accumulate).
        """
        ts = time.time() if now is None else now
        with self._lock:
            p = self._people.get(person_id)
            if p is None:
                return 0.0
            if self.score_halflife_sec > 0.0 and p.score > 0.0:
                elapsed = max(0.0, ts - p.score_ts)
                if elapsed > 0.0:
                    p.score *= 0.5 ** (elapsed / self.score_halflife_sec)
            p.score += delta
            p.score_ts = ts
            p.last_seen = ts
            return p.score

    def mark_staff(self, person_id: int) -> None:
        """Latch the store-global person as staff (badge vote locked on a camera)."""
        with self._lock:
            p = self._people.get(person_id)
            if p is not None:
                p.is_staff = True

    def is_staff(self, person_id: int) -> bool:
        with self._lock:
            p = self._people.get(person_id)
            return p.is_staff if p is not None else False

    def get_score(self, person_id: int) -> float:
        with self._lock:
            p = self._people.get(person_id)
            return p.score if p else 0.0

    def camera_count(self, person_id: int) -> int:
        with self._lock:
            p = self._people.get(person_id)
            return len(p.cameras) if p else 0

    def prune(self, now: float | None = None) -> int:
        ts = time.time() if now is None else now
        with self._lock:
            return self._prune(ts)

    def _prune(self, ts: float) -> int:
        stale = [pid for pid, p in self._people.items() if ts - p.last_seen > self.window_sec]
        for pid in stale:
            del self._people[pid]
        return len(stale)

    def __len__(self) -> int:
        with self._lock:
            return len(self._people)
