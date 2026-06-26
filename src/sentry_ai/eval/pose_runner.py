"""Replay labelled pose sequences through the Stage-1 behavior engine and score it.

The VLM harness (`runner.py`) measures Stage-2 on *video clips*. This module
measures **Stage-1** — `BehaviorScorer`, the YOLO-pose → behavior engine — on
*skeleton sequences* (no pixels). That's the half a pose-only dataset like
PoseLift can exercise, and it's exactly the regime the founder cares about:
"recognise the behaviour from the YOLO-pose stream".

How it works, per clip (one video = many per-person tracks over time):
  * a synthetic clock advances by 1/fps each frame, so the engine's time-based
    gates (loiter seconds, sequence window, hold-release) behave as they would
    live — NOT at the mercy of how fast we replay.
  * every frame, every present person is fed to `scorer.score(...)`; we keep the
    max `risk_pct` over persons as that frame's anomaly score.
  * clip-level prediction = "theft" if the clip's peak score crosses a threshold.

Two views come out:
  * **video-level binary** (theft vs benign) via `metrics.binary_metrics`, swept
    over thresholds → a precision/recall curve (the engine's own HIGH=25 /
    CRITICAL=50 cutoffs are highlighted).
  * **frame-level ROC-AUC** — directly comparable to the PoseLift paper's
    STG-NF / TSGAD / GEPC numbers (their headline metric).

CAVEAT — items / require_holding: a pose-only dataset has no COCO object
detections, so `items` is always empty. The pocket/bag detectors gate on a held
item (`require_holding`, ai#9), so they will under-fire here. That is an honest
measurement of the pose-only path, and the gap it shows is the whole reason for
the learned skeleton-action layer (ADR-0030).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sentry_ai.eval.metrics import binary_metrics
from sentry_ai.live_worker.behavior import BehaviorScorer

# COCO-17 vertical extent (nose..ankles) used to estimate person height when a
# bbox is missing. Floor so a degenerate frame can't divide by ~0.
_MIN_PERSON_H = 1.0
# The engine's native level cutoffs (ADR-0024) — highlighted on the sweep so the
# baseline says "at the threshold the system actually uses today, recall is X".
ENGINE_CUTOFFS = (10.0, 25.0, 50.0)  # LOW/MEDIUM, MEDIUM/HIGH, HIGH/CRITICAL


@dataclass(slots=True)
class FrameKP:
    """One person's pose at one frame."""

    frame_idx: int
    kp: NDArray[np.float32]  # (17, 3) [x, y, conf]
    bbox: tuple[float, float, float, float] | None = None  # x, y, w, h


@dataclass(slots=True)
class PersonTrack:
    """One tracked person across a clip."""

    person_id: str
    frames: list[FrameKP]


@dataclass(slots=True)
class PoseClip:
    """One labelled clip: its per-person tracks + optional frame-level labels."""

    name: str
    persons: list[PersonTrack]
    n_frames: int
    # Per-frame binary anomaly label (0/1), len == n_frames, or None. PoseLift
    # supplies this (.npy). The clip's video-level label is derived from it.
    frame_labels: NDArray[np.int_] | None = None

    @property
    def label(self) -> str:
        """Video-level ground truth: theft if any frame is flagged anomalous."""
        if self.frame_labels is not None and int(self.frame_labels.sum()) > 0:
            return "theft"
        return "benign"


@dataclass(slots=True)
class ClipResult:
    """Replay output for one clip."""

    name: str
    label: str
    frame_scores: NDArray[np.float32]  # per-frame max risk_pct over persons
    peak: float
    # Per-frame ground-truth (0/1), aligned with frame_scores — copied from the
    # clip so frame-level AUC can be computed without re-threading the dataset.
    frame_labels: NDArray[np.int_]
    fired: list[str] = field(default_factory=list)  # union of behaviors that fired


def _person_height(fk: FrameKP) -> float:
    """Pixel height: bbox h if present, else the valid-keypoint vertical span."""
    if fk.bbox is not None and fk.bbox[3] > 0:
        return float(fk.bbox[3])
    kp = fk.kp
    valid = kp[(kp.shape[1] < 3) | (kp[:, 2] > 0.0)] if kp.shape[1] >= 3 else kp
    if len(valid) >= 2:
        span = float(valid[:, 1].max() - valid[:, 1].min())
        if span > 0:
            return span
    return _MIN_PERSON_H


def replay_clip(
    clip: PoseClip,
    *,
    fps: float = 15.0,
    weights: dict[str, float] | None = None,
    engine_params: dict[str, float] | None = None,
    detector_params: dict[str, dict[str, float]] | None = None,
) -> ClipResult:
    """Feed a clip frame-by-frame through a fresh BehaviorScorer with a synthetic
    clock; return per-frame max risk + the peak + behaviors that fired."""
    t = [0.0]
    scorer = BehaviorScorer(weights, clock=lambda: t[0])
    if engine_params or detector_params:
        scorer.update_params(engine_params, detector_params)

    # frame_idx -> list of (tracker_int, FrameKP)
    by_frame: dict[int, list[tuple[int, FrameKP]]] = {}
    id_map: dict[str, int] = {}
    for track in clip.persons:
        tid = id_map.setdefault(track.person_id, len(id_map))
        for fk in track.frames:
            by_frame.setdefault(fk.frame_idx, []).append((tid, fk))

    n = clip.n_frames
    frame_scores = np.zeros(n, dtype=np.float32)
    fired: set[str] = set()
    dt = 1.0 / fps if fps > 0 else 0.0
    for f in range(n):
        t[0] = f * dt
        best = 0.0
        for tid, fk in by_frame.get(f, ()):
            res = scorer.score(tid, np.asarray(fk.kp, dtype=np.float32), _person_height(fk))
            best = max(best, res.risk_pct)
            fired.update(res.behaviors)
        frame_scores[f] = best
        if f % 15 == 14:
            scorer.cleanup_stale()

    if clip.frame_labels is not None and len(clip.frame_labels) == n:
        frame_labels = np.asarray(clip.frame_labels, dtype=np.int_)
    else:
        frame_labels = np.zeros(n, dtype=np.int_)

    return ClipResult(
        name=clip.name,
        label=clip.label,
        frame_scores=frame_scores,
        peak=float(frame_scores.max()) if n else 0.0,
        frame_labels=frame_labels,
        fired=sorted(fired),
    )


def roc_auc(scores: NDArray[np.float32], labels: NDArray[np.int_]) -> float | None:
    """Rank-based ROC-AUC (Mann–Whitney U) with tie-aware average ranks. Returns
    None when one class is absent (AUC undefined). Pure numpy — no sklearn."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int_)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Average ranks within tied groups so equal scores don't bias the AUC.
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    sum_pos = float(ranks[labels == 1].sum())
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return round(auc, 4)


def _sweep(results: list[ClipResult], thetas: NDArray[np.float64]) -> list[dict[str, Any]]:
    """Video-level binary metrics at each peak-score threshold theta."""
    y_true = [r.label for r in results]
    peaks = [r.peak for r in results]
    rows: list[dict[str, Any]] = []
    for theta in thetas:
        y_pred = ["theft" if p >= theta else "benign" for p in peaks]
        m = binary_metrics(y_true, y_pred)
        rows.append({"theta": round(float(theta), 1), **m})
    return rows


def build_pose_report(results: list[ClipResult]) -> dict[str, Any]:
    """Baseline report: video-level threshold sweep (+ engine-cutoff rows and the
    best-F1 operating point) and frame-level ROC-AUC."""
    n_theft = sum(1 for r in results if r.label == "theft")
    thetas = np.unique(np.concatenate([np.linspace(0, 100, 101), np.asarray(ENGINE_CUTOFFS)]))
    sweep = _sweep(results, thetas)

    scored_f1 = [row for row in sweep if row["f1"] is not None]
    best = max(scored_f1, key=lambda r: r["f1"]) if scored_f1 else None
    at_cutoff = {
        str(int(c)): next((row for row in sweep if abs(row["theta"] - c) < 1e-6), None)
        for c in ENGINE_CUTOFFS
    }

    # Frame-level ROC-AUC: concat every clip's per-frame (score, label). This is
    # the metric the PoseLift paper reports for STG-NF / TSGAD / GEPC, so the
    # baseline is directly comparable.
    frame_auc = None
    if results:
        all_scores = np.concatenate([r.frame_scores for r in results])
        all_labels = np.concatenate([r.frame_labels for r in results])
        frame_auc = roc_auc(all_scores, all_labels)

    peak_auc = roc_auc(
        np.asarray([r.peak for r in results]),
        np.asarray([1 if r.label == "theft" else 0 for r in results]),
    )

    return {
        "n_clips": len(results),
        "n_theft": n_theft,
        "n_benign": len(results) - n_theft,
        "peak_auc": peak_auc,
        "frame_auc": frame_auc,
        "best_f1": best,
        "at_engine_cutoff": at_cutoff,
        "sweep": sweep,
        "rows": [
            {"name": r.name, "label": r.label, "peak": round(r.peak, 1), "fired": r.fired}
            for r in results
        ],
    }


def format_pose_summary(report: dict[str, Any]) -> str:
    lines = [
        f"Clips: {report['n_clips']}  (theft={report['n_theft']} benign={report['n_benign']})",
        f"Peak-score AUC (video-level): {report['peak_auc']}",
        f"Frame-level AUC: {report['frame_auc']}",
        "",
        "At the engine's native cutoffs (peak risk_pct >= cutoff → theft):",
    ]
    for c, row in report["at_engine_cutoff"].items():
        if row:
            lines.append(
                f"  >={c:>3}  precision={row['precision']} recall={row['recall']}"
                f" f1={row['f1']}  tp={row['tp']} fp={row['fp']} fn={row['fn']} tn={row['tn']}"
            )
    b = report["best_f1"]
    if b:
        lines += [
            "",
            f"Best-F1 operating point: theta={b['theta']}"
            f"  precision={b['precision']} recall={b['recall']} f1={b['f1']}",
        ]
    return "\n".join(lines)
