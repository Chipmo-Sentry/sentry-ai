"""Load the PoseLift dataset (TeCSAR-UNCC, WACV 2025) into `PoseClip`s.

PoseLift is privacy-preserving: skeleton keypoints only, no pixels. The REAL
on-disk layout (verified against the released data, 2026-06-29):

  * ``Pickle_files/Train/<cam>_<vid>.pkl``  — normal-only clips (no labels)
  * ``Pickle_files/Test/<cam>_<vid>.pkl``   — test clips
  * ``Json_files/.../gt/test_frame_mask/<cam0>_<vid0>.npy`` — per-frame 0/1 labels

Each ``.pkl`` is nested ``{ frame_idx: { person_id: [bbox, keypoints] } }`` — the
TOP level is the FRAME, not the person. ``bbox`` is ``[x1, y1, x2, y2]``;
``keypoints`` is COCO-17 stored as ``[y, x, conf]`` (row, col), which we swap to
``[x, y, conf]`` so it shares the bbox convention. The pkl frame index aligns with
the label .npy index. The .pkl and .npy names differ by zero-padding
(``1_222`` ↔ ``01_0222``), so labels are matched by the normalised ``(cam, vid)``.

Download (manual — Google Drive, no programmatic endpoint):
  https://github.com/TeCSAR-UNCC/PoseLift  → keep the folder structure, point the
  loader at the extracted root.
"""

from __future__ import annotations

import pickle  # noqa: S403 — trusted local dataset files, not untrusted input
import re
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sentry_ai.eval.pose_runner import FrameKP, PersonTrack, PoseClip

COCO_17 = 17


def _to_kp17(arr: Any) -> NDArray[np.float32] | None:
    """Coerce one frame's keypoints to (17, 3) [x, y, conf]. Accepts (17,3),
    (17,2), flat 51 or 34. Returns None if it can't. (No coordinate swap — that's
    a PoseLift frame-format detail handled in ``_kp_bbox``.)"""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        if a.size == COCO_17 * 3:
            a = a.reshape(COCO_17, 3)
        elif a.size == COCO_17 * 2:
            a = a.reshape(COCO_17, 2)
        else:
            return None
    if a.ndim != 2 or a.shape[0] != COCO_17:
        return None
    if a.shape[1] == 2:
        a = np.concatenate([a, np.ones((COCO_17, 1), dtype=np.float32)], axis=1)
    return a[:, :3].astype(np.float32)


def _bbox_xywh(arr: Any) -> tuple[float, float, float, float] | None:
    """PoseLift bbox [x1, y1, x2, y2] → (x, y, w, h) (FrameKP's convention)."""
    b = np.asarray(arr, dtype=np.float32).ravel()
    if b.size < 4:
        return None
    x1, y1, x2, y2 = (float(v) for v in b[:4])
    return (x1, y1, x2 - x1, y2 - y1)


def _kp_bbox(
    fv: Any,
) -> tuple[NDArray[np.float32], tuple[float, float, float, float] | None] | None:
    """A PoseLift per-person frame value → ((17,3)[x,y,conf], bbox xywh|None).

    Real format is the 2-element list ``[bbox(4), keypoints(17,3)]``; also tolerate
    a dict or raw keypoints. Keypoints are swapped [y,x]→[x,y] here."""
    kp_src: Any = fv
    bbox_src: Any = None
    if isinstance(fv, (list, tuple)) and len(fv) == 2:
        bbox_src, kp_src = fv[0], fv[1]
    elif isinstance(fv, dict):
        kp_src = fv.get("keypoints", fv.get("kp", fv.get("pose")))
        bbox_src = fv.get("bbox", fv.get("box"))
    kp = _to_kp17(kp_src)
    if kp is None:
        return None
    kp = kp[:, [1, 0, 2]]  # [y, x, conf] → [x, y, conf]
    return kp, (_bbox_xywh(bbox_src) if bbox_src is not None else None)


def load_poselift_pkl(pkl_path: Path) -> list[PersonTrack]:
    """Parse one PoseLift .pkl (``{frame: {person: [bbox, kp]}}``) into per-person
    tracks. Transposes the frame→person nesting into person→frames."""
    with pkl_path.open("rb") as f:
        obj = pickle.load(f)  # noqa: S301 — trusted local dataset
    if not isinstance(obj, dict):
        raise ValueError(
            f"{pkl_path.name}: top-level is {type(obj).__name__}, expected a "
            "{frame: {person: ...}} dict. Run inspect_pkl to print the structure."
        )
    by_person: dict[str, list[FrameKP]] = {}
    for fid, frame_data in obj.items():
        try:
            f_idx = int(fid)
        except (TypeError, ValueError):
            continue
        if not isinstance(frame_data, dict):
            continue
        for pid, val in frame_data.items():
            got = _kp_bbox(val)
            if got is not None:
                by_person.setdefault(str(pid), []).append(
                    FrameKP(frame_idx=f_idx, kp=got[0], bbox=got[1])
                )
    tracks = [
        PersonTrack(person_id=pid, frames=sorted(fr, key=lambda fk: fk.frame_idx))
        for pid, fr in by_person.items()
        if fr
    ]
    if not tracks:
        raise ValueError(
            f"{pkl_path.name}: parsed 0 person tracks — the pickle layout is "
            "unrecognised. Run inspect_pkl and adjust _kp_bbox()."
        )
    return tracks


def _clip_key(stem: str) -> tuple[int, int] | None:
    """'<cam>_<vid>' → (int cam, int vid) so '1_222' and '01_0222' match."""
    m = re.match(r"^0*(\d+)_0*(\d+)$", stem)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _index_labels(root: Path) -> dict[tuple[int, int], Path]:
    """Index every .npy under `root` by its normalised (cam, vid) key."""
    out: dict[tuple[int, int], Path] = {}
    for npy in root.rglob("*.npy"):
        k = _clip_key(npy.stem)
        if k is not None:
            out.setdefault(k, npy)
    return out


def load_clip(pkl_path: Path, labels: NDArray[np.int_] | None = None) -> PoseClip:
    """Build a PoseClip from a .pkl + an optional frame-label array.

    If ``labels`` is None, falls back to a sibling ``.npy`` (legacy layout); else
    the frame count is the max frame index seen."""
    tracks = load_poselift_pkl(pkl_path)
    if labels is None:
        sib = pkl_path.with_suffix(".npy")
        if sib.exists():
            labels = np.asarray(np.load(sib)).astype(np.int_).ravel()
    if labels is not None:
        n_frames = int(len(labels))
    else:
        n_frames = 1 + max((fk.frame_idx for t in tracks for fk in t.frames), default=-1)
    return PoseClip(name=pkl_path.stem, persons=tracks, n_frames=n_frames, frame_labels=labels)


def load_dataset(data_dir: Path) -> list[PoseClip]:
    """Load every .pkl under `data_dir` (recursive), attaching labels found
    anywhere under it by normalised (cam, vid). Returns all clips, sorted by name."""
    root = Path(data_dir)
    pkls = sorted(root.rglob("*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No .pkl files under {data_dir}")
    labels = _index_labels(root)
    seen: set[tuple[int, int]] = set()
    clips: list[PoseClip] = []
    for p in pkls:
        k = _clip_key(p.stem)
        if k is not None and k in seen:
            continue  # skip duplicate (e.g. a GT copy) of the same clip
        lp = labels.get(k) if k is not None else None
        arr = np.asarray(np.load(lp)).astype(np.int_).ravel() if lp is not None else None
        clips.append(load_clip(p, arr))
        if k is not None:
            seen.add(k)
    return clips


def load_split(data_dir: Path) -> tuple[list[PoseClip], list[PoseClip]]:
    """(train, test): a clip is TEST (eval, labelled) if a matching .npy exists,
    else TRAIN (normal-only). Matches PoseLift's Train/Test split without relying
    on directory names — purely on whether a label file is present."""
    train: list[PoseClip] = []
    test: list[PoseClip] = []
    for clip in load_dataset(data_dir):
        (test if clip.frame_labels is not None else train).append(clip)
    return train, test


def inspect_pkl(pkl_path: Path, *, max_items: int = 3, depth: int = 3) -> str:
    """Human-readable dump of a pickle's structure — types, dict keys, array
    shapes — to a couple of levels, to reverse-engineer the layout without guessing."""

    def describe(o: Any, d: int, prefix: str) -> list[str]:
        if d <= 0:
            return [f"{prefix}{type(o).__name__} …"]
        if isinstance(o, dict):
            lines = [f"{prefix}dict (len {len(o)})"]
            for i, (k, v) in enumerate(o.items()):
                if i >= max_items:
                    lines.append(f"{prefix}  … +{len(o) - max_items} more keys")
                    break
                lines.append(f"{prefix}  [{k!r}] ->")
                lines += describe(v, d - 1, prefix + "    ")
            return lines
        if isinstance(o, (list, tuple)):
            lines = [f"{prefix}{type(o).__name__} (len {len(o)})"]
            if o:
                lines += describe(o[0], d - 1, prefix + "  [0] ")
            return lines
        if isinstance(o, np.ndarray):
            return [f"{prefix}ndarray shape={o.shape} dtype={o.dtype}"]
        return [f"{prefix}{type(o).__name__}: {str(o)[:60]}"]

    with pkl_path.open("rb") as f:
        obj = pickle.load(f)  # noqa: S301 — trusted local dataset
    return "\n".join([f"== {pkl_path.name} ==", *describe(obj, depth, "")])
