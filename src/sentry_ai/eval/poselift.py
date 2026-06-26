"""Load the PoseLift dataset (TeCSAR-UNCC, WACV 2025) into `PoseClip`s.

PoseLift is privacy-preserving: skeleton keypoints only, no pixels. Per the
dataset README each video yields two files sharing a `<cam>_<vid>` stem:

  * ``<cam>_<vid>.pkl`` — per-person tracks: Person ID, bbox (XYWH), keypoints
    (XYC = x, y, confidence). Source pipeline: YOLOv8 → ByteTrack → HRNet, so the
    keypoints are COCO-17.
  * ``<cam>_<vid>.npy`` — per-frame binary anomaly label (0 normal / 1 shoplift).

The exact in-pickle Python structure is NOT documented upstream, so the loader
accepts the handful of shapes these TeCSAR pose dumps use in the wild and, when
it can't, points you at ``inspect_pkl`` (CLI: ``--inspect``) to print the real
layout so the normaliser can be nudged in one place (`_normalise_person`).

Download (manual — the data lives on Google Drive, no programmatic endpoint):
  https://github.com/TeCSAR-UNCC/PoseLift  → drop the .pkl/.npy pairs in one dir.
"""

from __future__ import annotations

import pickle  # noqa: S403 — trusted local dataset files, not untrusted input
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sentry_ai.eval.pose_runner import FrameKP, PersonTrack, PoseClip

COCO_17 = 17


def _to_kp17(arr: Any) -> NDArray[np.float32] | None:
    """Coerce one frame's keypoints to (17, 3) [x, y, conf]. Accepts (17,3),
    (17,2), flat 51 or 34, or a list thereof. Returns None if it can't."""
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
    elif a.shape[1] >= 3:
        a = a[:, :3]
    else:
        return None
    return a.astype(np.float32)


def _to_bbox(arr: Any) -> tuple[float, float, float, float] | None:
    if arr is None:
        return None
    b = np.asarray(arr, dtype=np.float32).ravel()
    if b.size >= 4:
        return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    return None


def _frames_from_mapping(data: dict[Any, Any]) -> list[FrameKP]:
    """person_data is {frame_id: <kp or {keypoints,bbox}>}."""
    out: list[FrameKP] = []
    for fid, fv in data.items():
        try:
            idx = int(fid)
        except (TypeError, ValueError):
            continue
        kp_src, bbox_src = fv, None
        if isinstance(fv, dict):
            kp_src = fv.get("keypoints", fv.get("kp", fv.get("pose")))
            bbox_src = fv.get("bbox", fv.get("box"))
        kp = _to_kp17(kp_src)
        if kp is not None:
            out.append(FrameKP(frame_idx=idx, kp=kp, bbox=_to_bbox(bbox_src)))
    return out


def _frames_from_arrays(kps: Any, bboxes: Any = None, frame_ids: Any = None) -> list[FrameKP]:
    """person_data carries parallel arrays: keypoints (T,17,3|T,51), optional
    bbox (T,4) and frame ids (T,)."""
    ka = np.asarray(kps, dtype=np.float32)
    if ka.ndim == 2 and ka.shape[1] in (COCO_17 * 3, COCO_17 * 2):
        ka = ka.reshape(ka.shape[0], COCO_17, -1)
    if ka.ndim != 3:
        return []
    ids = list(frame_ids) if frame_ids is not None else list(range(len(ka)))
    ba = np.asarray(bboxes, dtype=np.float32) if bboxes is not None else None
    out: list[FrameKP] = []
    for i in range(len(ka)):
        kp = _to_kp17(ka[i])
        if kp is None:
            continue
        bbox = _to_bbox(ba[i]) if ba is not None and i < len(ba) else None
        out.append(FrameKP(frame_idx=int(ids[i]), kp=kp, bbox=bbox))
    return out


def _normalise_person(person_data: Any) -> list[FrameKP]:
    """One person's value → frames. Handles {frame_id: ...}, {keypoints:[...]},
    and a raw (T,17,3) array. Tweak HERE if `--inspect` shows a new shape."""
    if isinstance(person_data, dict):
        if "keypoints" in person_data or "kp" in person_data:
            return _frames_from_arrays(
                person_data.get("keypoints", person_data.get("kp")),
                person_data.get("bbox", person_data.get("boxes")),
                person_data.get("frames", person_data.get("frame_ids")),
            )
        return _frames_from_mapping(person_data)
    arr = np.asarray(person_data, dtype=np.float32)
    if arr.ndim in (2, 3):
        return _frames_from_arrays(arr)
    return []


def load_poselift_pkl(pkl_path: Path) -> list[PersonTrack]:
    """Parse one PoseLift .pkl into per-person tracks."""
    with pkl_path.open("rb") as f:
        obj = pickle.load(f)  # noqa: S301 — trusted local dataset
    tracks: list[PersonTrack] = []
    items: Iterable[tuple[Any, Any]]
    if isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, (list, tuple)):
        items = enumerate(obj)
    else:
        raise ValueError(
            f"{pkl_path.name}: top-level is {type(obj).__name__}, expected dict/list. "
            "Run with --inspect to print the real structure."
        )
    for pid, pdata in items:
        frames = _normalise_person(pdata)
        if frames:
            frames.sort(key=lambda fk: fk.frame_idx)
            tracks.append(PersonTrack(person_id=str(pid), frames=frames))
    if not tracks:
        raise ValueError(
            f"{pkl_path.name}: parsed 0 person tracks. The pickle layout is "
            "unrecognised — run with --inspect and adjust _normalise_person()."
        )
    return tracks


def load_clip(pkl_path: Path) -> PoseClip:
    """Load a .pkl + its sibling .npy label into a PoseClip."""
    tracks = load_poselift_pkl(pkl_path)
    npy = pkl_path.with_suffix(".npy")
    frame_labels: NDArray[np.int_] | None = None
    if npy.exists():
        frame_labels = np.asarray(np.load(npy)).astype(np.int_).ravel()
    # Frame count: the label vector is authoritative; otherwise the max index seen.
    if frame_labels is not None:
        n_frames = int(len(frame_labels))
    else:
        n_frames = 1 + max((fk.frame_idx for t in tracks for fk in t.frames), default=-1)
    return PoseClip(
        name=pkl_path.stem, persons=tracks, n_frames=n_frames, frame_labels=frame_labels
    )


def load_dataset(data_dir: Path) -> list[PoseClip]:
    """Load every .pkl (with sibling .npy) under a directory, sorted by name."""
    pkls = sorted(Path(data_dir).glob("*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No .pkl files in {data_dir}")
    return [load_clip(p) for p in pkls]


def inspect_pkl(pkl_path: Path, *, max_items: int = 3, depth: int = 3) -> str:
    """Human-readable dump of a pickle's structure — types, dict keys, array
    shapes — to a couple of levels. Used by `--inspect` to reverse-engineer the
    layout without guessing."""

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
