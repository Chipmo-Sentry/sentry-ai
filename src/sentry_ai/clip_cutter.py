"""Cut a time window from THIS node's MediaMTX fmp4 recordings (docs/19 I5).

The backend (Railway) can't read the node's local recordings, so the breach
clip-cut moved here, beside MediaMTX + the VLM. On a live threshold breach the
backend calls /v1/cut-verify; we cut a [-5s, +10s] window from the recorded
segments, hand it to the VLM, and return the clip bytes + verdict.

Ported from sentry-backend/services/clip_cutter.py (same segment-matching +
ffmpeg-concat approach), pointed at `settings.mediamtx_recordings_dir`.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from sentry_ai.logging_setup import get_logger
from sentry_ai.settings import get_settings

log = get_logger("sentry_ai.clip_cutter")

# MediaMTX fmp4 segment filenames: "2026-05-28_14-58-34-265139.mp4"
_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:-\d+)?)")
# Assumed max length of the last (active) segment — 2× recordSegmentDuration.
_MAX_SEGMENT_SEC = 120


@dataclass(slots=True)
class CutResult:
    storage_path: str
    file_size_bytes: int
    duration_sec: float
    captured_at: datetime


class ClipCutError(RuntimeError):
    pass


def _parse_segment_ts(filename: str) -> datetime | None:
    m = _TS_RE.match(filename)
    if not m or "_" not in m.group(1):
        return None
    date_str, time_str_full = m.group(1).split("_", 1)
    parts = time_str_full.split("-")
    if len(parts) < 3:
        return None
    try:
        dt = datetime.fromisoformat(f"{date_str}T{parts[0]}:{parts[1]}:{parts[2]}")
        if len(parts) >= 4:
            with contextlib.suppress(ValueError):
                dt = dt + timedelta(microseconds=int(parts[3]))
        return dt
    except (ValueError, IndexError):
        return None


def _find_segments_in_window(
    cam_dir: Path, window_start: datetime, window_end: datetime
) -> list[Path]:
    if not cam_dir.is_dir():
        return []
    segments: list[tuple[datetime, Path]] = []
    for p in cam_dir.iterdir():
        if not p.is_file() or p.suffix != ".mp4" or p.name.startswith("live_"):
            continue
        ts = _parse_segment_ts(p.name)
        if ts is not None:
            segments.append((ts, p))
    segments.sort(key=lambda x: x[0])
    overlapping: list[Path] = []
    for i, (ts, path) in enumerate(segments):
        seg_end = (
            segments[i + 1][0]
            if i + 1 < len(segments)
            else ts + timedelta(seconds=_MAX_SEGMENT_SEC)
        )
        if ts < window_end and seg_end > window_start:
            overlapping.append(path)
    return overlapping


async def _run_ffmpeg(args: list[str], timeout: float = 60.0) -> None:  # noqa: ASYNC109
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError as e:
        proc.kill()
        raise ClipCutError(f"ffmpeg timed out after {timeout}s") from e
    if proc.returncode != 0:
        raise ClipCutError(
            f"ffmpeg exit {proc.returncode}: {stderr.decode('utf-8', 'replace')[-400:]}"
        )


async def cut_window(
    mediamtx_path: str, start_offset_sec: int = -5, duration_sec: int = 15
) -> CutResult:
    """Cut [now+offset, +duration] from this node's recordings → a temp mp4."""
    settings = get_settings()
    if not settings.mediamtx_recordings_dir:
        raise ClipCutError("MEDIAMTX_RECORDINGS_DIR not configured on this node")
    cam_dir = Path(settings.mediamtx_recordings_dir) / mediamtx_path
    if not cam_dir.is_dir():
        raise ClipCutError(f"camera dir not found: {cam_dir}")

    now = datetime.now()  # MediaMTX writes local-time filenames
    window_start = now + timedelta(seconds=start_offset_sec)
    window_end = window_start + timedelta(seconds=duration_sec)
    captured_at = now.astimezone(UTC).replace(microsecond=0) + timedelta(seconds=start_offset_sec)

    segments = _find_segments_in_window(cam_dir, window_start, window_end)
    if not segments:
        raise ClipCutError(f"no segments overlap [{window_start}, {window_end}] in {cam_dir}")

    out_dir = Path(tempfile.gettempdir())
    out_path = out_dir / f"live_{captured_at:%Y%m%dT%H%M%SZ}_{mediamtx_path}_{uuid4().hex[:8]}.mp4"
    first_ts = _parse_segment_ts(segments[0].name)
    if first_ts is None:
        raise ClipCutError("could not parse first segment timestamp")
    seek = max(0.0, (window_start - first_ts).total_seconds())

    list_path = out_dir / f"_concat_{uuid4().hex[:8]}.txt"
    try:
        list_path.write_text(
            "\n".join(f"file '{p.as_posix()}'" for p in segments), encoding="utf-8"
        )
        await _run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_path.as_posix(),
                "-ss",
                f"{seek:.3f}",
                "-t",
                f"{duration_sec}",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                "-an",
                "-movflags",
                "+faststart",
                out_path.as_posix(),
            ]
        )
    finally:
        list_path.unlink(missing_ok=True)

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise ClipCutError("ffmpeg produced empty output")
    log.info(
        "clip_cutter.ok", path=mediamtx_path, size=out_path.stat().st_size, segments=len(segments)
    )
    return CutResult(
        storage_path=out_path.resolve().as_posix(),
        file_size_bytes=out_path.stat().st_size,
        duration_sec=float(duration_sec),
        captured_at=captured_at,
    )
