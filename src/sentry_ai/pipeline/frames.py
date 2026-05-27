"""Extract evenly-spaced keyframes from an mp4 using ffmpeg."""

import asyncio
import io
import tempfile
from pathlib import Path

import ffmpeg
from PIL import Image


def _probe_duration_sec(clip_path: Path) -> float:
    """Run ffprobe synchronously to get duration in seconds."""
    info = ffmpeg.probe(str(clip_path))
    fmt = info.get("format", {})
    duration = float(fmt.get("duration", 0.0))
    if duration <= 0.0:
        raise ValueError(f"Could not probe duration for {clip_path}")
    return duration


def _extract_one_frame_sync(clip_path: Path, timestamp_sec: float, out_path: Path) -> None:
    """Synchronous ffmpeg call — to be awaited via asyncio.to_thread."""
    (
        ffmpeg.input(str(clip_path), ss=timestamp_sec)
        .output(str(out_path), vframes=1, format="image2", vcodec="mjpeg")
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True, quiet=True)
    )


def _resize_jpeg(raw_path: Path, max_dim: int, quality: int) -> bytes:
    """Open a JPEG, thumbnail in-place, re-encode to bytes (no temp file)."""
    img = Image.open(raw_path).convert("RGB")
    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()


async def extract_keyframes(
    clip_path: Path,
    *,
    count: int = 5,
    max_dim: int = 640,
    quality: int = 85,
) -> list[bytes]:
    """Extract `count` evenly-spaced JPEG keyframes from an mp4.

    Returns a list of JPEG-encoded bytes, max edge `max_dim`, sized for VLM input.
    """
    if count <= 0:
        raise ValueError("count must be >= 1")

    duration = await asyncio.to_thread(_probe_duration_sec, clip_path)

    # Evenly spaced: 10%, 30%, 50%, 70%, 90% for count=5 (avoid black edges)
    margin = 1.0 / (count + 1)
    timestamps = [(i + 1) * margin * duration for i in range(count)]

    frames: list[bytes] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for i, ts in enumerate(timestamps):
            raw_path = tmp_dir / f"frame_{i:02d}.jpg"
            await asyncio.to_thread(_extract_one_frame_sync, clip_path, ts, raw_path)
            jpeg = await asyncio.to_thread(_resize_jpeg, raw_path, max_dim, quality)
            frames.append(jpeg)

    return frames
