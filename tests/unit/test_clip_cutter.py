"""Node-side clip cutter — segment parsing + window matching (docs/19 I5)."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from sentry_ai import clip_cutter


def test_parse_segment_ts_with_micros() -> None:
    ts = clip_cutter._parse_segment_ts("2026-05-28_14-58-34-265139.mp4")
    assert ts == datetime(2026, 5, 28, 14, 58, 34, 265139)


def test_parse_segment_ts_no_micros() -> None:
    ts = clip_cutter._parse_segment_ts("2026-05-28_14-58-34.mp4")
    assert ts == datetime(2026, 5, 28, 14, 58, 34)


def test_parse_segment_ts_rejects_junk() -> None:
    assert clip_cutter._parse_segment_ts("not-a-segment.mp4") is None
    assert clip_cutter._parse_segment_ts("live_clip.mp4") is None


def test_find_segments_in_window(tmp_path: Path) -> None:
    base = datetime(2026, 5, 28, 15, 0, 0)
    # 3 consecutive 60s segments
    for i in range(3):
        ts = base + timedelta(seconds=60 * i)
        (tmp_path / f"{ts:%Y-%m-%d_%H-%M-%S}.mp4").write_bytes(b"x")
    (tmp_path / "live_ignoreme.mp4").write_bytes(b"x")  # our own output
    (tmp_path / "junk.txt").write_text("x")

    # Window [15:00:30, 15:00:45] overlaps only the first segment.
    got = clip_cutter._find_segments_in_window(
        tmp_path, base + timedelta(seconds=30), base + timedelta(seconds=45)
    )
    assert [p.name for p in got] == ["2026-05-28_15-00-00.mp4"]

    # Window spanning the seam picks up two segments.
    got2 = clip_cutter._find_segments_in_window(
        tmp_path, base + timedelta(seconds=55), base + timedelta(seconds=65)
    )
    assert len(got2) == 2
