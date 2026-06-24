"""Tests for the eval runner: manifest load, VLM run (mocked), report build."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sentry_ai.eval.runner import (
    EvalEntry,
    build_report,
    format_summary,
    load_manifest,
    run_clips,
)


def test_load_manifest_list_and_dict(tmp_path: Path) -> None:
    p = tmp_path / "m.json"
    p.write_text(
        json.dumps(
            {
                "clips": [
                    {"path": "a.mp4", "label": "theft"},
                    {
                        "path": "b.mp4",
                        "label": "benign",
                        "predicted": "browsing",
                        "confidence": 0.3,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    entries = load_manifest(p)
    assert len(entries) == 2
    assert entries[0].path == "a.mp4" and entries[0].predicted is None
    assert entries[1].predicted == "browsing" and entries[1].confidence == 0.3


@pytest.mark.asyncio
async def test_run_clips_fills_predictions_and_records_errors() -> None:
    entries = [
        EvalEntry(path="ok.mp4", label="theft"),
        EvalEntry(path="bad.mp4", label="benign"),
        EvalEntry(path="cached.mp4", label="theft", predicted="pocket_conceal", confidence=0.9),
    ]

    async def fake_verify(path: str) -> tuple[str, float]:
        if path == "bad.mp4":
            raise RuntimeError("ffmpeg blew up")
        return "pocket_conceal", 0.88

    await run_clips(entries, fake_verify)
    assert entries[0].predicted == "pocket_conceal" and entries[0].confidence == 0.88
    assert entries[1].predicted is None and "ffmpeg" in (entries[1].error or "")
    # an already-predicted entry is left untouched (no VLM call)
    assert entries[2].confidence == 0.9


def test_build_report_metrics_and_errors() -> None:
    entries = [
        EvalEntry(path="1", label="theft", predicted="pocket_conceal", confidence=0.9),
        EvalEntry(path="2", label="theft", predicted="browsing", confidence=0.4),  # missed
        EvalEntry(path="3", label="benign", predicted="browsing", confidence=0.2),
        EvalEntry(path="4", label="benign", predicted=None, error="boom"),  # errored
    ]
    rep = build_report(entries)
    assert rep["n_total"] == 4 and rep["n_scored"] == 3 and rep["n_errors"] == 1
    b = rep["binary"]
    assert b["tp"] == 1 and b["fn"] == 1 and b["tn"] == 1 and b["fp"] == 0
    assert b["recall"] == 0.5  # caught 1 of 2 thefts
    assert b["precision"] == 1.0  # the 1 theft call was right
    assert rep["errors"] == [{"path": "4", "error": "boom"}]
    # summary renders without error
    assert "BINARY" in format_summary(rep)


def test_build_report_empty() -> None:
    rep = build_report([])
    assert rep["n_scored"] == 0
    assert rep["binary"]["precision"] is None
    assert rep["confusion"] == {}
