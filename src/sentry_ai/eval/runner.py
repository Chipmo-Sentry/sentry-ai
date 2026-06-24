"""Run a labelled clip set through the Stage-2 VLM and score it.

A manifest is a JSON list (or {"clips": [...]}) of entries:
    {"path": "clips/x.mp4", "label": "theft"}            # ground truth
    {"path": "...", "label": "benign"}
    {"path": "...", "label": "pocket_conceal", "predicted": "browsing", "confidence": 0.4}

`label` is ground truth — either "theft"/"benign" or a VLM Category. If an entry
already carries `predicted` (e.g. exported from production feedback), it's scored
as-is (no VLM run); otherwise `run_clips` calls the verify pipeline to fill it.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentry_ai.eval.metrics import (
    binary_metrics,
    classification_report,
    confusion_matrix,
    to_binary,
)

# path -> (predicted_category, confidence)
VerifyFn = Callable[[str], Awaitable[tuple[str, float]]]


@dataclass
class EvalEntry:
    path: str
    label: str
    predicted: str | None = None
    confidence: float | None = None
    error: str | None = None


def load_manifest(path: Path) -> list[EvalEntry]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data["clips"] if isinstance(data, dict) else data
    entries: list[EvalEntry] = []
    for it in items:
        entries.append(
            EvalEntry(
                path=str(it["path"]),
                label=str(it["label"]),
                predicted=(None if it.get("predicted") is None else str(it["predicted"])),
                confidence=(None if it.get("confidence") is None else float(it["confidence"])),
            )
        )
    return entries


async def run_clips(entries: list[EvalEntry], verify_fn: VerifyFn) -> list[EvalEntry]:
    """Fill `predicted`/`confidence` for entries that don't already have them by
    running the VLM. One bad clip is recorded as an error and skipped — a single
    failure must not abort a long eval run."""
    for e in entries:
        if e.predicted is not None:
            continue
        try:
            cat, conf = await verify_fn(e.path)
            e.predicted, e.confidence = cat, conf
        except Exception as ex:  # noqa: BLE001 — record + continue past a bad clip
            e.error = str(ex)[:200]
    return entries


def build_report(entries: list[EvalEntry]) -> dict[str, Any]:
    """Classification report over the scored entries: binary (theft/benign)
    headline + multi-class per-category + confusion matrix + per-clip rows."""
    scored = [e for e in entries if e.predicted is not None]
    errored = [e for e in entries if e.predicted is None]
    y_true = [e.label for e in scored]
    y_pred = [e.predicted or "" for e in scored]
    return {
        "n_total": len(entries),
        "n_scored": len(scored),
        "n_errors": len(errored),
        "binary": binary_metrics(y_true, y_pred),
        "multiclass": classification_report(y_true, y_pred) if scored else {},
        "confusion": confusion_matrix(y_true, y_pred) if scored else {},
        "errors": [{"path": e.path, "error": e.error} for e in errored],
        "rows": [
            {
                "path": e.path,
                "label": e.label,
                "predicted": e.predicted,
                "confidence": e.confidence,
                "correct": to_binary(e.label) == to_binary(e.predicted or ""),
            }
            for e in scored
        ],
    }


def format_summary(report: dict[str, Any]) -> str:
    b = report["binary"]
    lines = [
        f"Clips: {report['n_scored']} scored, {report['n_errors']} errored,"
        f" {report['n_total']} total",
        "",
        "BINARY (theft vs benign):",
        f"  precision={b['precision']}  recall={b['recall']}  f1={b['f1']}  acc={b['accuracy']}",
        f"  tp={b['tp']} fp={b['fp']} fn={b['fn']} tn={b['tn']}",
    ]
    mc = report.get("multiclass") or {}
    if mc:
        lines += [
            "",
            f"Accuracy={mc['accuracy']}  macro_f1={mc['macro_f1']}",
            "Per-category:",
        ]
        for label, m in mc["labels"].items():
            lines.append(
                f"  {label:18s} P={m['precision']} R={m['recall']} F1={m['f1']} n={m['support']}"
            )
    if report["n_errors"]:
        lines += ["", f"⚠ {report['n_errors']} clip(s) failed to score — see report.errors"]
    return "\n".join(lines)
