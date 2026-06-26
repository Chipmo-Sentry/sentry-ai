"""CLI: score a labelled set against the engine.

    python -m sentry_ai.eval run   manifest.json [--out report.json]   # Stage-2 VLM
    python -m sentry_ai.eval score manifest.json [--out report.json]   # entry.predicted
    python -m sentry_ai.eval pose  data_dir/     [--out report.json]   # Stage-1 engine
    python -m sentry_ai.eval pose  data_dir/ --inspect                 # dump .pkl layout

`run` needs clips on disk + a reachable VLM (run it on the GPU node); `score`
needs no model and works on a manifest exported from production feedback. `pose`
replays a pose-only dataset (PoseLift) through the Stage-1 behavior engine — no
model, no pixels — to baseline behavior recognition (see pose_runner.py).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
from pathlib import Path

from sentry_ai.eval.runner import build_report, format_summary, load_manifest, run_clips


def _force_utf8_stdout() -> None:
    """The summaries use →/≥; a legacy Windows console is cp1252 and would crash
    on encode. Reconfigure to UTF-8 (no-op where already UTF-8)."""
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]


async def _default_verify_fn(path: str) -> tuple[str, float]:
    """Run the real verify pipeline with the node's configured provider."""
    from sentry_ai.dependencies import _get_client
    from sentry_ai.pipeline.verifier import verify_clip
    from sentry_ai.providers.factory import get_provider, resolve_provider_name

    provider = get_provider(resolve_provider_name(None), _get_client())
    output, _latency, _frames = await verify_clip(clip_path=Path(path), provider=provider)
    return output.category.value, output.confidence


def _run_pose(data_dir: Path, out: Path, *, fps: float, inspect: bool) -> int:
    """Replay a pose-only dataset through the Stage-1 engine (no model needed)."""
    from sentry_ai.eval.pose_runner import build_pose_report, format_pose_summary, replay_clip
    from sentry_ai.eval.poselift import inspect_pkl, load_dataset

    if inspect:
        pkls = sorted(data_dir.glob("*.pkl"))
        if not pkls:
            print(f"No .pkl files in {data_dir}")  # noqa: T201
            return 1
        print(inspect_pkl(pkls[0]))  # noqa: T201
        return 0

    clips = load_dataset(data_dir)
    results = [replay_clip(c, fps=fps) for c in clips]
    report = build_pose_report(results)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(format_pose_summary(report))  # noqa: T201 — CLI output is the point
    print(f"\n→ full report: {out}")  # noqa: T201
    return 0


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    ap = argparse.ArgumentParser(prog="python -m sentry_ai.eval")
    ap.add_argument("mode", choices=["run", "score", "pose"])
    ap.add_argument("target", type=Path, help="manifest.json (run/score) or data dir (pose)")
    ap.add_argument("--out", type=Path, default=Path("eval_report.json"))
    ap.add_argument("--fps", type=float, default=15.0, help="pose replay frame rate")
    ap.add_argument("--inspect", action="store_true", help="pose: dump first .pkl layout")
    args = ap.parse_args(argv)

    if args.mode == "pose":
        return _run_pose(args.target, args.out, fps=args.fps, inspect=args.inspect)

    entries = load_manifest(args.target)
    if args.mode == "run":
        asyncio.run(run_clips(entries, _default_verify_fn))
    report = build_report(entries)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(format_summary(report))  # noqa: T201 — CLI output is the point
    print(f"\n→ full report: {args.out}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
