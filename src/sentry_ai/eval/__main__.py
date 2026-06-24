"""CLI: score a labelled clip set against the Stage-2 VLM.

    python -m sentry_ai.eval run   manifest.json [--out report.json]   # run the VLM
    python -m sentry_ai.eval score manifest.json [--out report.json]   # use entry.predicted

`run` needs clips on disk + a reachable VLM (run it on the GPU node); `score`
needs no model and works on a manifest exported from production feedback.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from sentry_ai.eval.runner import build_report, format_summary, load_manifest, run_clips


async def _default_verify_fn(path: str) -> tuple[str, float]:
    """Run the real verify pipeline with the node's configured provider."""
    from sentry_ai.dependencies import _get_client
    from sentry_ai.pipeline.verifier import verify_clip
    from sentry_ai.providers.factory import get_provider, resolve_provider_name

    provider = get_provider(resolve_provider_name(None), _get_client())
    output, _latency, _frames = await verify_clip(clip_path=Path(path), provider=provider)
    return output.category.value, output.confidence


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sentry_ai.eval")
    ap.add_argument("mode", choices=["run", "score"])
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--out", type=Path, default=Path("eval_report.json"))
    args = ap.parse_args(argv)

    entries = load_manifest(args.manifest)
    if args.mode == "run":
        asyncio.run(run_clips(entries, _default_verify_fn))
    report = build_report(entries)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(format_summary(report))  # noqa: T201 — CLI output is the point
    print(f"\n→ full report: {args.out}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
