"""CLI: train + eval the skeleton-action anomaly model on PoseLift.

    python -m sentry_ai.skeleton train data_dir/ --out model.pt
    python -m sentry_ai.skeleton train data_dir/ --out model.pt --onnx model.onnx
    python -m sentry_ai.skeleton eval  data_dir/ --model model.pt [--out report.json]

Run on a box with the PoseLift .pkl/.npy pairs (a free Google Colab GPU, or any
CPU machine — torch is CPU-only here, just slower). `train` learns NORMAL motion
and saves a checkpoint; both commands print the learned model's frame-level
ROC-AUC NEXT TO the rule-based BehaviorScorer baseline's, so you can see whether
the learned model actually beats today's engine before promoting it.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

from sentry_ai.eval.pose_runner import build_pose_report, replay_clip
from sentry_ai.eval.poselift import load_split
from sentry_ai.skeleton.infer import evaluate, load_checkpoint
from sentry_ai.skeleton.train import TrainConfig, export_onnx, save_checkpoint, train_autoencoder


def _force_utf8_stdout() -> None:
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]


def _baseline_report(clips: list[Any]) -> dict[str, Any]:
    """Rule-based BehaviorScorer baseline on the same clips (the bar to beat)."""
    return build_pose_report([replay_clip(c) for c in clips])


def _print_comparison(learned: dict[str, Any], baseline: dict[str, Any]) -> None:
    print("\n=== Frame-level ROC-AUC (higher = better) ===")
    print(f"  rule baseline (BehaviorScorer): {baseline['frame_auc']}")
    print(f"  learned anomaly model:          {learned['frame_auc']}")
    lr, ba = learned["frame_auc"], baseline["frame_auc"]
    if lr is not None and ba is not None:
        verdict = "BEATS" if lr > ba else "does NOT beat"
        print(f"  → learned model {verdict} the baseline ({lr} vs {ba})")
    print(f"\n  (video-level peak AUC: learned={learned['peak_auc']} baseline={baseline['peak_auc']})")


def _cmd_train(args: argparse.Namespace) -> int:
    train_clips, test_clips = load_split(Path(args.data))
    # Train on the normal (Train) split; if a dataset has no labelled split,
    # train on everything. Eval on the labelled (Test) split.
    fit_clips = train_clips or test_clips
    eval_clips = test_clips or train_clips
    print(f"Loaded {len(train_clips)} train (normal) + {len(test_clips)} test (labelled) clips")
    cfg = TrainConfig(length=args.window, stride=args.stride, epochs=args.epochs, seed=args.seed)
    model, meta = train_autoencoder(fit_clips, cfg, device=args.device, log=print)
    print(f"Trained on {meta['n_normal_windows']} normal windows; threshold={meta['threshold']:.6f}")
    save_checkpoint(model, meta, args.out)
    print(f"Saved checkpoint -> {args.out}")
    if args.onnx:
        try:
            export_onnx(model, meta, args.onnx)
            print(f"Exported ONNX -> {args.onnx}")
        except RuntimeError as e:
            # The checkpoint is already saved; a missing export-only dep shouldn't
            # fail the whole run — just tell the user how to enable it.
            print(f"ONNX export skipped: {e}")
    learned = evaluate(eval_clips, model, meta, device=args.device)
    _print_comparison(learned, _baseline_report(eval_clips))
    if args.report:
        Path(args.report).write_text(json.dumps(learned, indent=2), encoding="utf-8")
        print(f"Wrote learned report -> {args.report}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    train_clips, test_clips = load_split(Path(args.data))
    clips = test_clips or train_clips
    print(f"Eval on {len(clips)} clips from {args.data}")
    model, meta = load_checkpoint(args.model, device=args.device)
    learned = evaluate(clips, model, meta, device=args.device)
    _print_comparison(learned, _baseline_report(clips))
    if args.out:
        Path(args.out).write_text(json.dumps(learned, indent=2), encoding="utf-8")
        print(f"Wrote learned report -> {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    ap = argparse.ArgumentParser(prog="sentry_ai.skeleton", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train the anomaly model on a PoseLift dir")
    t.add_argument("data")
    t.add_argument("--out", default="skeleton_anomaly.pt", help="checkpoint output path")
    t.add_argument("--onnx", default=None, help="also export ONNX here (for the edge)")
    t.add_argument("--report", default=None, help="write the learned JSON report here")
    t.add_argument("--window", type=int, default=32)
    t.add_argument("--stride", type=int, default=8)
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cpu", help="cpu | cuda (Colab)")
    t.set_defaults(fn=_cmd_train)

    e = sub.add_parser("eval", help="eval a saved checkpoint vs the rule baseline")
    e.add_argument("data")
    e.add_argument("--model", required=True, help="checkpoint .pt from train")
    e.add_argument("--out", default=None, help="write the learned JSON report here")
    e.add_argument("--device", default="cpu")
    e.set_defaults(fn=_cmd_eval)

    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
