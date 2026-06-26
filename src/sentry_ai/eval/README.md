# Detection-quality eval harness

Measures how accurate the Stage-2 VLM actually is against a **labelled** clip set —
precision, recall, F1, a confusion matrix — so a prompt / model / threshold change
is a data-driven regression gate, not a guess.

Why this matters: the live dashboard (`/admin/analytics/quality`) gives precision +
calibration from staff feedback on **alerts**, but it can't measure **recall** —
a real theft that never alerted never becomes a row. A curated clip set that
includes known thefts (alerted or not) is the only way to measure what the system
**misses**. That's what this harness is for.

## Manifest

A JSON list (or `{"clips": [...]}`) of entries:

```json
{ "path": "clips/x.mp4", "label": "theft" }
{ "path": "clips/y.mp4", "label": "benign" }
{ "path": "clips/z.mp4", "label": "pocket_conceal", "predicted": "browsing", "confidence": 0.4 }
```

- `label` — ground truth. Use `theft`/`benign` for a binary set, or a VLM
  `Category` (`pocket_conceal`, `bag_conceal`, `cart_pickup`, `browsing`, `other`)
  for a full per-category report. Concealment categories collapse to `theft`.
- `predicted`/`confidence` — optional. If present the entry is scored as-is (no VLM
  run). This is how a manifest **exported from production feedback** is scored
  offline (see `GET /api/v1/admin/eval/dataset` in the backend).

## Run it

On the GPU node (clips on disk + the VLM reachable):

```bash
uv run python -m sentry_ai.eval run manifest.json --out report.json
```

Score a production-feedback export offline (no model needed):

```bash
uv run python -m sentry_ai.eval score exported_manifest.json --out report.json
```

The summary prints binary precision/recall/F1 + per-category breakdown; the full
confusion matrix + per-clip rows land in `report.json`.

## Building a labelled set

1. **From production feedback (fastest):** `GET /api/v1/admin/eval/dataset` returns
   feedback'd alerts as manifest entries (staff verdict = label, VLM = predicted).
   Score it with `score`. Gives **precision** on real data immediately.
2. **Curated (for recall):** hand-label a folder of clips — crucially include real
   thefts that the system may have missed — and `run` it. Only this measures recall.

## Pose mode — baseline the Stage-1 behavior engine (no VLM, no pixels)

`run`/`score` measure the **Stage-2 VLM** on video clips. `pose` measures
**Stage-1** — `BehaviorScorer`, the YOLO-pose → behavior engine — on a *pose-only*
dataset (skeleton keypoints, no pixels). This is what quantifies "does the engine
recognise concealment from the pose stream", which a VLM clip set can't isolate.

Built for **PoseLift** (TeCSAR-UNCC, WACV 2025) — real retail CCTV pose tracks
labelled pocket / bag / under-clothes concealment. Apache-2.0. The data is on
Google Drive (manual download, no programmatic endpoint); drop the `.pkl` + `.npy`
pairs in one directory.

```bash
# Reverse-engineer the .pkl layout first if load fails:
uv run python -m sentry_ai.eval pose data/poselift/ --inspect

# Replay every clip through the engine and score it:
uv run python -m sentry_ai.eval pose data/poselift/ --out pose_report.json [--fps 15]
```

Each clip is replayed frame-by-frame through a fresh scorer with a **synthetic
clock** (1/fps per frame), so loiter/sequence/hold-release timing is faithful.
Output:

- **video-level binary** (theft vs benign) swept over peak-score thresholds, with
  the engine's own LOW/MEDIUM/HIGH/CRITICAL cutoffs (10/25/50) highlighted and the
  best-F1 operating point;
- **frame-level ROC-AUC** — directly comparable to the PoseLift paper's STG-NF /
  TSGAD / GEPC numbers.

**CAVEAT (read before trusting recall):** a pose-only set has no COCO object
detections, so `items` is always empty and the pocket/bag detectors — gated on a
held item (`require_holding`, ai#9) — under-fire. That is an honest measure of the
pose-only path; the gap it shows is the motivation for the learned skeleton-action
layer (ADR-0030).
