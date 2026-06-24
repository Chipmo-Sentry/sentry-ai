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
