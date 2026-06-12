# sentry-ai

The AI brain of **Chipmo Sentry**. One FastAPI service with two jobs:

1. **Live behaviour worker** — pulls each camera's RTSP stream, runs **YOLO26-pose + ByteTrack** per
   person, scores behaviour on a **0–100 risk scale** with a state-machine engine, and streams overlay
   metadata to the backend in real time.
2. **VLM verification** — when a person breaches a camera's risk threshold (or a clip is uploaded), cuts
   a short clip, extracts keyframes, and asks a self-hosted **Vision-Language Model** (Qwen3-VL-4B or
   MiniCPM-V 2.6 via Ollama) to classify the event and explain it **in Mongolian**.

Python 3.11 · FastAPI · Ollama · YOLO26 (ultralytics) · supervision (ByteTrack) · OpenCV · ffmpeg · Apache-2.0

> **Never runs on Railway.** This service needs a GPU, low-latency local video, and UDP — it runs on the
> Predator laptop (RTX 4060) today and a public GPU VPS at scale ([ADR-0016](../docs/07-DECISIONS.md)).
> It reaches the Railway backend out-bound via a Cloudflare Tunnel.

---

## The model stack

| Role | Model | Notes |
|---|---|---|
| **VLM** (default) | Qwen3-VL-4B — Ollama tag `qwen3-vl:4b-instruct` | provider name `qwen3-vl-4b` (ADR-0026); A/B-switchable per node |
| **VLM** (alt) | MiniCPM-V 2.6 — Ollama tag `minicpm-v:8b` (Q4) | provider name `minicpm-v-2.6` (rollback slot). `qwen2.5-vl-7b` deprecated, kept for rollback |
| **Pose** | `yolo26s-pose.pt` (default) / `yolo26n-pose.pt` (CPU-fast) | COCO-17 keypoints per person (ADR-0026; `yolo11*` = rollback) |
| **Items** | `yolo26n.pt` | retail-shrink classes: backpack, handbag, bottle, … |
| **Re-ID** | HistogramEmbedder (default) / OSNet (`torchreid`, optional) | cross-camera person identity |
| **Text embed** | `nomic-embed-text` (Ollama) | RAG few-shot; optional (`EMBED_MODEL=""` disables) |

VLM providers are pluggable via `providers/factory.py` — the backend can request a provider per call, and
each AI node can be configured centrally.

### Cross-camera re-ID

`live_worker/reid.py` links one shopper across the store's cameras so suspicion **accumulates** as they
move between aisles. Each person is cropped to the **torso** (via pose keypoints, with a body-proportion
fallback) before embedding, a quality gate skips people too small/occluded to identify reliably, and the
`StorePersonRegistry` matches against a small **gallery** of recent embeddings (robust to pose/lighting).
The default `histogram` embedder is dependency-light but weak; set **`REID_MODEL=osnet`** on the GPU node
for the learned model — install it once with
`uv pip install "torchreid @ git+https://github.com/KaiyangZhou/deep-person-reid.git"` (the code falls back
to the histogram automatically if torchreid isn't present).

## Behaviour engine v2 (ADR-0024)

`live_worker/behavior.py` scores each tracked person every analysed frame:

- **State machine:** `IDLE → SUSPICIOUS → PRODUCT_INTERACTION → CONCEALMENT → ALERT`.
- **Detectors** (weighted): looking-around, loitering, rapid movement, item pickup, body-block, crouch,
  wrist-to-torso, **bag interaction**, **pocket interaction**.
- **Sequence engine:** ordered patterns earn bonuses; `item_pickup → wrist_to_torso → conceal_hide` is a
  *critical* sequence that forces an **ALERT**.
- **Absolute 0–100 score** → 4 levels: LOW / MEDIUM / HIGH / CRITICAL (drives 🟢🟡🔴 overlay colour).
- **Robustness:** keypoint-confidence gating (joints below 0.3 are treated as undetected), temporal
  smoothing (noisy detectors must persist N frames), state decay between frames.

Weights, thresholds, and custom dimensions are tunable from the backend behaviour catalog and hot-reloaded
via the config poller.

---

## How a request flows

**Clip verify (Stage 2):**
```
backend POST /v1/verify {clip_path, store_id?, provider?, rag_query?}
  → ffmpeg extracts 5 keyframes (≈640px)
  → optional RAG: fetch similar past staff-verified cases → inject as Mongolian context
  → Jinja prompt (prompts/verify_v1.j2) + frames → Ollama VLM
  → strict Pydantic parse of the JSON verdict (2× retry → graceful fallback)
  → VerifyResponse {category, confidence, reasoning, model_name, latency_ms, embedding?}
```

**Live breach (cut-verify):** `POST /v1/cut-verify` cuts the relevant MediaMTX fmp4 segments
(`-5s … +10s` around the event), runs the same VLM pass, and returns the clip (base64) plus the verdict —
the backend stores the clip and creates the Alert.

**Live worker:** `POST /v1/live/start {camera_id, rtsp_url, store_id?}` spawns one thread per camera:
read frame → YOLO pose → ByteTrack id → (periodically) item detect → behaviour score → re-ID registry →
emit metadata. A background `MetadataEmitter` batches frames to the backend's `/internal/live-metadata`.

---

## API

| Method · Path | Purpose |
|---|---|
| `POST /v1/verify` | verify an mp4 clip (path-based) |
| `POST /v1/cut-verify` | cut MediaMTX segments around a live breach + verify |
| `POST /v1/live/start` · `/stop/{cam}` · `GET /status` · `/emitter` · `/snapshot/{cam}` | live worker control + debug |
| `GET /v1/models` | list available provider names |
| `GET /healthz` | status, version, Ollama reachability, loaded models |

All `/v1/*` routes require a Bearer service token when `AI_SERVICE_TOKEN` is set (production).

---

## Prerequisites

- Python 3.11, [`uv`](https://docs.astral.sh/uv/), and **ffmpeg** on `PATH`.
- An [Ollama](https://ollama.com) install with a VLM pulled: `ollama pull minicpm-v:8b`.
- For GPU inference: an NVIDIA GPU with a CUDA-enabled torch (the installer pulls the `cu128` wheels).

## Quick start

```bash
uv sync
cp .env.example .env          # set OLLAMA_BASE_URL, DEFAULT_PROVIDER, SENTRY_BACKEND_URL, …
uv run uvicorn sentry_ai.main:app --reload --port 8001
#  → http://localhost:8001/healthz
#  → http://localhost:8001/docs

# smoke test
curl -X POST http://localhost:8001/v1/verify \
  -H 'content-type: application/json' \
  -d '{"clip_path": "/abs/path/to/sample.mp4"}'
```

## Project layout

```
src/sentry_ai/
├── main.py               — FastAPI app + lifespan (auto-start live workers, spawn heartbeat child)
├── settings.py           — pydantic-settings (Ollama, provider, pairing, live config)
├── auth.py               — Bearer service-token guard
├── pair.py               — `python -m sentry_ai.pair` (6-digit code → JWT + .env)
├── heartbeat.py / heartbeat_cli.py — telemetry + config-poll in a SEPARATE child process
├── system_metrics.py     — CPU/RAM/GPU sampling (psutil + NVML)
├── rag.py                — embed_text + retrieve_context (few-shot loop)
├── clip_cutter.py        — cut MediaMTX fmp4 segments for live-breach verify
├── api/v1/               — health, verify, live
├── schemas/              — VerifyRequest/Response, VLMOutput (strict), CutVerify*
├── pipeline/             — frames (ffmpeg keyframes), prompt (Jinja), verifier (orchestrator)
├── providers/            — base Protocol, ollama_client, minicpm_v, qwen_vl, factory
└── live_worker/          — manager, camera_worker, yolo_runner, yolo_det, tracker (ByteTrack),
                            behavior (engine v2), reid, emitter, config_poller
prompts/                  — verify_v1.j2 (Mongolian task) + categories.md (label semantics)
installer/                — Inno Setup wizard → ChipmoSentryAi-Setup.exe
```

---

## Pairing, telemetry & observability

- **Pair** a node to a backend: `uv run python -m sentry_ai.pair --code 123456 --backend <url> --public-url <url>`.
  This redeems the 6-digit code and stores the returned node JWT + `AI_NODE_ID` in `.env`.
- **Heartbeat** runs as a separate child process (immune to GPU/CUDA GIL stalls), posting telemetry
  (FPS, active cameras, CPU/RAM/GPU, per-dependency health) every `HEARTBEAT_INTERVAL_SEC` and receiving
  central config (provider / frame-skip / enabled) in the response.
- **RAG loop:** verdict reasoning is embedded and stored as a `verified_case`; the next clip at the same
  store retrieves similar cases and shows them to the VLM as Mongolian few-shot context. No retraining.

The backend's observability dashboard ([docs/19](../docs/19-AI-OBSERVABILITY-DASHBOARD.md)) renders all of
this — live + historical resource charts, alert/feedback analytics, and the RAG loop.

---

## Testing, lint, type-check

```bash
uv run pytest tests/unit/           # Ollama mocked via respx
uv run pytest tests/integration/    # needs a real Ollama
uv run ruff format . && uv run ruff check .
uv run mypy src/sentry_ai           # strict
```

---

## Packaging — the one-installer GPU node

`installer/sentry-ai-server.iss` builds **`ChipmoSentryAi-Setup.exe`** (released by `release.yml` on tag).
The wizard collects backend URL / pairing code / Ollama URL / tunnel name, runs `uv sync` at install time
(downloads the CUDA torch wheels), and registers three Windows services via NSSM:

- **ChipmoSentryIngest** — MediaMTX (RTSP ingest)
- **ChipmoSentryAi** — this FastAPI service
- **ChipmoSentryTunnel** — cloudflared (out-bound to Railway)

Ollama is **not** bundled — the operator brings their own install, and the service points at it via
`OLLAMA_BASE_URL`. Full walkthrough: [docs/17-AI-SERVER-SETUP.md](../docs/17-AI-SERVER-SETUP.md).

A `Dockerfile` (multi-stage, ffmpeg in the runtime image, no Ollama) exists for containerised GPU hosts,
but **Railway is explicitly refused** for this service.

---

## Related repos

- [sentry-backend](https://github.com/Chipmo-Sentry/sentry-backend) — receives verdicts + live metadata
- [sentry-ingest](https://github.com/Chipmo-Sentry/sentry-ingest) — MediaMTX (co-located on this host)
- [sentry-agent-pc](https://github.com/Chipmo-Sentry/sentry-agent-pc) — pushes camera streams here

Platform overview: [Sentry-v.3 README](../README.md).
