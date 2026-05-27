# sentry-ai

AI inference service for **Chipmo Sentry**. Receives a clip path from [sentry-backend](https://github.com/Chipmo-Sentry/sentry-backend), runs Stage 2 verification with a self-hosted Vision-Language Model via [Ollama](https://ollama.com), and returns a structured shoplifting verdict.

Python 3.11 · FastAPI · httpx + Ollama · ffmpeg · MiniCPM-V 2.6 / Qwen2.5-VL (pluggable) · Apache 2.0

---

## How it works

```
[sentry-backend]                                    [USER LAPTOP — M1]
  POST /api/v1/clips    (mp4 upload)
       │
       │  POST /v1/verify  { clip_path, store_id, camera_id, provider? }
       └──────────────────────────────────────────►  [sentry-ai]
                                                        │
                                                        │ ① ffmpeg → 5 JPEG keyframes (640px)
                                                        │ ② Jinja render verify_v1.j2
                                                        │ ③ Ollama /api/chat (model=minicpm-v:8b)
                                                        │ ④ Pydantic strict parse VLMOutput
                                                        ▼
                                                   { category, confidence, reasoning,
                                                     model_name, inference_latency_ms,
                                                     frames_used }
```

There is **no per-API-call AI cost** — Ollama runs locally on the user's RTX 4060. When customer count grows, migrate Ollama to a Hetzner GPU VPS (`OLLAMA_BASE_URL` switch only — code unchanged).

---

## Prerequisites

1. **Python 3.11** (exact — 3.12+ has subtle ffmpeg-subprocess differences)
2. **uv 0.11.16+** (`pip install uv`)
3. **ffmpeg 6.x** on PATH (`ffmpeg -version`)
4. **Ollama** running locally:
   ```bash
   # one-time
   ollama pull minicpm-v:8b      # ~5.5 GB Q4
   # or Qwen2.5-VL (Session 2):
   # ollama pull qwen2.5-vl:7b   # ~4.8 GB
   ollama serve                  # default port 11434
   ```

---

## Quick start

```bash
uv sync
cp .env.example .env
# Edit .env — most defaults are fine; only SENTRY_BACKEND_SERVICE_TOKEN needs the
# value printed by sentry-backend on startup.

uv run uvicorn sentry_ai.main:app --reload --port 8001
# → http://localhost:8001/healthz
# → http://localhost:8001/docs
```

### Smoke a verify request

```bash
curl -X POST http://localhost:8001/v1/verify \
  -H "Content-Type: application/json" \
  -d '{"clip_path": "/absolute/path/to/clip.mp4"}'
```

---

## Project layout

```
src/sentry_ai/
├── main.py               — FastAPI app + lifespan + request_id middleware
├── settings.py           — pydantic-settings (OLLAMA_BASE_URL, frames_per_clip, …)
├── logging_setup.py      — structlog config
├── dependencies.py       — OllamaClient lifecycle
│
├── providers/
│   ├── base.py           — VLMProvider Protocol
│   ├── ollama_client.py  — thin async wrapper over Ollama /api/chat + /api/tags
│   ├── minicpm_v.py      — MiniCPM-V 2.6 provider
│   └── factory.py        — name → provider registry
│
├── pipeline/
│   ├── frames.py         — ffmpeg keyframe extraction (5×, 640px JPEG)
│   ├── prompt.py         — Jinja loader rooted at ../prompts
│   └── verifier.py       — orchestrate: extract → prompt → infer → parse
│
├── schemas/
│   ├── verify.py         — VerifyRequest / VerifyResponse
│   ├── vlm_output.py     — strict Pydantic Category + VLMOutput + VLMParseError
│   └── health.py         — HealthResponse
│
└── api/v1/
    ├── verify.py         — POST /v1/verify
    └── health.py         — GET /healthz, GET /v1/models

prompts/
├── verify_v1.j2          — externalized Mongolian prompt
└── categories.md         — semantic definitions (referenced from backend too)
```

---

## API

```
POST /v1/verify
    Body: VerifyRequest = {
        clip_path: str,            # absolute path on host filesystem
        store_id?: UUID,
        camera_id?: UUID,
        provider?: str             # override default — A/B test toggle
    }
    Returns: VerifyResponse = {
        category: "browsing"|"cart_pickup"|"pocket_conceal"|"other",
        confidence: 0.0..1.0,
        reasoning: str,
        model_name: str,
        inference_latency_ms: int,
        frames_used: int
    }

GET  /healthz            { status, version, ollama_reachable, loaded_models }
GET  /v1/models          ["minicpm-v-2.6", …]   (providers we know about)
GET  /openapi.json
GET  /docs
```

---

## Testing

```bash
uv run pytest tests/unit/          # mocked Ollama via respx, fast
uv run pytest tests/integration/   # requires real Ollama + model pulled
uv run ruff format --check .
uv run ruff check .
uv run mypy src/sentry_ai
```

Integration tests are gated by `OLLAMA_BASE_URL` reachability — skipped in CI.

---

## Robustness

- **Strict output schema** — Pydantic `VLMOutput` rejects any VLM reply that
  isn't valid JSON, missing fields, or out-of-range confidence.
- **Parse retries** — up to `RETRY_ON_PARSE_ERROR` (default 2) attempts on
  malformed JSON, then **graceful fallback** to `{category: "other",
  confidence: 0.0, reasoning: "<error>"}` so the pipeline always returns
  something the backend can persist.
- **Concurrency cap** — single Ollama request at a time on laptop is the
  default; multi-request batching is Hetzner-tier optimization (Session 3).

---

## Related repos

- [sentry-backend](https://github.com/Chipmo-Sentry/sentry-backend) — calls `/v1/verify` after each clip upload
- [sentry-ingest](https://github.com/Chipmo-Sentry/sentry-ingest) — receives camera streams (M2)

Platform overview: [Sentry-v.3 README](../README.md) (local workspace)
