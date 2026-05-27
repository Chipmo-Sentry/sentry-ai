# syntax=docker/dockerfile:1.7
# ============================================================================
# Stage 1 — builder: install deps into a venv, copy source
# ============================================================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ============================================================================
# Stage 2 — runtime: minimal image with ffmpeg (NOT Ollama — that's on host)
# ============================================================================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# ffmpeg for keyframe extraction; curl for HEALTHCHECK; libgl for PIL on slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src ./src
COPY prompts ./prompts

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT:-8001}/healthz || exit 1

# Railway sets PORT dynamically; default 8001 to coexist with sentry-backend.
CMD ["sh", "-c", "exec uvicorn sentry_ai.main:app --host 0.0.0.0 --port ${PORT:-8001}"]
