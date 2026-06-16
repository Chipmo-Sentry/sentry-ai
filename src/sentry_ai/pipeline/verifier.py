"""End-to-end pipeline: clip path → keyframes → prompt → VLM → VLMOutput."""

import time
from pathlib import Path

from sentry_ai.pipeline.frames import extract_keyframes
from sentry_ai.pipeline.prompt import render_prompt
from sentry_ai.providers.base import VLMProvider
from sentry_ai.schemas.vlm_output import Category, VLMOutput, VLMParseError
from sentry_ai.settings import get_settings


async def verify_clip(
    *,
    clip_path: Path,
    provider: VLMProvider,
    store_context: str | None = None,
) -> tuple[VLMOutput, int, int]:
    """Run Stage 2 on a clip. Returns (output, latency_ms, frames_used)."""
    settings = get_settings()

    frames = await extract_keyframes(
        clip_path,
        count=settings.frames_per_clip,
        max_dim=settings.frame_max_dim,
        quality=settings.frame_jpeg_quality,
    )

    prompt = render_prompt("verify_v1.j2", store_context=store_context)

    from sentry_ai import runtime_config

    last_err: VLMParseError | None = None
    started = time.perf_counter()
    for _attempt in range(settings.retry_on_parse_error + 1):
        try:
            output = await provider.verify(
                frames, prompt, timeout_sec=settings.inference_timeout_sec
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            # Record the GPU run so the dashboard shows the VLM's activity history.
            runtime_config.record_vlm_verify(latency_ms)
            return output, latency_ms, len(frames)
        except VLMParseError as e:
            last_err = e
            continue

    # All retries exhausted — fall back to a neutral verdict so the pipeline
    # always returns something (better than 500ing the backend). The VLM still
    # ran on the GPU, so still count it.
    latency_ms = int((time.perf_counter() - started) * 1000)
    runtime_config.record_vlm_verify(latency_ms)
    return (
        VLMOutput(
            category=Category.other,
            confidence=0.0,
            reasoning=f"VLM parse failed after retries: {last_err}",
        ),
        latency_ms,
        len(frames),
    )
