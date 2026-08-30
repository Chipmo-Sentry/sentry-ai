"""End-to-end pipeline: clip path → keyframes → prompt → VLM → VLMOutput."""

import time
from pathlib import Path

import httpx

from sentry_ai.logging_setup import get_logger
from sentry_ai.pipeline.frames import extract_keyframes
from sentry_ai.pipeline.prompt import render_prompt
from sentry_ai.providers.base import VLMProvider
from sentry_ai.schemas.vlm_output import Category, VLMOutput, VLMParseError
from sentry_ai.settings import get_settings

log = get_logger("sentry_ai.pipeline.verifier")


async def verify_clip(
    *,
    clip_path: Path,
    provider: VLMProvider,
    store_context: str | None = None,
    behavior_hint: str | None = None,
) -> tuple[VLMOutput, int, int]:
    """Run Stage 2 on a clip. Returns (output, latency_ms, frames_used).

    `behavior_hint` is a short Mongolian summary of what the edge behavior
    engine already flagged (e.g. which concealment signals + peak risk); it
    focuses the VLM on the suspected moment WITHOUT letting the hint alone
    decide — the prompt still requires the action to be visible in the frames.
    """
    settings = get_settings()
    from sentry_ai.runtime_config import resolve_frame_max_dim, resolve_frames_per_clip

    # frames_per_clip + frame_max_dim are operator-tunable per node (central
    # control); the resolved value wins over the .env default so a superadmin edit
    # changes how many frames / what resolution the VLM sees on the NEXT breach.
    frames_per_clip = resolve_frames_per_clip(settings.frames_per_clip)
    frame_max_dim = resolve_frame_max_dim(settings.frame_max_dim)

    frames = await extract_keyframes(
        clip_path,
        count=frames_per_clip,
        max_dim=frame_max_dim,
        quality=settings.frame_jpeg_quality,
    )

    prompt = render_prompt(
        "verify_v1.j2",
        store_context=store_context,
        frame_count=frames_per_clip,
        behavior_hint=behavior_hint,
    )

    from sentry_ai import runtime_config

    last_err: VLMParseError | None = None
    started = time.perf_counter()
    # docs/33 P0-5: a COLD VLM (model paged out / first call after idle) routinely
    # exceeds the steady-state timeout, and the resulting httpx timeout used to
    # propagate as a silent scan failure until the model warmed. Retry a transport
    # error ONCE with an extended timeout (covers the cold load); a second
    # transport failure re-raises so the caller can clean up + report the miss.
    timeout_sec: int = settings.inference_timeout_sec
    transport_retried = False
    parse_failures = 0
    while parse_failures <= settings.retry_on_parse_error:
        try:
            output = await provider.verify(frames, prompt, timeout_sec=timeout_sec)
            latency_ms = int((time.perf_counter() - started) * 1000)
            # Record the GPU run so the dashboard shows the VLM's activity history.
            runtime_config.record_vlm_verify(latency_ms)
            runtime_config.record_vlm_verdict(
                category=output.category.value,
                confidence=output.confidence,
                latency_ms=latency_ms,
                frames_used=len(frames),
                parsed=True,
            )
            return output, latency_ms, len(frames)
        except VLMParseError as e:
            last_err = e
            parse_failures += 1  # transport retry deliberately doesn't consume these
            continue
        except httpx.HTTPError as e:
            if transport_retried:
                raise  # second transport failure — genuinely down, caller handles
            transport_retried = True
            timeout_sec = max(timeout_sec * 2, 90)  # cold-load headroom
            log.warning(
                "verify.vlm_transport_retry",
                error=str(e)[:160],
                retry_timeout_sec=timeout_sec,
            )
            continue

    # All retries exhausted — fall back to a neutral verdict so the pipeline
    # always returns something (better than 500ing the backend). The VLM still
    # ran on the GPU, so still count it.
    latency_ms = int((time.perf_counter() - started) * 1000)
    runtime_config.record_vlm_verify(latency_ms)
    # Capture the RAW unparseable output — the single most useful diagnostic for
    # "why does the VLM keep failing" on the stage page.
    runtime_config.record_vlm_verdict(
        category="other",
        confidence=0.0,
        latency_ms=latency_ms,
        frames_used=len(frames),
        parsed=False,
        raw=(last_err.raw if last_err is not None else None),
    )
    return (
        VLMOutput(
            actions=[Category.other],
            confidence=0.0,
            reasoning=f"VLM parse failed after retries: {last_err}",
        ),
        latency_ms,
        len(frames),
    )
