"""FastAPI application entrypoint for sentry-ai."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

import structlog
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from sentry_ai import __version__
from sentry_ai.api.v1 import health as health_v1
from sentry_ai.api.v1 import live as live_v1
from sentry_ai.api.v1 import verify as verify_v1
from sentry_ai.dependencies import close_client
from sentry_ai.live_worker import get_manager
from sentry_ai.logging_setup import configure_logging, get_logger
from sentry_ai.runtime import is_railway_host
from sentry_ai.settings import get_settings


def _parse_auto_start(spec: str) -> list[tuple[str, str]]:
    """Parse `cam1=url1,cam2=url2` env value into [(camera_id, rtsp_url), ...]."""
    out: list[tuple[str, str]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        cam_id, _, url = item.partition("=")
        cam_id = cam_id.strip()
        url = url.strip()
        if cam_id and url:
            out.append((cam_id, url))
    return out


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    log = get_logger("sentry_ai.lifespan")
    settings = get_settings()
    log.info(
        "starting",
        version=__version__,
        environment=settings.environment,
        ollama_base_url=settings.ollama_base_url,
        default_provider=settings.default_provider,
    )

    # I6 (ADR-0029 §12): a public verdict node must not run auth-open. Raises in
    # staging/production when AI_SERVICE_TOKEN is unset — fail before serving.
    from sentry_ai.auth import assert_service_token_configured

    assert_service_token_configured(settings)

    # M1-LIVE L2: auto-start live workers from settings.
    # Red-line #2 / ADR-0007: RTSP ingest must NEVER run on Railway — refuse to
    # auto-start workers there even if LIVE_AUTO_START is set by mistake.
    if settings.live_auto_start:
        if is_railway_host():
            log.error("live.auto_start_refused_on_railway", spec=settings.live_auto_start)
        else:
            manager = get_manager()
            for cam_id, rtsp_url in _parse_auto_start(settings.live_auto_start):
                log.info("live.auto_start", camera_id=cam_id, rtsp_url=rtsp_url)
                manager.start_camera(
                    cam_id,
                    rtsp_url,
                    frame_skip=settings.live_frame_skip,
                    store_id=settings.live_auto_start_store_id,
                )

    # AI-node heartbeat — own daemon thread (resilient to event-loop starvation
    # from the GPU worker threads; an asyncio task here stops beating under load).
    from sentry_ai.heartbeat import start_heartbeat, stop_heartbeat

    start_heartbeat()

    # Node diagnostics push — rich per-stage diag (worker errors, VLM verdicts +
    # raw-on-failure) to the backend for the stage diagnostics console (docs/26).
    from sentry_ai.diag import start_diag, stop_diag

    start_diag()

    yield

    log.info("stopping")
    stop_diag()
    stop_heartbeat()
    get_manager().stop_all()
    await close_client()


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        request_id = request.headers.get("x-request-id", uuid4().hex)
        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()
        response.headers["x-request-id"] = request_id
        return response


def create_app() -> FastAPI:
    app = FastAPI(
        title="Sentry AI",
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)
    app.include_router(health_v1.router)
    app.include_router(verify_v1.router)
    app.include_router(live_v1.router)
    return app


app = create_app()
