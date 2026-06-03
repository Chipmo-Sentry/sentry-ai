"""FastAPI application entrypoint for sentry-ai."""

import asyncio
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

    # M1-LIVE L2: auto-start live workers from settings
    if settings.live_auto_start:
        manager = get_manager()
        for cam_id, rtsp_url in _parse_auto_start(settings.live_auto_start):
            log.info("live.auto_start", camera_id=cam_id, rtsp_url=rtsp_url)
            manager.start_camera(cam_id, rtsp_url, frame_skip=settings.live_frame_skip)

    # AI-node heartbeat (only runs when paired).
    from sentry_ai.heartbeat import heartbeat_loop

    hb_task = asyncio.create_task(heartbeat_loop())

    yield

    log.info("stopping")
    hb_task.cancel()
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
        title="Chipmo Sentry AI",
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)
    app.include_router(health_v1.router)
    app.include_router(verify_v1.router)
    app.include_router(live_v1.router)
    return app


app = create_app()
