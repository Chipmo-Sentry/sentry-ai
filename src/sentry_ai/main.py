"""FastAPI application entrypoint for sentry-ai."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

import structlog
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from sentry_ai import __version__
from sentry_ai.api.v1 import health as health_v1
from sentry_ai.api.v1 import verify as verify_v1
from sentry_ai.dependencies import close_client
from sentry_ai.logging_setup import configure_logging, get_logger
from sentry_ai.settings import get_settings


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
    yield
    log.info("stopping")
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
    return app


app = create_app()
