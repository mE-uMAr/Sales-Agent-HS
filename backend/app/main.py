"""Application entry point.

Run with::

    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.routes_admin import public_router
from app.api.routes_admin import router as admin_router
from app.api.routes_chat import router as chat_router
from app.api.security import limiter
from app.config import get_settings
from app.leads.outbox import OutboxWorker
from app.leads.service import close_lead_service, get_lead_service
from app.observability import configure_logging, get_logger
from app.persistence.db import dispose_engine, init_db

logger = get_logger(__name__)


async def _housekeeping_loop(app: FastAPI) -> None:
    """Close abandoned conversations and keep the rate-limit table small."""
    settings = get_settings()
    interval = max(60, settings.session_idle_minutes * 60 // 2)

    from app.chat.service import get_chat_service

    while True:
        try:
            await asyncio.sleep(interval)
            await get_chat_service().sweep_idle_sessions()
            limiter.prune()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("housekeeping iteration failed")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.ensure_runtime_dirs()

    await init_db()

    if settings.auto_index_on_startup:
        _ensure_index()

    lead_service = get_lead_service()
    outbox: OutboxWorker | None = None
    if settings.outbox_enabled and lead_service.forwarding_enabled:
        outbox = OutboxWorker(lead_service.sink)
        outbox.start()

    housekeeping = asyncio.create_task(_housekeeping_loop(app), name="housekeeping")

    logger.info(
        "service ready",
        extra={
            "version": __version__,
            "environment": settings.environment,
            "llm_provider": settings.llm_provider,
            "lead_sink": settings.lead_sink,
            "forwarding": lead_service.forwarding_enabled,
        },
    )

    if settings.is_production:
        _warn_on_insecure_defaults(settings)

    try:
        yield
    finally:
        housekeeping.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await housekeeping
        if outbox is not None:
            await outbox.stop()
        await close_lead_service()
        await dispose_engine()
        logger.info("service stopped")


def _ensure_index() -> None:
    """Build the knowledge index on first boot.

    A missing index is not fatal — the bot degrades to "I don't know" and hands
    off, which is the designed behaviour — so a failure here is logged loudly
    rather than allowed to stop the service.
    """
    from app.knowledge.ingest import build_index
    from app.knowledge.retriever import get_retriever, reset_retriever

    if get_retriever().is_ready():
        return

    logger.info("knowledge index missing; building it now")
    try:
        report = build_index(reset=True)
        reset_retriever()
        logger.info("knowledge index built at startup", extra=report.as_dict())
    except Exception:
        logger.exception(
            "could not build the knowledge index; the assistant will answer "
            "'I don't know' until this is fixed"
        )


def _warn_on_insecure_defaults(settings: object) -> None:
    problems: list[str] = []
    if settings.session_token_secret.get_secret_value() == "change_me_in_production":  # type: ignore[attr-defined]
        problems.append("SESSION_TOKEN_SECRET")
    if settings.admin_api_key.get_secret_value() == "change_me_in_production":  # type: ignore[attr-defined]
        problems.append("ADMIN_API_KEY")
    if "*" in settings.allowed_origins:  # type: ignore[attr-defined]
        problems.append("ALLOWED_ORIGINS")
    if problems:
        logger.error(
            "running in production with insecure defaults",
            extra={"unset": problems},
        )


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Hashed Systems Assistant",
        description=(
            "Retrieval-grounded sales assistant that replaces the website "
            "contact form and captures a qualified lead from every conversation."
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Widget-Key"],
        max_age=3600,
    )

    app.include_router(public_router)
    app.include_router(chat_router)
    app.include_router(admin_router)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Never leak a traceback to a visitor; log it in full on our side.
        logger.exception("unhandled error", extra={"path": request.url.path})
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "detail": "Something went wrong on our side.",
            },
        )

    return app


app = create_app()
