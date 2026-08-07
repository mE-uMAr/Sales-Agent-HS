"""Async SQLAlchemy engine, session factory and schema bootstrap.

Schema is created with ``create_all`` rather than migrations. For a single
service with a handful of tables that is the honest trade: it is one less moving
part, and the schema is small enough to recreate. Introduce Alembic the first
time you need to alter a populated table in production.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings
from app.observability import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Declarative base shared by every table in the application."""


_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


@event.listens_for(Engine, "connect")
def _configure_sqlite(dbapi_connection: object, _record: object) -> None:
    """WAL + enforced foreign keys. No-op for non-SQLite backends."""
    if type(dbapi_connection).__module__.split(".")[0] not in {"sqlite3", "aiosqlite"}:
        return
    cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
    finally:
        cursor.close()


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        settings = get_settings()
        settings.ensure_runtime_dirs()
        _engine = create_async_engine(
            settings.database_url,
            echo=False,
            pool_pre_ping=True,
            future=True,
        )
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(
            get_engine(), expire_on_commit=False, class_=AsyncSession
        )
    return _sessionmaker


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Transactional scope: commits on success, rolls back on any exception."""
    async with get_sessionmaker()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create any missing tables. Safe to call on every startup."""
    # Imported for their side effect of registering tables on Base.metadata.
    from app.chat import records as _chat_records  # noqa: F401
    from app.leads import models as _lead_models  # noqa: F401

    engine = get_engine()
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    logger.info("database ready", extra={"tables": sorted(Base.metadata.tables)})


async def dispose_engine() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None
