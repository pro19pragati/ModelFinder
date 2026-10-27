"""SQLite / SQLAlchemy setup.

The database stores *experiment metadata only* (never the raw datasets), which
keeps the file small even when the uploaded CSV is large. Uploaded files live on
disk under ``storage/uploads`` and are referenced by ``Dataset.stored_path``.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import settings
from logging_config import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None

# The experiment workers run in background threads; serialising writes through a
# lock avoids SQLite "database is locked" errors and keeps the history ordered.
db_write_lock = threading.RLock()


def _create_engine(url: str) -> Engine:
    is_sqlite = url.startswith("sqlite")
    kwargs: dict = {"echo": False}
    if is_sqlite:
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    engine = create_engine(url, **kwargs)
    if is_sqlite:
        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, _connection_record):  # pragma: no cover
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()
    return engine


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        Path(settings.storage_dir).mkdir(parents=True, exist_ok=True)
        _engine = _create_engine(settings.database_url)
        logger.info("Database engine created for %s", settings.database_url)
    return _engine


def get_session_factory() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)
    return _SessionLocal


def init_db(engine: Optional[Engine] = None) -> None:
    """Create tables if they do not exist yet."""
    from models import (  # noqa: WPS433 (import here to avoid circular imports)
        AgentDecision,
        Dataset,
        Experiment,
        ExperimentSession,
    )

    target = engine or get_engine()
    Base.metadata.create_all(
        bind=target,
        tables=[
            Dataset.__table__,
            ExperimentSession.__table__,
            Experiment.__table__,
            AgentDecision.__table__,
        ],
    )
    logger.info("Database schema ready")


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a scoped session."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Context manager for background workers (commits or rolls back)."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
