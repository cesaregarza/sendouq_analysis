from __future__ import annotations

import os
from typing import Optional

from sqlalchemy import create_engine as _sa_create_engine
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker

from .constants import SCHEMA

Base = declarative_base()


def create_engine(url: Optional[str] = None, *, echo: bool = False) -> Engine:
    """Create a SQLAlchemy engine.

    Resolution order for URL:
    - explicit ``url`` arg
    - env ``RANKINGS_DATABASE_URL``
    - env ``DATABASE_URL``
    """
    database_url = (
        url or os.getenv("RANKINGS_DATABASE_URL") or os.getenv("DATABASE_URL")
    )
    if not database_url:
        raise RuntimeError(
            "No database URL provided. Set RANKINGS_DATABASE_URL or DATABASE_URL."
        )
    return _sa_create_engine(database_url, echo=echo, future=True)


def create_session_factory(engine: Engine):
    """Return a configured sessionmaker bound to the engine."""
    return sessionmaker(
        bind=engine, autoflush=False, autocommit=False, future=True
    )


def ensure_schema(engine: Engine) -> None:
    """Create the rankings schema if it does not exist (idempotent)."""
    # Works in Postgres; other DBs ignore schema creation semantics
    try:
        with engine.begin() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
    except Exception:
        # Best-effort; for engines without schema support
        pass


def create_all(engine: Engine) -> None:
    """Create all tables in the rankings schema (idempotent)."""
    from . import models  # noqa: F401 - ensure models are imported

    ensure_schema(engine)
    Base.metadata.create_all(engine)
