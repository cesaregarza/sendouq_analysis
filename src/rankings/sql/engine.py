from __future__ import annotations

import os
from typing import Optional

from sqlalchemy import create_engine as _sa_create_engine
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker

from .constants import SCHEMA

Base = declarative_base()


def _build_url_from_env() -> str | None:
    """Construct a Postgres URL from component env vars.

    Recognized variables (RANKINGS_* preferred, falls back to POSTGRES_*):
      - HOST, PORT (default 5432)
      - NAME (database name; default 'rankings_db')
      - USER, PASSWORD
      - SSLMODE (optional)
    """
    # Accept both canonical RANKINGS_DB_* and legacy RANKING_DB_* envs
    host = (
        os.getenv("RANKINGS_DB_HOST")
        or os.getenv("RANKING_DB_HOST")
        or os.getenv("POSTGRES_HOST")
    )
    user = (
        os.getenv("RANKINGS_DB_USER")
        or os.getenv("RANKING_DB_USER")
        or os.getenv("POSTGRES_USER")
    )
    if not host or not user:
        return None
    port = (
        os.getenv("RANKINGS_DB_PORT")
        or os.getenv("RANKING_DB_PORT")
        or os.getenv("POSTGRES_PORT")
        or "5432"
    )
    name = (
        os.getenv("RANKINGS_DB_NAME")
        or os.getenv("RANKING_DB_NAME")
        or os.getenv("POSTGRES_DB")
        or "rankings_db"
    )
    password = (
        os.getenv("RANKINGS_DB_PASSWORD")
        or os.getenv("RANKING_DB_PASSWORD")
        or os.getenv("POSTGRES_PASSWORD")
        or ""
    )
    sslmode = (
        os.getenv("RANKINGS_DB_SSLMODE")
        or os.getenv("RANKING_DB_SSLMODE")
        or os.getenv("POSTGRES_SSLMODE")
    )

    auth = f"{user}:{password}" if password != "" else f"{user}"
    url = f"postgresql://{auth}@{host}:{port}/{name}"
    if sslmode:
        url = f"{url}?sslmode={sslmode}"
    return url


def create_engine(url: Optional[str] = None, *, echo: bool = False) -> Engine:
    """Create a SQLAlchemy engine.

    Resolution order for URL:
    - explicit ``url`` arg
    - env ``RANKINGS_DATABASE_URL``
    - env ``DATABASE_URL``
    """
    database_url = (
        url
        or os.getenv("RANKINGS_DATABASE_URL")
        or os.getenv("DATABASE_URL")
        or _build_url_from_env()
    )
    if not database_url:
        raise RuntimeError(
            "No database URL provided. Set RANKINGS_DATABASE_URL or DATABASE_URL, "
            "or provide component env vars (RANKINGS_DB_HOST/USER/[PASSWORD]/[NAME]/[PORT]/[SSLMODE])."
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
