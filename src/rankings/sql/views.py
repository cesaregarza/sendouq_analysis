from __future__ import annotations

"""Helpers for Postgres materialized views used by the rankings pipeline."""

import logging
import re

import sqlalchemy as db
from sqlalchemy.engine import Engine

from rankings.sql.constants import SCHEMA as DEFAULT_SCHEMA

logger = logging.getLogger(__name__)

_VALID_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

CHECK_TABLE_EXISTS_QUERY = """
SELECT EXISTS (
    SELECT 1
    FROM information_schema.tables
    WHERE table_schema = :schema
      AND table_name = :table_name
);
"""


def _validate_schema(schema: str) -> str:
    if not schema:
        raise ValueError("schema must be non-empty")
    schema = schema.strip()
    if not _VALID_IDENTIFIER_RE.match(schema):
        raise ValueError(f"Invalid schema name: {schema!r}")
    return schema


def _event_times_view_sql(schema: str) -> str:
    return f"""
CREATE MATERIALIZED VIEW IF NOT EXISTS {schema}.tournament_event_times AS
SELECT
  t.tournament_id,
  (CASE WHEN t.start_time_ms < 1000000000000 THEN t.start_time_ms*1000 ELSE t.start_time_ms END)::bigint AS start_ms,
  (CASE
     WHEN MAX(m.last_game_finished_at_ms) IS NULL THEN NULL
     WHEN MAX(m.last_game_finished_at_ms) < 1000000000000 THEN MAX(m.last_game_finished_at_ms)*1000
     ELSE MAX(m.last_game_finished_at_ms)
   END)::bigint AS end_ms,
  COALESCE(
    (CASE
       WHEN MAX(m.last_game_finished_at_ms) IS NULL THEN NULL
       WHEN MAX(m.last_game_finished_at_ms) < 1000000000000 THEN MAX(m.last_game_finished_at_ms)*1000
       ELSE MAX(m.last_game_finished_at_ms)
     END),
    (CASE WHEN t.start_time_ms < 1000000000000 THEN t.start_time_ms*1000 ELSE t.start_time_ms END)
  )::bigint AS event_ms,
  t.is_ranked
FROM {schema}.tournaments t
LEFT JOIN {schema}.matches m ON m.tournament_id = t.tournament_id
GROUP BY t.tournament_id, t.start_time_ms, t.is_ranked;
"""


def _event_times_indexes(schema: str) -> tuple[str, ...]:
    return (
        f"CREATE UNIQUE INDEX IF NOT EXISTS tet_pk ON {schema}.tournament_event_times(tournament_id)",
        f"CREATE INDEX IF NOT EXISTS tet_event_ms ON {schema}.tournament_event_times(event_ms)",
        f"CREATE INDEX IF NOT EXISTS tet_event_ms_rk ON {schema}.tournament_event_times(event_ms, is_ranked)",
    )


def _refresh_event_times_sql(schema: str) -> str:
    return f"REFRESH MATERIALIZED VIEW {schema}.tournament_event_times"


def _missing_tables(
    connection: db.engine.Connection, schema: str, tables: tuple[str, ...]
) -> list[str]:
    missing: list[str] = []
    for table_name in tables:
        exists = connection.execute(
            db.text(CHECK_TABLE_EXISTS_QUERY),
            {"schema": schema, "table_name": table_name},
        ).scalar()
        if not exists:
            missing.append(table_name)
    return missing


def ensure_tournament_event_times_view(
    engine: Engine, *, schema: str = DEFAULT_SCHEMA
) -> None:
    """Ensure the tournament_event_times materialized view exists and is refreshed."""
    if not hasattr(engine, "connect"):
        logging.getLogger(__name__).debug(
            "Skipping tournament_event_times refresh; engine lacks connect()"
        )
        return
    schema = _validate_schema(schema)
    try:
        with engine.connect() as connection:
            missing = _missing_tables(
                connection, schema, ("tournaments", "matches")
            )
        if missing:
            logger.info(
                "Skipping tournament_event_times materialized view because missing tables: %s",
                ", ".join(sorted(missing)),
            )
            return

        with engine.begin() as connection:
            logger.info(
                "Ensuring tournament_event_times materialized view exists"
            )
            connection.execute(db.text(_event_times_view_sql(schema)))
            for statement in _event_times_indexes(schema):
                connection.execute(db.text(statement))

        with engine.connect().execution_options(  # type: ignore[arg-type]
            isolation_level="AUTOCOMMIT"
        ) as connection:
            logger.info("Refreshing tournament_event_times materialized view")
            connection.execute(db.text(_refresh_event_times_sql(schema)))
            logger.info("tournament_event_times materialized view refreshed")
    except Exception:  # pragma: no cover - surfaced to caller for logging
        logger.exception(
            "Failed to ensure tournament_event_times materialized view is up to date"
        )
        raise
