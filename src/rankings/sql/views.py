from __future__ import annotations

"""Helpers for Postgres materialized views in the rankings schema."""

import logging

from sqlalchemy.engine import Engine

from rankings.sql.constants import SCHEMA as RANKINGS_SCHEMA
from sendouq_analysis.sql.views import ensure_tournament_event_times_view as _ensure


def ensure_rankings_tournament_event_times_view(engine: Engine) -> None:
    """Ensure tournament_event_times view in the rankings schema is up to date."""
    if not hasattr(engine, "connect"):
        logging.getLogger(__name__).debug(
            "Skipping tournament_event_times refresh; engine lacks connect()"
        )
        return
    _ensure(engine, schema=RANKINGS_SCHEMA)
