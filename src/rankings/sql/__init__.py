"""SQL utilities for the rankings module.

This package defines:
- Schema constants (configurable via env)
- SQLAlchemy models for the rankings schema
- Engine/session helpers
- Lightweight loaders that return Polars DataFrames compatible with
  `rankings.core` engines (player-level only)

Environment variables:
- RANKINGS_DB_SCHEMA: default "rankings" (set to "comp_rankings" if desired)
- RANKINGS_DATABASE_URL or DATABASE_URL: SQLAlchemy URL for the DB engine
"""

from __future__ import annotations

from rankings.sql import models
from rankings.sql.constants import SCHEMA
from rankings.sql.engine import (
    create_all,
    create_engine,
    create_session_factory,
    ensure_schema,
)
from rankings.sql.load import (
    load_core_tables,
    load_matches_df,
    load_player_appearance_teams_df,
    load_player_appearances_df,
    load_player_ranking_stats_df,
    load_players_df,
)

__all__ = [
    # Config
    "SCHEMA",
    # Engine helpers
    "create_engine",
    "create_session_factory",
    "ensure_schema",
    "create_all",
    # Loaders
    "load_core_tables",
    "load_matches_df",
    "load_players_df",
    "load_player_appearance_teams_df",
    "load_player_appearances_df",
    "load_player_ranking_stats_df",
    # Models submodule
    "models",
]
