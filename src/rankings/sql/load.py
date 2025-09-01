from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import polars as pl
from sqlalchemy import text
from sqlalchemy.engine import Engine

from rankings.sql.constants import SCHEMA


def _read_sql(
    engine: Engine, sql: str, params: Optional[dict[str, Any]] = None
) -> pl.DataFrame:
    """Read SQL into a Polars DataFrame via pandas for compatibility.

    This uses pandas as an intermediary to avoid adding a new dependency layer.
    """
    with engine.connect() as conn:
        pdf = pd.read_sql_query(text(sql), conn, params=params)
    return pl.from_pandas(pdf) if not pdf.empty else pl.DataFrame([])


def load_matches_df(
    engine: Engine,
    *,
    since_ms: Optional[int] = None,
    until_ms: Optional[int] = None,
    only_ranked: bool = False,
) -> pl.DataFrame:
    """Load matches as a Polars DataFrame for rankings.core consumption.

    Columns produced (if present):
    - match_id, tournament_id, stage_id, group_id, round_id
    - match_number, status, last_game_finished_at, match_created_at
    - team1_id, team1_position, team1_score
    - team2_id, team2_position, team2_score
    - winner_team_id, loser_team_id, is_bye
    """

    def _to_db_seconds(v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        try:
            iv = int(v)
        except Exception:
            return None
        # Heuristic: treat values > 10^12 as milliseconds
        return iv // 1000 if iv > 1_000_000_000_000 else iv

    where = []
    params: dict[str, Any] = {}
    # Prefer last_game_finished_at_ms, but fall back to created_at_ms when missing
    time_expr = "COALESCE(m.last_game_finished_at_ms, m.created_at_ms)"
    since_ts = _to_db_seconds(since_ms)
    until_ts = _to_db_seconds(until_ms)
    if since_ts is not None:
        where.append(f"{time_expr} >= :since_ts")
        params["since_ts"] = since_ts
    if until_ts is not None:
        where.append(f"{time_expr} <= :until_ts")
        params["until_ts"] = until_ts
    if only_ranked:
        where.append("COALESCE(t.is_ranked, false) = true")

    where_clause = f"WHERE {' AND '.join(where)}" if where else ""

    join_clause = (
        "JOIN {SCHEMA}.tournaments t ON t.tournament_id = m.tournament_id"
        if only_ranked
        else ""
    )

    sql = f"""
        SELECT
            m.match_id,
            m.tournament_id,
            m.stage_id,
            m.group_id,
            m.round_id,
            m.number AS match_number,
            m.status,
            m.last_game_finished_at_ms AS last_game_finished_at,
            m.created_at_ms AS match_created_at,
            m.team1_id,
            m.team1_position,
            m.team1_score,
            m.team2_id,
            m.team2_position,
            m.team2_score,
            m.winner_team_id,
            m.loser_team_id,
            COALESCE(m.is_bye, false) AS is_bye
        FROM {SCHEMA}.matches m
        {join_clause}
        {where_clause}
    """.format(
        SCHEMA=SCHEMA, join_clause=join_clause, where_clause=where_clause
    )

    df = _read_sql(engine, sql, params)
    if df.is_empty():
        return df
    # Normalize dtypes for joins and engine expectations
    int_cols = [
        "match_id",
        "tournament_id",
        "stage_id",
        "group_id",
        "round_id",
        "team1_id",
        "team1_position",
        "team1_score",
        "team2_id",
        "team2_position",
        "team2_score",
        "winner_team_id",
        "loser_team_id",
    ]
    cast_map = {
        c: pl.col(c).cast(pl.Int64, strict=False)
        for c in int_cols
        if c in df.columns
    }
    # Booleans and timestamps remain as-is; ensure is_bye boolean
    if "is_bye" in df.columns:
        cast_map["is_bye"] = pl.col("is_bye").cast(pl.Boolean, strict=False)
    if cast_map:
        df = df.with_columns(list(cast_map.values()))
    # Ensure timestamps are floats (assume DB stores seconds since epoch)
    if "last_game_finished_at" in df.columns:
        df = df.with_columns(
            pl.col("last_game_finished_at").cast(pl.Float64, strict=False)
        )
    if "match_created_at" in df.columns:
        df = df.with_columns(
            pl.col("match_created_at").cast(pl.Float64, strict=False)
        )
    return df


def load_players_df(engine: Engine) -> pl.DataFrame:
    """Load player roster entries with usernames for rankings.core.

    Columns produced:
    - tournament_id, team_id, user_id, username, discord_id, country, roster_created_at, is_owner
    """
    sql = f"""
        SELECT
            r.tournament_id,
            r.team_id,
            r.player_id AS user_id,
            p.display_name AS username,
            p.discord_id,
            p.country,
            r.joined_at_ms AS roster_created_at,
            COALESCE(r.is_owner, false) AS is_owner
        FROM {SCHEMA}.roster_entries r
        JOIN {SCHEMA}.players p ON p.player_id = r.player_id
    """

    df = _read_sql(engine, sql)
    if df.is_empty():
        return df
    # Normalize key columns to Int64
    cast_map = {
        "tournament_id": pl.col("tournament_id").cast(pl.Int64, strict=False),
        "team_id": pl.col("team_id").cast(pl.Int64, strict=False),
        "user_id": pl.col("user_id").cast(pl.Int64, strict=False),
    }
    # is_owner boolean
    if "is_owner" in df.columns:
        cast_map["is_owner"] = pl.col("is_owner").cast(pl.Boolean, strict=False)
    df = df.with_columns(list(cast_map.values()))
    return df


def load_core_tables(
    engine: Engine,
    *,
    since_ms: Optional[int] = None,
    until_ms: Optional[int] = None,
    only_ranked: bool = False,
) -> dict[str, pl.DataFrame]:
    """Return the two DataFrames used by rankings.core engines.

    Returns a dict with keys: "matches", "players".
    """
    matches = load_matches_df(
        engine, since_ms=since_ms, until_ms=until_ms, only_ranked=only_ranked
    )
    players = load_players_df(engine)
    return {"matches": matches, "players": players}


def load_player_appearances_df(engine: Engine) -> pl.DataFrame:
    """Load per-match player appearances from the database.

    Returns a DataFrame with columns: tournament_id, match_id, user_id.
    """
    sql = f"""
        SELECT
            tournament_id,
            match_id,
            player_id AS user_id
        FROM {SCHEMA}.player_appearances
    """
    df = _read_sql(engine, sql)
    if df.is_empty():
        return df
    df = df.with_columns(
        [
            pl.col("tournament_id").cast(pl.Int64, strict=False),
            pl.col("match_id").cast(pl.Int64, strict=False),
            pl.col("user_id").cast(pl.Int64, strict=False),
        ]
    ).unique(subset=["tournament_id", "match_id", "user_id"])
    return df


def load_player_appearance_teams_df(engine: Engine) -> pl.DataFrame:
    """Load cached team assignments for appearances from the database.

    Returns a DataFrame with columns: tournament_id, match_id, user_id, team_id.
    """
    sql = f"""
        SELECT
            tournament_id,
            match_id,
            player_id AS user_id,
            team_id
        FROM {SCHEMA}.player_appearance_teams
    """
    df = _read_sql(engine, sql)
    if df.is_empty():
        return df
    return df.with_columns(
        [
            pl.col("tournament_id").cast(pl.Int64, strict=False),
            pl.col("match_id").cast(pl.Int64, strict=False),
            pl.col("user_id").cast(pl.Int64, strict=False),
            pl.col("team_id").cast(pl.Int64, strict=False),
        ]
    ).unique(subset=["tournament_id", "match_id", "user_id"])


def load_player_ranking_stats_df(
    engine: Engine,
    *,
    build_version: str | None = None,
    calculated_at_ms: int | None = None,
) -> pl.DataFrame:
    """Load per-player ranking stats (tournament_count, last_active_ms) for runs.

    Optionally filter by build version and/or calculation timestamp.
    """
    where = []
    params: dict[str, Any] = {}
    if build_version is not None:
        where.append("build_version = :bv")
        params["bv"] = build_version
    if calculated_at_ms is not None:
        where.append("calculated_at_ms = :ts")
        params["ts"] = int(calculated_at_ms)
    where_clause = f"WHERE {' AND '.join(where)}" if where else ""

    sql = f"""
        SELECT
            player_id,
            calculated_at_ms,
            build_version,
            tournament_count,
            last_active_ms
        FROM {SCHEMA}.player_ranking_stats
        {where_clause}
    """
    df = _read_sql(engine, sql, params)
    if df.is_empty():
        return df
    return df.with_columns(
        [
            pl.col("player_id").cast(pl.Int64, strict=False),
            pl.col("calculated_at_ms").cast(pl.Int64, strict=False),
            pl.col("tournament_count").cast(pl.Int64, strict=False),
            pl.col("last_active_ms").cast(pl.Int64, strict=False),
        ]
    )
