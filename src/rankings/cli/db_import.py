from __future__ import annotations

"""CLI to import scraped tournament JSON payloads into the rankings database.

Parses tournament JSON files (per-ID, batch, or snapshot formats) and upserts
normalized rows into the Postgres schema. Also extracts per-match player
appearances from the players route payload when present.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List

import numpy as np
import polars as pl
from sqlalchemy.dialects.postgresql import insert as pg_insert

from rankings.core import parse_tournaments_data
from rankings.scraping.storage import (
    extract_appearances_from_players_payload as extract_appearances,
)
from rankings.sql import create_all as rankings_create_all
from rankings.sql import create_engine as rankings_create_engine
from rankings.sql import models as RM
from rankings.sql.constants import SCHEMA as RANKINGS_SCHEMA


def _find_json_files(root: Path) -> list[Path]:
    """Return sorted list of tournament JSON files under `root` (recursive).

    Matches per-ID, batch, and snapshot file patterns.
    """
    patterns = [
        "tournament_*.json",
        "tournaments_*.json",
        "tournaments_continuous_*.json",
    ]
    files: list[Path] = []
    for p in root.rglob("*.json"):
        name = p.name
        if any(Path(name).match(pat) for pat in patterns):
            files.append(p)
    # Sort for stable processing
    return sorted(files)


def _load_json_payload(path: Path):
    """Load a JSON file and return a list of tournament objects.

    Accepts payloads as a single dict or a list of dicts.
    """
    with path.open("r") as f:
        data = json.load(f)
    # Ensure list of tournament objects
    if isinstance(data, dict):
        return [data]
    elif isinstance(data, list):
        return data
    else:
        return []


def _bulk_insert(df: pl.DataFrame | None, model, engine) -> None:
    """Insert rows using SQLAlchemy Core with ON CONFLICT DO NOTHING (no pandas)."""
    if df is None or df.is_empty():
        return

    def _py(v):
        try:
            if isinstance(v, np.generic):
                return v.item()
            if isinstance(v, np.ndarray):
                return [_py(x) for x in v.tolist()]
            if isinstance(v, (list, tuple, set)):
                return [_py(x) for x in list(v)]
            if isinstance(v, dict):
                return {k: _py(x) for k, x in v.items()}
            return v
        except Exception:
            return v

    rows = [{k: _py(v) for k, v in r.items()} for r in df.iter_rows(named=True)]
    if not rows:
        return

    table = model.__table__
    stmt = pg_insert(table).values(rows).on_conflict_do_nothing()
    with engine.begin() as conn:
        conn.execute(stmt)


def _upsert_tournaments(df: pl.DataFrame | None, engine) -> None:
    """Upsert tournaments so new nullable columns (counts, jsonb) backfill existing rows."""
    if df is None or df.is_empty():
        return

    def _py(v):
        try:
            if isinstance(v, np.generic):
                return v.item()
            if isinstance(v, np.ndarray):
                return [_py(x) for x in v.tolist()]
            if isinstance(v, (list, tuple, set)):
                return [_py(x) for x in list(v)]
            if isinstance(v, dict):
                return {k: _py(x) for k, x in v.items()}
            return v
        except Exception:
            return v

    # Limit to columns that exist in the table
    table = RM.Tournament.__table__
    allowed_cols = {c.name for c in table.columns}
    # Ensure we only carry one row per tournament_id to avoid ON CONFLICT affecting same row twice
    # Prefer the last occurrence to reflect latest parsed values
    seen = {}
    for r in df.iter_rows(named=True):
        item = {k: _py(v) for k, v in r.items() if k in allowed_cols}
        tid = item.get("tournament_id")
        if tid is not None:
            seen[tid] = item
    rows = list(seen.values())
    if not rows:
        return

    stmt = pg_insert(table).values(rows)
    # Only update columns we intend to backfill (avoid clobbering is_ranked, etc.)
    backfill_cols = {
        "team_count",
        "match_count",
        "stage_count",
        "group_count",
        "round_count",
        "participated_users_count",
        # Allow updating rank flag and finalized status from fresh parses
        "is_ranked",
        "is_finalized",
        "tags",
        "meta",
    }
    present_cols = set(rows[0].keys()) if rows else set()
    update_cols = list((backfill_cols & present_cols) - {"tournament_id"})
    set_map = {col: getattr(stmt.excluded, col) for col in update_cols}
    if set_map:
        stmt = stmt.on_conflict_do_update(
            index_elements=[table.c.tournament_id], set_=set_map
        )
    else:
        stmt = stmt.on_conflict_do_nothing(
            index_elements=[table.c.tournament_id]
        )
    with engine.begin() as conn:
        conn.execute(stmt)


def _update_is_ranked(df: pl.DataFrame | None, engine) -> None:
    """Update is_ranked column from parser's settings_is_ranked without touching other fields."""
    if df is None or df.is_empty():
        return
    # Prefer explicit is_ranked if present, else map from settings_is_ranked
    cols = df.columns
    if "is_ranked" in cols:
        upd = df.select(["tournament_id", "is_ranked"])
    elif "settings_is_ranked" in cols:
        upd = df.select(
            [
                pl.col("tournament_id"),
                pl.col("settings_is_ranked")
                .cast(pl.Boolean)
                .alias("is_ranked"),
            ]
        )
    else:
        return
    # Deduplicate by tournament_id
    upd = upd.unique(subset=["tournament_id"])
    # Build param list
    rows = upd.iter_rows(named=True)
    from sqlalchemy import text as sqltext

    with engine.begin() as conn:
        conn.execute(
            sqltext(
                f"UPDATE {RANKINGS_SCHEMA}.tournaments AS t SET is_ranked = :is_ranked WHERE t.tournament_id = :tournament_id"
            ),
            list(rows),
        )


def import_file(
    engine, payload: list[dict], max_remaining: int | None = None
) -> int:
    """Parse a single JSON payload and write to Postgres.

    Args:
        engine: SQLAlchemy engine connected to the rankings database.
        payload: List of tournament JSON objects (already loaded).
        max_remaining: Optional cap on how many tournaments to ingest from this payload.

    Returns:
        Number of tournaments ingested from this payload.
    """
    # If caller wants to cap tournaments, slice payload first
    if max_remaining is not None and max_remaining >= 0:
        payload = payload[:max_remaining]

    # Ensure new columns exist (forward-compatible schema upgrade).
    # If lacking ALTER privilege (typical for app user), skip silently.
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql(
                f"""
                ALTER TABLE {RANKINGS_SCHEMA}.tournaments
                  ADD COLUMN IF NOT EXISTS team_count INTEGER,
                  ADD COLUMN IF NOT EXISTS match_count INTEGER,
                  ADD COLUMN IF NOT EXISTS stage_count INTEGER,
                  ADD COLUMN IF NOT EXISTS group_count INTEGER,
                  ADD COLUMN IF NOT EXISTS round_count INTEGER,
                  ADD COLUMN IF NOT EXISTS participated_users_count INTEGER
                """
            )
    except Exception:
        # Permission denied or other non-critical error; proceed with inserts
        pass

    tables = parse_tournaments_data(payload)

    # Normalize optional frames
    tournaments = tables.get("tournaments")
    stages = tables.get("stages")
    groups = tables.get("groups")
    rounds = tables.get("rounds")
    teams = tables.get("teams")
    players = tables.get("players")
    matches = tables.get("matches")

    # Normalized dataframes we may reference later for aliases; default to None
    tdf = sdf = gdf = rdf = tmdf = pldf_players = mdf = None

    count = int(tournaments.height) if tournaments is not None else 0

    # Transform tournaments to rankings.sql.models.Tournament
    if tournaments is not None and not tournaments.is_empty():
        # tags: list stays as Python list for JSONB; meta reserved
        tdf = tournaments.with_columns(
            [
                pl.col("settings_is_ranked")
                .cast(pl.Boolean)
                .alias("is_ranked"),
                pl.lit(None).alias("format_hint"),
                pl.col("start_time").alias("start_time_ms"),
                pl.col("tournament_id").alias("tournament_id"),
                pl.col("map_picking_style"),
                pl.col("rules"),
                pl.lit(None).alias("meta"),
                # Ensure JSONB-compatible Python lists
                pl.col("tags")
                .map_elements(
                    lambda x: (list(x) if x is not None else None),
                    return_dtype=pl.List(pl.Utf8),
                )
                .alias("tags"),
                pl.col("is_finalized").cast(pl.Boolean).alias("is_finalized"),
                # Carry parser-derived counts for ranked filtering parity
                pl.col("team_count").alias("team_count"),
                pl.col("match_count").alias("match_count"),
                pl.col("stage_count").alias("stage_count"),
                pl.col("group_count").alias("group_count"),
                pl.col("round_count").alias("round_count"),
                pl.col("participated_users_count").alias(
                    "participated_users_count"
                ),
            ]
        ).select(
            [
                "tournament_id",
                "name",
                "description",
                "start_time_ms",
                "is_finalized",
                "is_ranked",
                "format_hint",
                "map_picking_style",
                "rules",
                "tags",
                "meta",
                "team_count",
                "match_count",
                "stage_count",
                "group_count",
                "round_count",
                "participated_users_count",
            ]
        )
        _upsert_tournaments(tdf, engine)

    if stages is not None and not stages.is_empty():
        sdf = stages.with_columns(
            [
                pl.col("tournament_id").alias("tournament_id"),
                pl.col("stage_id").alias("stage_id"),
                pl.col("stage_name").alias("name"),
                pl.col("stage_number").alias("number"),
                pl.col("stage_type").alias("type"),
            ]
        )
        # Flatten settings already done in parser; build JSON dict from columns starting with setting_
        setting_cols = [c for c in sdf.columns if c.startswith("setting_")]
        if setting_cols:

            def _to_settings(d: dict) -> dict:
                def _py(v):
                    try:
                        # numpy scalars
                        if hasattr(v, "item"):
                            return v.item()
                        # numpy arrays / sequences
                        if hasattr(v, "tolist"):
                            return v.tolist()
                        # polars list -> python list
                        if isinstance(v, (tuple, set)):
                            return list(v)
                        return v
                    except Exception:
                        return v

                return {
                    k.replace("setting_", ""): _py(d[k])
                    for k in d.keys()
                    if d.get(k) is not None
                }

            sdf = sdf.with_columns(
                pl.struct(setting_cols)
                .map_elements(_to_settings, return_dtype=pl.Object)
                .alias("settings")
            ).select(
                [
                    "stage_id",
                    "tournament_id",
                    "name",
                    "number",
                    "type",
                    "settings",
                ]
            )
        else:
            sdf = sdf.select(
                [
                    "stage_id",
                    "tournament_id",
                    "name",
                    "number",
                    "type",
                ]
            ).with_columns(pl.lit(None).alias("settings"))
        _bulk_insert(sdf, RM.Stage, engine)

    if groups is not None and not groups.is_empty():
        gdf = groups.rename(
            {"group_id": "group_id", "group_number": "number"}
        ).select(["group_id", "stage_id", "number"])
        _bulk_insert(gdf, RM.Group, engine)

    if rounds is not None and not rounds.is_empty():
        rdf = rounds.select(
            [
                pl.col("round_id"),
                pl.col("stage_id"),
                pl.col("group_id"),
                pl.col("round_number").alias("number"),
                pl.col("maps_count"),
                pl.col("maps_type"),
            ]
        )
        _bulk_insert(rdf, RM.Round, engine)

    if teams is not None and not teams.is_empty():
        tmdf = teams.select(
            [
                pl.col("team_id"),
                pl.col("tournament_id"),
                pl.col("team_name").alias("name"),
                pl.col("seed"),
                pl.col("prefers_not_to_host"),
                pl.col("no_screen"),
                pl.col("dropped_out"),
                pl.col("created_at").alias("created_at_ms"),
            ]
        )
        _bulk_insert(tmdf, RM.TournamentTeam, engine)

    if players is not None and not players.is_empty():
        # Insert players (distinct)
        pldf_players = (
            players.select(
                [
                    pl.col("user_id").alias("player_id"),
                    pl.col("username").alias("display_name"),
                    pl.col("discord_id"),
                    pl.col("country").cast(pl.Utf8).alias("country"),
                ]
            )
            .with_columns(
                pl.col("country")
                .str.strip_chars()
                .str.to_uppercase()
                .str.slice(0, 2)
            )
            .unique(subset=["player_id"])
        )
        _bulk_insert(pldf_players, RM.Player, engine)

        # Roster entries
        redf = players.select(
            [
                pl.col("tournament_id"),
                pl.col("team_id"),
                pl.col("user_id").alias("player_id"),
                pl.col("is_owner"),
                pl.col("roster_created_at").alias("joined_at_ms"),
            ]
        )
        _bulk_insert(redf, RM.RosterEntry, engine)

        # Note: legacy PlayerAlias table is deprecated in favor of unified external_ids.
        # Alias rows for players are now handled via ExternalID below.

    if matches is not None and not matches.is_empty():
        mdf = matches.select(
            [
                pl.col("match_id"),
                pl.col("tournament_id"),
                pl.col("stage_id"),
                pl.col("group_id"),
                pl.col("round_id"),
                pl.col("match_number").alias("number"),
                # Status might be int or string across payloads; cast to string
                pl.col("status").cast(pl.Utf8).alias("status"),
                pl.col("match_created_at").alias("created_at_ms"),
                pl.col("last_game_finished_at").alias(
                    "last_game_finished_at_ms"
                ),
                pl.col("team1_id"),
                pl.col("team1_position"),
                pl.col("team1_score"),
                pl.col("team2_id"),
                pl.col("team2_position"),
                pl.col("team2_score"),
                pl.col("winner_team_id"),
                pl.col("loser_team_id"),
                pl.col("is_bye"),
            ]
        )
        _bulk_insert(mdf, RM.Match, engine)

    # External ID mapping for all entity types (no UUIDs)
    def _insert_alias(
        df: pl.DataFrame | None, entity_type: str, id_col: str
    ) -> None:
        if df is None or df.is_empty():
            return
        cols = [
            pl.lit(entity_type).alias("entity_type"),
            pl.lit("sendou").alias("provider"),
            pl.col(id_col).alias("internal_id"),
            pl.col(id_col).cast(pl.Utf8).alias("external_id"),
        ]
        adf = df.select(cols).unique(
            subset=["provider", "entity_type", "external_id"]
        )
        _bulk_insert(adf, RM.ExternalID, engine)

    # Use transformed DataFrames (IDs only)
    _insert_alias(tdf, "tournament", "tournament_id")
    _insert_alias(tmdf, "team", "team_id")
    _insert_alias(pldf_players, "player", "player_id")
    _insert_alias(mdf, "match", "match_id")
    _insert_alias(sdf, "stage", "stage_id")
    _insert_alias(gdf, "group", "group_id")
    _insert_alias(rdf, "round", "round_id")

    # Player appearances from players route payload (source of truth for who played)
    try:
        appearance_rows: list[dict] = []
        for entry in payload:
            tournament_node = (
                entry.get("tournament", {}) if isinstance(entry, dict) else {}
            )
            ctx = (
                tournament_node.get("ctx", {})
                if isinstance(tournament_node, dict)
                else {}
            )
            tournament_id = ctx.get("id")
            if tournament_id is None:
                continue
            players_payload = entry.get("player_matches")
            if not players_payload:
                continue
            appearance_rows.extend(
                extract_appearances(int(tournament_id), players_payload)
            )
        if appearance_rows:
            appearances_df = pl.DataFrame(appearance_rows)
            # Normalize schema and rename user_id -> player_id for DB
            appearances_df = appearances_df.select(
                [
                    pl.col("tournament_id").cast(pl.Int64, strict=False),
                    pl.col("match_id").cast(pl.Int64, strict=False),
                    pl.col("user_id")
                    .cast(pl.Int64, strict=False)
                    .alias("player_id"),
                ]
            ).unique(subset=["tournament_id", "match_id", "player_id"])
            # Filter to known players to avoid FK violations; unknowns are rare
            try:
                unique_pids = (
                    appearances_df.select("player_id")
                    .unique()
                    .to_series()
                    .to_list()
                )
                if unique_pids:
                    from sqlalchemy import text as sqltext

                    with engine.connect() as conn:
                        rows = conn.execute(
                            sqltext(
                                f"SELECT player_id FROM {RANKINGS_SCHEMA}.players WHERE player_id = ANY(:ids)"
                            ),
                            {"ids": unique_pids},
                        ).fetchall()
                    known = {int(r[0]) for r in rows}
                    if known:
                        appearances_df = appearances_df.filter(
                            pl.col("player_id").is_in(list(known))
                        )
                    else:
                        appearances_df = pl.DataFrame([])
            except Exception:
                # Best effort; if the filter fails, fall back to raw insert (may error)
                pass
            _bulk_insert(appearances_df, RM.PlayerAppearance, engine)
    except Exception:
        # Non-fatal: continue if appearances missing or malformed
        pass

    return count


def main(argv: list[str] | None = None) -> None:
    """Entry point: import JSON files to the database based on CLI args."""
    parser = argparse.ArgumentParser(
        description="Import local tournament JSON data into the rankings database"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/tournaments",
        help="Directory with saved tournament JSON files (recursive)",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=os.getenv("RANKINGS_DATABASE_URL") or os.getenv("DATABASE_URL"),
        help="Database URL (overrides env)",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default=os.getenv("RANKINGS_DB_SCHEMA", RANKINGS_SCHEMA),
        help="Database schema (rankings or comp_rankings)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to import (file-level)",
    )
    parser.add_argument(
        "--max-tournaments",
        type=int,
        default=None,
        help="Cap total tournaments ingested across all files",
    )
    parser.add_argument(
        "--sslmode",
        type=str,
        choices=[
            "disable",
            "allow",
            "prefer",
            "require",
            "verify-ca",
            "verify-full",
        ],
        default=None,
        help="Set libpq sslmode in the connection URL (e.g., disable for local dev)",
    )

    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    # Initialize DB
    db_url = args.db_url
    if db_url and args.sslmode:
        # Append/override sslmode query param in URL
        from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

        parts = urlparse(db_url)
        q = dict(parse_qsl(parts.query, keep_blank_values=True))
        q["sslmode"] = args.sslmode
        parts = parts._replace(query=urlencode(q))
        db_url = urlunparse(parts)

    # If no URL but sslmode provided, set component env var so engine builder picks it up
    if not db_url and args.sslmode:
        os.environ["RANKINGS_DB_SSLMODE"] = args.sslmode
    engine = rankings_create_engine(db_url)
    rankings_create_all(engine)

    files = _find_json_files(data_dir)
    if args.limit:
        files = files[: args.limit]

    total_tournaments = 0
    for i, path in enumerate(files, 1):
        try:
            payload = _load_json_payload(path)
            if not payload:
                continue
            remaining = None
            if args.max_tournaments is not None:
                remaining = max(args.max_tournaments - total_tournaments, 0)
                if remaining == 0:
                    print("Reached max tournaments cap; stopping.")
                    break
            ingested = import_file(engine, payload, max_remaining=remaining)
            total_tournaments += ingested
            print(
                f"[{i}/{len(files)}] Imported {ingested} tournaments from {path.name}"
            )
        except Exception as e:
            print(f"[{i}/{len(files)}] ERROR importing {path}: {e}")

    print(f"Done. Total tournaments ingested: {total_tournaments}")


if __name__ == "__main__":
    main()
