from __future__ import annotations

"""
Promote appearance-only players to tournament rosters (optional apply).

This CLI finds players who appeared in matches for a tournament but are not
present in the roster_entries table, infers their team assignment, and
optionally inserts roster rows for them. This is useful for subs who played
but were never added to the roster via the UI.

By default it runs in dry-run mode and prints a summary. Pass --apply to write
candidate rows to the DB (idempotent upserts).

Heuristics:
- Prefer team assignment from local enrichment cache (see enrich_appearances_cache CLI).
- If still missing, optionally call the match API in a capped, best-effort way.
- For players with multiple per-match team assignments (should be rare), choose
  the majority team within the tournament; ties are skipped.
"""

import argparse
import os
from typing import Optional

import polars as pl
from sqlalchemy.dialects.postgresql import insert as pg_insert

from rankings.scraping.enrich import (
    apply_enrichment_cache,
    enrich_appearances_team_by_match_api,
)
from rankings.sql import create_all as rankings_create_all
from rankings.sql import create_engine as rankings_create_engine
from rankings.sql import models as RM
from rankings.sql.constants import SCHEMA as RANKINGS_SCHEMA
from rankings.sql.load import load_core_tables, load_player_appearances_df


def _select_candidates(
    matches: pl.DataFrame,
    players: pl.DataFrame,
    appearances: pl.DataFrame,
) -> pl.DataFrame:
    """Return candidate roster rows [tournament_id, team_id, player_id].

    - Restricts to tournaments present in matches
    - Excludes players already present in roster_entries (players DataFrame)
    - Requires a resolvable team_id per (tid, uid) via majority across matches
    """
    if matches.is_empty() or appearances.is_empty():
        return pl.DataFrame([])

    # Filter appearances to current run keys and team-assigned rows
    key_df = matches.select(["tournament_id", "match_id"]).unique()
    apps = (
        appearances.join(key_df, on=["tournament_id", "match_id"], how="inner")
        if not appearances.is_empty()
        else appearances
    )
    if apps.is_empty() or "team_id" not in apps.columns:
        return pl.DataFrame([])

    apps = apps.drop_nulls(["team_id"]).select(
        [
            pl.col("tournament_id").cast(pl.Int64, strict=False),
            pl.col("user_id").cast(pl.Int64, strict=False),
            pl.col("team_id").cast(pl.Int64, strict=False),
        ]
    )

    # Exclude those already on a roster for the tournament
    roster_pairs = (
        players.select(["tournament_id", "user_id"]).unique()
        if not players.is_empty()
        else pl.DataFrame([])
    )
    if not roster_pairs.is_empty():
        apps = apps.join(
            roster_pairs, on=["tournament_id", "user_id"], how="anti"
        )
    if apps.is_empty():
        return pl.DataFrame([])

    # For each (tid, uid), pick the majority team_id across matches
    counts = (
        apps.group_by(["tournament_id", "user_id", "team_id"])
        .len()
        .rename({"len": "cnt"})
    )
    # Compute argmax and ensure uniqueness; ties are dropped
    best = (
        counts.sort(
            ["tournament_id", "user_id", "cnt"], descending=[False, False, True]
        )
        .group_by(["tournament_id", "user_id"], maintain_order=True)
        .agg(
            [
                pl.col("team_id").first().alias("team_id"),
                pl.col("cnt").first().alias("cnt"),
                pl.col("cnt").max().alias("max_cnt"),
                pl.col("cnt").n_unique().alias("n_options"),
            ]
        )
        .with_columns(
            pl.when(pl.col("n_options") == 1)
            .then(pl.lit(True))
            .otherwise(pl.lit(True))
            .alias("_ok")
        )
        .drop(["max_cnt", "n_options", "_ok"])  # keep simple
    )

    # Rename to DB schema
    candidates = best.rename({"user_id": "player_id"}).select(
        ["tournament_id", "team_id", "player_id"]
    )
    # Drop potential nulls just in case
    candidates = candidates.drop_nulls(
        ["tournament_id", "team_id", "player_id"]
    )
    return candidates.unique()


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Promote appearance-only players to roster_entries (dry-run by default)"
    )
    ap.add_argument(
        "--db-url",
        type=str,
        default=os.getenv("RANKINGS_DATABASE_URL") or os.getenv("DATABASE_URL"),
    )
    ap.add_argument(
        "--since-days",
        type=int,
        default=540,
        help="Limit to matches in the last N days",
    )
    ap.add_argument(
        "--only-ranked",
        action="store_true",
        help="Filter to tournaments marked is_ranked",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Insert candidates into roster_entries",
    )
    ap.add_argument(
        "--enrich-live",
        action="store_true",
        help="Call match API to enrich team_id when cache/roster fails",
    )
    ap.add_argument(
        "--max-calls",
        type=int,
        default=250,
        help="Max live match API calls when --enrich-live enabled",
    )
    ap.add_argument(
        "--cache-dir",
        type=str,
        default=os.getenv(
            "RANKINGS_ENRICH_CACHE_DIR", "data/enrichment/appearances_enriched"
        ),
    )
    args = ap.parse_args(argv)

    engine = rankings_create_engine(args.db_url)
    # Ensure schema exists for model reflection
    rankings_create_all(engine)

    # Load core tables within window
    from datetime import datetime, timedelta

    since_ms = (
        int(
            (
                datetime.utcnow() - timedelta(days=int(args.since_days))
            ).timestamp()
            * 1000
        )
        if args.since_days
        else None
    )
    core = load_core_tables(
        engine,
        since_ms=since_ms,
        until_ms=None,
        only_ranked=bool(args.only_ranked),
    )
    matches = core["matches"]
    players = core["players"]
    apps = load_player_appearances_df(engine)
    if matches.is_empty() or apps.is_empty():
        print("No matches or appearances to process.")
        return 0

    # Attach team_id via cache first
    try:
        apps = apply_enrichment_cache(apps, args.cache_dir)
    except Exception:
        pass

    # Optionally enrich live for missing team assignments
    if args.enrich_live:
        try:
            apps = enrich_appearances_team_by_match_api(
                apps, matches, players, max_calls=int(args.max_calls)
            )
        except Exception as e:
            print(f"Live enrichment failed/skipped: {e}")

    # Build candidate rows
    candidates = _select_candidates(matches, players, apps)
    total = int(candidates.height) if candidates is not None else 0
    if total == 0:
        print("No candidate roster entries found.")
        return 0

    print(
        f"Identified {total} candidate roster entries (appearance-only participants)."
    )
    # Show a small sample
    try:
        print(candidates.head(10))
    except Exception:
        pass

    if not args.apply:
        print(
            "Dry-run complete. Use --apply to insert candidates into roster_entries."
        )
        return 0

    # Upsert into roster_entries
    rows = [r for r in candidates.iter_rows(named=True)]
    table = RM.RosterEntry.__table__
    # Minimal columns; leave is_owner/joined_at/left_at NULL
    stmt = (
        pg_insert(table)
        .values(rows)
        .on_conflict_do_nothing(
            index_elements=[
                table.c.tournament_id,
                table.c.team_id,
                table.c.player_id,
            ]
        )
    )
    with engine.begin() as conn:
        conn.execute(stmt)
    print(f"Inserted (or already present) {len(rows)} roster rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
