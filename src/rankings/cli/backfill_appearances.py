from __future__ import annotations

"""
Backfill player appearances for tournaments by fetching the public players route.

This CLI iterates tournaments found in the DB and writes per-match player
appearances into {SCHEMA}.player_appearances. It filters out unknown players to
avoid foreign key violations and inserts idempotently.

Usage examples:
  python -m rankings.cli.backfill_appearances --all --only-ranked
  python -m rankings.cli.backfill_appearances --since-days 540 --workers 8
  python -m rankings.cli.backfill_appearances --tournament-ids 2442 1961 2130
"""

import argparse
import concurrent.futures as cf
import math
import os
from typing import Iterable, Optional

import polars as pl
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from rankings.scraping.calendar_api import fetch_tournament_players
from rankings.scraping.storage import (
    extract_appearances_from_players_payload as extract,
)
from rankings.sql import create_all as rankings_create_all
from rankings.sql import create_engine as rankings_create_engine
from rankings.sql import models as RM
from rankings.sql.constants import SCHEMA as RANKINGS_SCHEMA


def _iter_tournament_ids(
    engine,
    *,
    all_time: bool = False,
    since_days: Optional[int] = None,
    only_ranked: bool = False,
) -> list[int]:
    where = []
    params = {}
    if since_days is not None and not all_time:
        where.append(
            "COALESCE(m.last_game_finished_at_ms, m.created_at_ms) >= :since_ts"
        )
        from datetime import datetime, timedelta

        params["since_ts"] = int(
            (datetime.utcnow() - timedelta(days=int(since_days))).timestamp()
        )
    if only_ranked:
        where.append("COALESCE(t.is_ranked, false) = true")
    where_clause = f"WHERE {' AND '.join(where)}" if where else ""
    sql = f"""
        SELECT DISTINCT m.tournament_id
        FROM {RANKINGS_SCHEMA}.matches m
        JOIN {RANKINGS_SCHEMA}.tournaments t ON t.tournament_id = m.tournament_id
        {where_clause}
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    return [int(r[0]) for r in rows]


def _existing_players_lookup(engine, player_ids: list[int]) -> set[int]:
    if not player_ids:
        return set()
    # Chunk IN list for safety
    out: set[int] = set()
    with engine.connect() as conn:
        for i in range(0, len(player_ids), 1000):
            chunk = player_ids[i : i + 1000]
            q = text(
                f"SELECT player_id FROM {RANKINGS_SCHEMA}.players WHERE player_id = ANY(:ids)"
            )
            res = conn.execute(q, {"ids": chunk}).fetchall()
            out.update(int(r[0]) for r in res)
    return out


def _insert_appearances(engine, df: pl.DataFrame) -> int:
    if df is None or df.is_empty():
        return 0
    # Normalize schema to DB layout
    df = df.select(
        [
            pl.col("tournament_id").cast(pl.Int64, strict=False),
            pl.col("match_id").cast(pl.Int64, strict=False),
            pl.col("user_id").cast(pl.Int64, strict=False).alias("player_id"),
        ]
    ).unique(subset=["tournament_id", "match_id", "player_id"])
    rows = [r for r in df.iter_rows(named=True)]
    if not rows:
        return 0
    table = RM.PlayerAppearance.__table__
    stmt = pg_insert(table).values(rows).on_conflict_do_nothing()
    with engine.begin() as conn:
        res = conn.execute(stmt)
    # SQLAlchemy may not expose affected rows for DO NOTHING; return attempted
    return len(rows)


def _process_one(
    engine, tournament_id: int, *, filter_unknown_players: bool = True
) -> tuple[int, int, int]:
    """Return (tid, attempted_rows, kept_rows)."""
    payload = fetch_tournament_players(int(tournament_id))
    rows = extract(int(tournament_id), payload)
    if not rows:
        return (tournament_id, 0, 0)
    df = pl.DataFrame(rows)
    if filter_unknown_players:
        unique_ids = df.select("user_id").unique().to_series().to_list()
        known = _existing_players_lookup(
            engine, [int(x) for x in unique_ids if x is not None]
        )
        if known:
            df = df.filter(pl.col("user_id").is_in(list(known)))
        else:
            df = pl.DataFrame([])
    kept = 0 if df.is_empty() else int(df.height)
    _insert_appearances(engine, df)
    return (tournament_id, len(rows), kept)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Backfill player_appearances from public players route"
    )
    ap.add_argument(
        "--db-url",
        type=str,
        default=os.getenv("RANKINGS_DATABASE_URL") or os.getenv("DATABASE_URL"),
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Process all tournaments present in matches table",
    )
    ap.add_argument(
        "--since-days",
        type=int,
        default=None,
        help="Limit to tournaments with matches in the last N days",
    )
    ap.add_argument(
        "--only-ranked",
        action="store_true",
        help="Limit to tournaments marked is_ranked",
    )
    ap.add_argument(
        "--tournament-ids",
        nargs="*",
        type=int,
        default=None,
        help="Explicit list of tournament IDs to process",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel workers for fetching and inserting",
    )
    ap.add_argument(
        "--no-filter-unknown",
        action="store_true",
        help="Do not filter out unknown players (may violate FKs)",
    )
    args = ap.parse_args(argv)

    engine = rankings_create_engine(args.db_url)
    rankings_create_all(engine)

    if args.tournament_ids:
        tids = list(dict.fromkeys(int(t) for t in args.tournament_ids))
    else:
        tids = _iter_tournament_ids(
            engine,
            all_time=bool(args.all),
            since_days=args.since_days,
            only_ranked=bool(args.only_ranked),
        )

    if not tids:
        print("No tournaments selected.")
        return 0

    print(
        f"Backfilling appearances for {len(tids)} tournaments (workers={args.workers})"
    )
    total_attempted = 0
    total_kept = 0

    def _wrap(tid: int):
        try:
            return _process_one(
                engine, tid, filter_unknown_players=not args.no_filter_unknown
            )
        except Exception as e:
            print(f"TID {tid} failed: {e}")
            return (tid, 0, 0)

    if args.workers and args.workers > 1:
        with cf.ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
            for tid, attempted, kept in ex.map(_wrap, tids):
                total_attempted += attempted
                total_kept += kept
    else:
        for tid in tids:
            tid, attempted, kept = _wrap(tid)
            total_attempted += attempted
            total_kept += kept

    print(
        f"Finished. Extracted={total_attempted} rows; inserted/kept={total_kept} (unknowns filtered={not args.no_filter_unknown})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
