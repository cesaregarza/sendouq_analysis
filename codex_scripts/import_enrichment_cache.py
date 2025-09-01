#!/usr/bin/env python3
"""
One-off helper: import local enrichment cache files into the DB mapping table.

Why: On ephemeral instances we don't want to rely on local files; this takes
an existing cache folder (Feather/IPC/Parquet) and upserts the mapping into
`player_appearance_teams` so future runs reuse it without any API calls.

Usage:
  python codex_scripts/import_enrichment_cache.py \
    --cache-dir data/enrichment/appearances_enriched \
    --db-url $RANKINGS_DATABASE_URL \
    --chunk-size 20000

Safe to re-run. Upserts on (tournament_id, match_id, player_id).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import polars as pl
from sqlalchemy import text as sqltext
from sqlalchemy.dialects.postgresql import insert as pg_insert

from rankings.sql import create_all as rankings_create_all
from rankings.sql import create_engine as rankings_create_engine
from rankings.sql import models as RM
from rankings.sql.constants import SCHEMA as RANKINGS_SCHEMA


def _load_env(dotenv_path: str | None) -> None:
    """Lightweight .env loader (no extra dependency).

    Sets os.environ keys only if not already present.
    """
    if not dotenv_path:
        return
    p = Path(dotenv_path)
    if not p.exists():
        return
    try:
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)
    except Exception:
        pass


def _iter_cache_files(cache_dir: Path):
    pats = ("*.feather", "*.ipc", "*.parquet")
    for pat in pats:
        for p in cache_dir.glob(pat):
            if p.is_file():
                yield p
    for pat in pats:
        for p in cache_dir.rglob(pat):
            if p.is_file():
                yield p


def _read_one(path: Path) -> pl.DataFrame | None:
    try:
        if path.suffix.lower() in {".feather", ".ipc"}:
            df = pl.read_ipc(str(path))
        else:
            df = pl.read_parquet(str(path))
    except Exception:
        return None
    need = {"tournament_id", "match_id", "user_id", "team_id"}
    if not need.issubset(set(df.columns)):
        return None
    df = df.select(
        [
            pl.col("tournament_id").cast(pl.Int64, strict=False),
            pl.col("match_id").cast(pl.Int64, strict=False),
            pl.col("user_id").cast(pl.Int64, strict=False),
            pl.col("team_id").cast(pl.Int64, strict=False),
        ]
    ).drop_nulls(["tournament_id", "match_id", "user_id", "team_id"])  # type: ignore[arg-type]
    return df if not df.is_empty() else None


def _upsert(engine, df: pl.DataFrame, chunk_size: int) -> int:
    if df.is_empty():
        return 0
    table = RM.PlayerAppearanceTeam.__table__
    total = 0
    for i in range(0, df.height, chunk_size):
        part = df.slice(i, chunk_size)
        rows = [
            {
                "tournament_id": int(r["tournament_id"]),
                "match_id": int(r["match_id"]),
                "player_id": int(r["user_id"]),
                "team_id": int(r["team_id"]),
            }
            for r in part.iter_rows(named=True)
        ]
        if not rows:
            continue
        stmt = pg_insert(table).values(rows).on_conflict_do_update(
            index_elements=[
                table.c.tournament_id,
                table.c.match_id,
                table.c.player_id,
            ],
            set_={"team_id": pg_insert(table).excluded.team_id},
        )
        with rankings_create_engine(args.db_url).begin() as conn:  # type: ignore[name-defined]
            conn.execute(stmt)
        total += len(rows)
    return total


def _fetch_existing_ids(engine, table: str, col: str, ids: list[int]) -> set[int]:
    if not ids:
        return set()
    out: set[int] = set()
    with engine.connect() as conn:
        for i in range(0, len(ids), 1000):
            chunk = ids[i : i + 1000]
            q = sqltext(
                f"SELECT {col} FROM {RANKINGS_SCHEMA}.{table} WHERE {col} = ANY(:ids)"
            )
            rows = conn.execute(q, {"ids": chunk}).fetchall()
            out.update(int(r[0]) for r in rows)
    return out


def _filter_existing_foreign_keys(engine, df: pl.DataFrame) -> pl.DataFrame:
    """Best-effort filter to avoid FK violations by removing unknown references.

    Keeps only rows where tournament_id, match_id, player_id, and team_id exist
    in tournaments, matches, players, and tournament_teams respectively.
    """
    if df.is_empty():
        return df
    tids = [
        int(x)
        for x in df.select("tournament_id").unique()["tournament_id"].to_list()
        if x is not None
    ]
    mids = [
        int(x)
        for x in df.select("match_id").unique()["match_id"].to_list()
        if x is not None
    ]
    pids = [
        int(x)
        for x in df.select("user_id").unique()["user_id"].to_list()
        if x is not None
    ]
    teams = [
        int(x)
        for x in df.select("team_id").unique()["team_id"].to_list()
        if x is not None
    ]

    existing_tids = _fetch_existing_ids(engine, "tournaments", "tournament_id", tids)
    existing_mids = _fetch_existing_ids(engine, "matches", "match_id", mids)
    existing_pids = _fetch_existing_ids(engine, "players", "player_id", pids)
    existing_teams = _fetch_existing_ids(
        engine, "tournament_teams", "team_id", teams
    )

    before = df.height
    df2 = df.filter(
        pl.col("tournament_id").is_in(list(existing_tids))
        & pl.col("match_id").is_in(list(existing_mids))
        & pl.col("user_id").is_in(list(existing_pids))
        & pl.col("team_id").is_in(list(existing_teams))
    )
    removed = before - df2.height
    if removed > 0:
        print(f"Filtered out {removed} rows due to missing FK references")
    return df2


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Import local enrichment cache into DB.")
    ap.add_argument("--dotenv", type=str, default=".env", help="Path to .env (default: .env)")
    ap.add_argument("--db-url", type=str, default=None, help="DB URL (falls back to env)")
    ap.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory (falls back to env RANKINGS_ENRICH_CACHE_DIR)",
    )
    ap.add_argument("--chunk-size", type=int, default=20000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Load .env before reading env-based defaults
    _load_env(args.dotenv)

    db_url = args.db_url or os.getenv("RANKINGS_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        print("Error: DB URL not provided and not found in env.")
        raise SystemExit(2)

    cache_dir_arg = args.cache_dir or os.getenv("RANKINGS_ENRICH_CACHE_DIR") or "data/enrichment/appearances_enriched"

    cache_dir = Path(cache_dir_arg)
    if not cache_dir.exists():
        print(f"Cache dir not found: {cache_dir}")
        raise SystemExit(1)

    frames: list[pl.DataFrame] = []
    files = list(_iter_cache_files(cache_dir))
    for p in files:
        df = _read_one(p)
        if df is not None:
            frames.append(df)

    if not frames:
        print("No valid cache files found.")
        raise SystemExit(0)

    all_df = pl.concat(frames, how="vertical_relaxed").unique(
        subset=["tournament_id", "match_id", "user_id"]
    )
    print(
        f"Prepared {all_df.height} unique assignments from {len(files)} files in {cache_dir}"
    )
    if args.dry_run:
        print("Dry run: not writing to DB.")
        raise SystemExit(0)

    engine = rankings_create_engine(db_url)
    # Ensure schema/tables exist (idempotent)
    try:
        rankings_create_all(engine)
    except Exception as e:
        print(f"Warning: failed to create schema/tables: {e}")
    # Ensure references exist to avoid FK violations
    all_df = _filter_existing_foreign_keys(engine, all_df)
    inserted = _upsert(engine, all_df, int(args.chunk_size))
    print(f"Upserted {inserted} rows into player_appearance_teams")
