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
from sqlalchemy.dialects.postgresql import insert as pg_insert

from rankings.sql import create_engine as rankings_create_engine
from rankings.sql import models as RM


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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Import local enrichment cache into DB.")
    ap.add_argument(
        "--db-url",
        type=str,
        default=os.getenv("RANKINGS_DATABASE_URL") or os.getenv("DATABASE_URL"),
    )
    ap.add_argument(
        "--cache-dir",
        type=str,
        default=os.getenv("RANKINGS_ENRICH_CACHE_DIR", "data/enrichment/appearances_enriched"),
    )
    ap.add_argument("--chunk-size", type=int, default=20000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
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

    engine = rankings_create_engine(args.db_url)
    inserted = _upsert(engine, all_df, int(args.chunk_size))
    print(f"Upserted {inserted} rows into player_appearance_teams")
