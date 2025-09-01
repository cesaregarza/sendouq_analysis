from __future__ import annotations

"""
Import local enrichment cache files into the DB mapping table.

Scans a directory for Feather/IPC/Parquet assignment files produced by
enrichment (columns: [tournament_id, match_id, user_id, team_id]), normalizes
types, de-duplicates, and upserts into `player_appearance_teams`.

Usage:
  python -m rankings.cli.import_enrichment_cache \
    --cache-dir data/enrichment/appearances_enriched \
    --db-url $RANKINGS_DATABASE_URL \
    --chunk-size 20000

Notes:
  - Idempotent upsert on (tournament_id, match_id, player_id)
  - Safe to re-run; later files override earlier team_id on conflict
"""

import argparse
import os
from pathlib import Path
from typing import Iterable

import polars as pl
from sqlalchemy.dialects.postgresql import insert as pg_insert

from rankings.sql import create_engine as rankings_create_engine
from rankings.sql import models as RM


def _iter_cache_files(cache_dir: Path) -> Iterable[Path]:
    # Accept both flat and nested layouts
    patterns = ["*.feather", "*.ipc", "*.parquet"]
    for pat in patterns:
        for p in cache_dir.glob(pat):
            if p.is_file():
                yield p
    for pat in patterns:
        for p in cache_dir.rglob(pat):
            if p.is_file():
                yield p


def _read_one(path: Path) -> pl.DataFrame | None:
    try:
        suf = path.suffix.lower()
        if suf in {".feather", ".ipc"}:
            df = pl.read_ipc(str(path))
        else:
            df = pl.read_parquet(str(path))
        # Normalize and filter
        cols = set(df.columns)
        needed = {"tournament_id", "match_id", "user_id", "team_id"}
        if not needed.issubset(cols):
            return None
        df = df.select(
            [
                pl.col("tournament_id").cast(pl.Int64, strict=False),
                pl.col("match_id").cast(pl.Int64, strict=False),
                pl.col("user_id").cast(pl.Int64, strict=False),
                pl.col("team_id").cast(pl.Int64, strict=False),
            ]
        )
        # Drop rows missing any of the keys or team
        df = df.drop_nulls(["tournament_id", "match_id", "user_id", "team_id"])  # type: ignore[arg-type]
        if df.is_empty():
            return None
        return df
    except Exception:
        return None


def _upsert(engine, df: pl.DataFrame, chunk_size: int) -> int:
    if df.is_empty():
        return 0
    total = 0
    table = RM.PlayerAppearanceTeam.__table__
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
        stmt = pg_insert(table).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                table.c.tournament_id,
                table.c.match_id,
                table.c.player_id,
            ],
            set_={"team_id": stmt.excluded.team_id},
        )
        with engine.begin() as conn:
            conn.execute(stmt)
        total += len(rows)
    return total


def main(argv: list[str] | None = None) -> int:
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
    args = ap.parse_args(argv)

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        print(f"Cache dir not found: {cache_dir}")
        return 1

    dfs: list[pl.DataFrame] = []
    seen = set()
    files = list(_iter_cache_files(cache_dir))
    if not files:
        print("No cache files found.")
        return 0
    for p in files:
        df = _read_one(p)
        if df is None or df.is_empty():
            continue
        # Avoid re-processing exact duplicates by content hash of first 100 rows keys
        # (best-effort; dedup properly below)
        dfs.append(df)

    if not dfs:
        print("No valid cache data to import.")
        return 0

    all_df = pl.concat(dfs, how="vertical_relaxed").unique(
        subset=["tournament_id", "match_id", "user_id"]
    )
    print(
        f"Prepared {all_df.height} unique assignments from {len(files)} files in {cache_dir}"
    )
    if args.dry_run:
        print("Dry run: not writing to DB.")
        return 0

    engine = rankings_create_engine(args.db_url)
    inserted = _upsert(engine, all_df, int(args.chunk_size))
    print(f"Upserted {inserted} rows into player_appearance_teams")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

