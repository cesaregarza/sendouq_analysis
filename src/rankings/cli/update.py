from __future__ import annotations

"""
Update-and-rank pipeline:
  - Discover recently completed tournaments
  - Compare with DB to find missing tournaments
  - Scrape missing tournaments to a fresh folder and import into DB
  - Load core tables, run the ranking engine, and save outputs

Usage:
  poetry run rankings_update \
    --weeks-back 2 \
    --data-dir data/tournaments \
    --compiled-out data/compiled \
    --only-ranked \
    --since-days 120 \
    --sslmode disable
"""

import argparse
import os
import subprocess
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Tuple

import polars as pl
from sqlalchemy.dialects.postgresql import insert as pg_insert

from rankings.algorithms import ExposureLogOddsEngine
from rankings.analysis.engine_state import save_engine_state
from rankings.cli import db_import as import_cli
from rankings.core import ExposureLogOddsConfig
from rankings.scraping.batch import scrape_tournament_batch
from rankings.scraping.calendar_api import (
    fetch_calendar_week,
    fetch_tournament_metadata,
    is_tournament_finalized,
    iter_weeks_back,
)
from rankings.sql import create_all as rankings_create_all
from rankings.sql import create_engine as rankings_create_engine
from rankings.sql import models as RM
from rankings.sql.constants import SCHEMA as RANKINGS_SCHEMA
from rankings.sql.load import load_core_tables


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _db_existing_tournament_ids(db_url: str | None) -> set[int]:
    engine = rankings_create_engine(db_url)
    rankings_create_all(engine)
    ids: set[int] = set()
    from sqlalchemy import text

    with engine.connect() as conn:
        rows = conn.execute(
            text(f"SELECT tournament_id FROM {RANKINGS_SCHEMA}.tournaments")
        ).fetchall()
        for (tid,) in rows:
            try:
                ids.add(int(tid))
            except Exception:
                continue
    return ids


def _discover_recent_finalized(weeks_back: int) -> list[int]:
    weeks: Iterable[Tuple[int, int]] = iter_weeks_back(
        date.today(), max_weeks=max(weeks_back, 1)
    )
    # Unique IDs across weeks
    seen: set[int] = set()
    ids: list[int] = []
    for y, w in weeks:
        events = fetch_calendar_week(y, w)
        for e in events:
            tid = e.get("tournamentId")
            if isinstance(tid, int) and tid not in seen:
                seen.add(tid)
                ids.append(tid)

    # Filter finalized via metadata
    finalized: list[int] = []
    for tid in ids:
        try:
            meta = fetch_tournament_metadata(tid)
            if is_tournament_finalized(meta):
                finalized.append(tid)
            # politeness delay
            time.sleep(0.15)
        except Exception:
            # ignore failures
            pass
    return finalized


def _import_new_payloads(db_url: str | None, json_dir: Path) -> int:
    engine = rankings_create_engine(db_url)
    rankings_create_all(engine)
    files = import_cli._find_json_files(json_dir)
    total = 0
    for path in files:
        payload = import_cli._load_json_payload(path)
        if not payload:
            continue
        total += import_cli.import_file(engine, payload)
    return total


def _write_manifest(run_dir: Path, data: dict) -> None:
    import json

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(data, indent=2))


def _git_commit_short() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


def _get_build_version() -> str:
    return (
        os.getenv("RANKINGS_BUILD")
        or os.getenv("GIT_COMMIT")
        or _git_commit_short()
        or "unknown"
    )


def _persist_rankings(
    engine, ranks: pl.DataFrame, build_version: str, calculated_at_ms: int
) -> int:
    if ranks is None or ranks.is_empty():
        return 0
    df = ranks.rename({"id": "player_id"}).with_columns(
        [
            pl.lit(int(calculated_at_ms)).alias("calculated_at_ms"),
            pl.lit(build_version).alias("build_version"),
        ]
    )
    rows = [r for r in df.iter_rows(named=True)]
    table = RM.PlayerRanking.__table__
    stmt = pg_insert(table).values(rows)
    stmt = stmt.on_conflict_do_nothing(
        index_elements=[
            table.c.player_id,
            table.c.calculated_at_ms,
            table.c.build_version,
        ]
    )
    with engine.begin() as conn:
        conn.execute(stmt)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Discover, scrape, import, and rank recent tournaments"
    )
    parser.add_argument(
        "--weeks-back",
        type=int,
        default=2,
        help="ISO weeks to scan back from today",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/tournaments",
        help="Base directory for JSON payloads",
    )
    parser.add_argument(
        "--compiled-out",
        type=str,
        default="data/compiled",
        help="Directory for compiled outputs",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=os.getenv("RANKINGS_DATABASE_URL") or os.getenv("DATABASE_URL"),
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
        help="Set libpq sslmode in DB URL",
    )
    parser.add_argument(
        "--since-days",
        type=int,
        default=120,
        help="Only include matches within past N days for ranking",
    )
    parser.add_argument(
        "--until-days",
        type=int,
        default=None,
        help="Upper window bound N days back from now",
    )
    parser.add_argument(
        "--only-ranked",
        action="store_true",
        help="Filter to tournaments marked is_ranked for ranking",
    )
    parser.add_argument(
        "--save-to-db",
        action="store_true",
        help="Persist rankings to DB (player_rankings)",
    )
    parser.add_argument(
        "--build-version",
        type=str,
        default=None,
        help="Explicit build version tag to store with rankings (overrides env/git)",
    )
    parser.add_argument(
        "--write-parquet",
        action="store_true",
        help="Write Parquet snapshots (matches/players and rankings)",
    )

    args = parser.parse_args(argv)

    # Build DB URL with sslmode override if provided
    db_url = args.db_url
    if db_url and args.sslmode:
        from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

        parts = urlparse(db_url)
        q = dict(parse_qsl(parts.query, keep_blank_values=True))
        q["sslmode"] = args.sslmode
        parts = parts._replace(query=urlencode(q))
        db_url = urlunparse(parts)

    existing = _db_existing_tournament_ids(db_url)
    finalized = _discover_recent_finalized(args.weeks_back)
    to_fetch = [tid for tid in finalized if tid not in existing]

    print(
        f"Discovered finalized tournaments: {len(finalized)}; missing in DB: {len(to_fetch)}"
    )
    run_ts = _timestamp()

    scraped_dir = Path(args.data_dir) / f"update_{run_ts}"
    scraped_dir.mkdir(parents=True, exist_ok=True)
    scraped_count = 0
    if to_fetch:
        res = scrape_tournament_batch(
            tournament_ids=to_fetch, output_dir=str(scraped_dir), batch_size=50
        )
        print(
            f"Scrape result: scraped={res.get('scraped')} failed={res.get('failed')} head_failed={res.get('failed_ids')[:10] if res.get('failed_ids') else []}"
        )
        scraped_count = int(res.get("scraped") or 0)
    else:
        print("No new finalized tournaments to scrape.")

    # Import newly scraped JSONs
    imported = 0
    if scraped_count > 0:
        imported = _import_new_payloads(db_url, scraped_dir)
        print(f"Imported {imported} tournaments into DB from {scraped_dir}")

    # Compile and rank
    engine = rankings_create_engine(db_url)
    rankings_create_all(engine)

    since_ms = (
        int(
            (
                datetime.utcnow() - timedelta(days=int(args.since_days))
            ).timestamp()
            * 1000
        )
        if args.since_days is not None
        else None
    )
    until_ms = (
        int(
            (
                datetime.utcnow() - timedelta(days=int(args.until_days))
            ).timestamp()
            * 1000
        )
        if args.until_days is not None
        else None
    )

    tables = load_core_tables(
        engine,
        since_ms=since_ms,
        until_ms=until_ms,
        only_ranked=args.only_ranked,
    )
    matches = tables.get("matches")
    players = tables.get("players")
    if matches is None:
        matches = pl.DataFrame([])
    if players is None:
        players = pl.DataFrame([])

    out_base = Path(args.compiled_out)
    out_run = out_base / run_ts
    # Defer creating run dir until we actually write something

    # Optionally write compiled snapshots
    if args.write_parquet:
        out_run.mkdir(parents=True, exist_ok=True)
        matches.write_parquet(str(out_run / "matches.parquet"))
        players.write_parquet(str(out_run / "players.parquet"))

    ranks_rows = 0
    if not matches.is_empty() and not players.is_empty():
        eng = ExposureLogOddsEngine(ExposureLogOddsConfig())
        ranks = eng.rank_players(matches, players)
        ranks_rows = int(ranks.height)
        if args.write_parquet:
            out_run.mkdir(parents=True, exist_ok=True)
            ranks.write_parquet(str(out_run / "rankings.parquet"))
        # Always save compact state for influence
        out_run.mkdir(parents=True, exist_ok=True)
        save_engine_state(eng, str(out_run / "engine_state.json"))
        if args.save_to_db:
            build_version = args.build_version or _get_build_version()
            calculated_at_ms = int(datetime.utcnow().timestamp() * 1000)
            inserted = _persist_rankings(
                engine, ranks, build_version, calculated_at_ms
            )
            print(f"Saved {inserted} rankings to DB (build={build_version})")
        print(f"Engine run complete: {ranks_rows} rankings written")
    else:
        print("Skipping engine run (empty matches or players).")

    _write_manifest(
        out_run,
        {
            "run_ts": run_ts,
            "weeks_back": args.weeks_back,
            "scraped": scraped_count,
            "imported": imported,
            "compiled_matches": int(matches.height),
            "compiled_players": int(players.height),
            "rankings_rows": ranks_rows,
            "only_ranked": bool(args.only_ranked),
            "since_days": args.since_days,
            "until_days": args.until_days,
        },
    )

    print(f"Run complete. Outputs: {out_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
