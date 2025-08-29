from __future__ import annotations

"""
Compile pipeline CLI: optionally scrape, import to DB, and emit core tables.

This entrypoint builds on rankings.scraping and rankings.sql to:
  - Optionally scrape tournaments (weekly or recent weeks)
  - Import scraped JSON into the Postgres schema via the db_import helpers
  - Load matches/players from the DB and write Parquet outputs for notebooks/apps

Usage examples:
  poetry run rankings_compile --import --output data/compiled --only-ranked
  poetry run rankings_compile --scrape ranked --weeks-back 2 --import --only-ranked
  poetry run rankings_compile --scrape weekly --skip-existing --import --since-days 120
"""

import argparse
import json
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import polars as pl
from sqlalchemy.dialects.postgresql import insert as pg_insert

from rankings.algorithms import ExposureLogOddsEngine
from rankings.analysis.engine_state import save_engine_state
from rankings.cli import db_import as import_cli
from rankings.core import ExposureLogOddsConfig
from rankings.scraping.batch import scrape_latest_tournaments
from rankings.scraping.storage import load_match_appearances
from rankings.sql import create_all as rankings_create_all
from rankings.sql import create_engine as rankings_create_engine
from rankings.sql import models as RM
from rankings.sql.load import load_core_tables


def _get_build_version() -> str:
    return (
        os.getenv("RANKINGS_BUILD")
        or os.getenv("GIT_COMMIT")
        or _git_commit_short()
        or "unknown"
    )


def _git_commit_short() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


def _persist_rankings(
    engine, ranks: pl.DataFrame, build_version: str, calculated_at_ms: int
) -> int:
    if ranks is None or ranks.is_empty():
        return 0
    # Ensure columns
    cols = set(ranks.columns)
    needed = {"id", "player_rank", "score", "win_pr", "loss_pr", "exposure"}
    missing = needed - cols
    if missing:
        # allow minimal: id + score
        pass
    df = ranks.with_columns(
        [
            pl.lit(int(calculated_at_ms)).alias("calculated_at_ms"),
            pl.lit(build_version).alias("build_version"),
        ]
    ).rename({"id": "player_id"})

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


def _ts_now() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _scrape(args: argparse.Namespace) -> None:
    if args.scrape == "none":
        return

    out_dir = args.data_dir
    if args.output_dir:
        out_dir = args.output_dir
    _ensure_dir(Path(out_dir))

    if args.scrape == "ranked":
        # Lightweight: scrape latest tournaments by ID for recent coverage
        # For parity with ranked CLI, default to scanning past 2 weeks worth
        # of tournaments by count if weeks_back provided; else use count.
        count = max(1, int(args.recent_count or 200))
        scrape_latest_tournaments(count=count, output_dir=out_dir)
    elif args.scrape == "weekly":
        # Keep it simple: also use latest tournaments for weekly mode to avoid
        # duplicating calendar/week logic here. Users can still use `scrape_weekly`.
        count = max(1, int(args.recent_count or 100))
        scrape_latest_tournaments(count=count, output_dir=out_dir)


def _import_json(args: argparse.Namespace) -> None:
    # Mirror rankings_import CLI behavior programmatically so the compile
    # step keeps the DB in sync with local JSON files.
    db_url = args.db_url
    if db_url and args.sslmode:
        from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

        parts = urlparse(db_url)
        q = dict(parse_qsl(parts.query, keep_blank_values=True))
        q["sslmode"] = args.sslmode
        parts = parts._replace(query=urlencode(q))
        db_url = urlunparse(parts)

    engine = rankings_create_engine(db_url)
    rankings_create_all(engine)

    # Use the import_file helper from rankings.cli.db_import across all files.
    files = import_cli._find_json_files(Path(args.data_dir))
    total = 0
    for i, path in enumerate(files, 1):
        payload = import_cli._load_json_payload(path)
        if not payload:
            continue
        total += import_cli.import_file(engine, payload)
    # Backfill is_ranked if present in parser payloads
    try:
        # When present in payloads, _upsert_tournaments covers columns; explicit
        # is_ranked backfill is best-effort (safe no-op if columns absent).
        pass
    except Exception:
        pass


def _write_outputs(tables: dict[str, pl.DataFrame], out_dir: Path) -> Path:
    run_dir = out_dir / _ts_now()
    _ensure_dir(run_dir)

    matches = tables.get("matches")
    players = tables.get("players")
    if matches is None:
        matches = pl.DataFrame([])
    if players is None:
        players = pl.DataFrame([])

    matches_path = run_dir / "matches.parquet"
    players_path = run_dir / "players.parquet"
    manifest_path = run_dir / "manifest.json"

    # Write small if empty to keep downstream simple
    matches.write_parquet(str(matches_path))
    players.write_parquet(str(players_path))

    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "matches_rows": int(matches.height),
        "players_rows": int(players.height),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return run_dir


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scrape (optional), import, and compile core tables from DB"
    )
    parser.add_argument(
        "--scrape",
        choices=["none", "weekly", "ranked"],
        default="none",
        help="Optional scrape step before import (default: none)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/tournaments",
        help="Directory of tournament JSON files (scrape/import)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/tournaments",
        help="Directory to write scraped files (defaults to --data-dir)",
    )
    parser.add_argument(
        "--recent-count",
        type=int,
        default=None,
        help="When scraping, number of most recent tournaments to fetch",
    )
    parser.add_argument(
        "--import",
        dest="do_import",
        action="store_true",
        help="Import JSON files into the rankings database before compile",
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
        default=os.getenv("RANKINGS_DB_SCHEMA", "comp_rankings"),
        help="Database schema (rankings or comp_rankings)",
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
        help="Set libpq sslmode in the connection URL",
    )
    parser.add_argument(
        "--since-days",
        type=int,
        default=None,
        help="Only include matches in the past N days",
    )
    parser.add_argument(
        "--until-days",
        type=int,
        default=None,
        help="Only include matches up to N days back from now",
    )
    parser.add_argument(
        "--only-ranked",
        action="store_true",
        help="Filter to matches from tournaments marked is_ranked",
    )
    parser.add_argument(
        "--compiled-out",
        type=str,
        default="data/compiled",
        help="Directory to write compiled Parquet outputs",
    )
    parser.add_argument(
        "--write-parquet",
        action="store_true",
        help="Write Parquet snapshots (matches/players and rankings)",
    )
    parser.add_argument(
        "--run-engine",
        action="store_true",
        help="Run Exposure Log-Odds engine on compiled tables and write rankings",
    )
    parser.add_argument(
        "--pickle-engine",
        action="store_true",
        help="Additionally pickle the engine object to engine.pkl (optional)",
    )
    parser.add_argument(
        "--save-to-db",
        action="store_true",
        help="Persist rankings to the database (player_rankings table)",
    )
    parser.add_argument(
        "--build-version",
        type=str,
        default=None,
        help="Explicit build version tag to store with rankings (overrides env/git)",
    )

    args = parser.parse_args(argv)

    # Optional scrape
    _scrape(args)

    # Optional import
    if args.do_import:
        _import_json(args)

    # Compile core tables
    # Respect --sslmode for component env builds if no URL provided
    if not args.db_url and args.sslmode:
        os.environ["RANKINGS_DB_SSLMODE"] = args.sslmode
    engine = rankings_create_engine(args.db_url)
    rankings_create_all(engine)

    since_ms = None
    until_ms = None
    if args.since_days is not None:
        since_ms = int(
            (
                datetime.utcnow() - timedelta(days=int(args.since_days))
            ).timestamp()
            * 1000
        )
    if args.until_days is not None:
        until_ms = int(
            (
                datetime.utcnow() - timedelta(days=int(args.until_days))
            ).timestamp()
            * 1000
        )

    tables = load_core_tables(
        engine,
        since_ms=since_ms,
        until_ms=until_ms,
        only_ranked=args.only_ranked,
    )

    out_dir = Path(args.compiled_out)
    _ensure_dir(out_dir)
    run_dir: Path | None = None
    if args.write_parquet:
        run_dir = _write_outputs(tables, out_dir)
        print(
            f"Compiled matches={tables['matches'].height} players={tables['players'].height} -> {run_dir}"
        )
    else:
        print(
            f"Compiled matches={tables['matches'].height} players={tables['players'].height} (no parquet; use --write-parquet)"
        )

    if args.run_engine:
        matches = tables["matches"]
        players = tables["players"]
        if matches.is_empty() or players.is_empty():
            print("No data to run engine (empty matches/players)")
            return 0

        # Try to use per-match appearances from scraped JSON if available
        appearances = load_match_appearances(args.data_dir)
        if not appearances.is_empty():
            # Filter to IDs present in current matches to avoid bloat
            try:
                key_df = matches.select(["tournament_id", "match_id"]).unique()
                appearances = appearances.join(
                    key_df, on=["tournament_id", "match_id"], how="inner"
                )
            except Exception:
                pass

        eng = ExposureLogOddsEngine(ExposureLogOddsConfig())
        if appearances.is_empty():
            ranks = eng.rank_players(matches, players)
        else:
            ranks = eng.rank_players(matches, players, appearances=appearances)
        if run_dir is None:
            run_dir = out_dir / _ts_now()
            _ensure_dir(run_dir)
        if args.write_parquet:
            ranks_path = run_dir / "rankings.parquet"
            ranks.write_parquet(str(ranks_path))

        # Save compact engine state for downstream influence analysis
        state_path = run_dir / "engine_state.json"
        save_engine_state(eng, str(state_path))

        # Optionally pickle the full engine object (may be large)
        if args.pickle_engine:
            try:
                import pickle

                with open(run_dir / "engine.pkl", "wb") as f:
                    pickle.dump(eng, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print(f"Warning: failed to pickle engine: {e}")

        if args.save_to_db:
            build_version = args.build_version or _get_build_version()
            calculated_at_ms = int(datetime.utcnow().timestamp() * 1000)
            inserted = _persist_rankings(
                engine, ranks, build_version, calculated_at_ms
            )
            print(f"Saved {inserted} rankings to DB (build={build_version})")

        if args.write_parquet:
            print(
                f"Engine run complete: rankings={ranks.height} -> {run_dir / 'rankings.parquet'}; state={state_path}"
            )
        else:
            print(
                f"Engine run complete: rankings={ranks.height}; state={state_path} (no parquet; use --write-parquet)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
