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
from typing import Any, Dict, Iterable, Optional, Tuple

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
from rankings.utils.spaces_upload import upload_outputs

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # fallback if not installed; will error on load


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _db_existing_tournament_ids(
    db_url: str | None, sslmode: Optional[str]
) -> set[int]:
    # Respect sslmode for component env builds if no URL provided
    if not db_url and sslmode:
        os.environ["RANKINGS_DB_SSLMODE"] = sslmode
    # Ensure sslmode env is respected when using component envs
    if not db_url and sslmode:
        os.environ["RANKINGS_DB_SSLMODE"] = str(sslmode)
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
    # Optional path to config file; default resolved later
    parser.add_argument(
        "--config",
        type=str,
        default=os.getenv("RANKINGS_CONFIG_PATH"),
        help="Path to YAML config with defaults (overrides package default)",
    )
    parser.add_argument(
        "--weeks-back",
        type=int,
        default=None,
        help="ISO weeks to scan back from today",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Base directory for JSON payloads",
    )
    parser.add_argument(
        "--compiled-out",
        type=str,
        default=None,
        help="Directory for compiled outputs",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
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
        default=None,
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

    parser.add_argument(
        "--upload-s3",
        action="store_true",
        help="Upload outputs (engine_state, manifest) to S3 using env RANKINGS_S3_URI or RANKINGS_S3_BUCKET/PREFIX",
    )

    args = parser.parse_args(argv)

    # Load config (YAML) and merge defaults
    def _load_config(path: Optional[str]) -> Dict[str, Any]:
        if path:
            cfg_path = Path(path)
        else:
            # Default to package config: src/rankings/config.yaml
            cfg_path = Path(__file__).resolve().parent / "config.yaml"
            if not cfg_path.exists():
                alt = Path.cwd() / "src" / "rankings" / "config.yaml"
                cfg_path = alt if alt.exists() else cfg_path
        if not cfg_path.exists():
            return {}
        if yaml is None:
            raise RuntimeError(
                f"pyyaml is required to load config from {cfg_path} but is not installed"
            )
        with cfg_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Invalid config format at {cfg_path}")
        return data

    cfg = _load_config(args.config)

    def _get(key: str, cli_value: Any, default: Any) -> Any:
        return (
            cli_value
            if (cli_value is not None and cli_value != False)
            else cfg.get(key, default)
        )

    # Resolve effective settings from CLI > config > fallback
    weeks_back = _get("weeks_back", args.weeks_back, 2)
    data_dir = _get("data_dir", args.data_dir, None)
    compiled_out = _get("compiled_out", args.compiled_out, "data/compiled")
    only_ranked = bool(_get("only_ranked", args.only_ranked, True))
    save_to_db = bool(_get("save_to_db", args.save_to_db, True))
    since_days = _get("since_days", args.since_days, 120)
    until_days = _get("until_days", args.until_days, None)
    sslmode = _get("sslmode", args.sslmode, None)
    write_parquet = bool(_get("write_parquet", args.write_parquet, False))
    upload_s3 = bool(_get("upload_s3", args.upload_s3, False))
    s3_prefix_cfg = cfg.get("s3_prefix")
    # Build/version can be sourced from config as a fallback before env/git
    build_version_cfg = cfg.get("build_version")

    # Build DB URL with sslmode override if provided
    env_db_url = os.getenv("RANKINGS_DATABASE_URL") or os.getenv("DATABASE_URL")
    db_url = args.db_url or cfg.get("db_url") or env_db_url
    if db_url and sslmode:
        from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

        parts = urlparse(db_url)
        q = dict(parse_qsl(parts.query, keep_blank_values=True))
        q["sslmode"] = sslmode
        parts = parts._replace(query=urlencode(q))
        db_url = urlunparse(parts)

    existing = _db_existing_tournament_ids(db_url, sslmode)
    finalized = _discover_recent_finalized(int(weeks_back))
    to_fetch = [tid for tid in finalized if tid not in existing]

    print(
        f"Discovered finalized tournaments: {len(finalized)}; missing in DB: {len(to_fetch)}"
    )
    run_ts = _timestamp()

    # Default scratch path under /tmp if not specified
    data_dir = (
        data_dir
        or os.getenv("RANKINGS_TEMP_DATA_DIR")
        or "/tmp/sendouq/tournaments"
    )
    scraped_dir = Path(str(data_dir)) / f"update_{run_ts}"
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
            (datetime.utcnow() - timedelta(days=int(since_days))).timestamp()
            * 1000
        )
        if since_days is not None
        else None
    )
    until_ms = (
        int(
            (datetime.utcnow() - timedelta(days=int(until_days))).timestamp()
            * 1000
        )
        if until_days is not None
        else None
    )

    tables = load_core_tables(
        engine,
        since_ms=since_ms,
        until_ms=until_ms,
        only_ranked=only_ranked,
    )
    matches = tables.get("matches")
    players = tables.get("players")
    if matches is None:
        matches = pl.DataFrame([])
    if players is None:
        players = pl.DataFrame([])

    out_base = Path(str(compiled_out))
    out_run = out_base / run_ts
    # Defer creating run dir until we actually write something

    # Optionally write compiled snapshots
    if write_parquet:
        out_run.mkdir(parents=True, exist_ok=True)
        matches.write_parquet(str(out_run / "matches.parquet"))
        players.write_parquet(str(out_run / "players.parquet"))

    ranks_rows = 0
    if not matches.is_empty() and not players.is_empty():
        eng = ExposureLogOddsEngine(ExposureLogOddsConfig())
        ranks = eng.rank_players(matches, players)
        ranks_rows = int(ranks.height)
        if write_parquet:
            out_run.mkdir(parents=True, exist_ok=True)
            ranks.write_parquet(str(out_run / "rankings.parquet"))
        # Always save compact state for influence
        out_run.mkdir(parents=True, exist_ok=True)
        save_engine_state(eng, str(out_run / "engine_state.json"))
        if save_to_db:
            build_version = (
                args.build_version or build_version_cfg or _get_build_version()
            )
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
            "weeks_back": int(weeks_back),
            "scraped": scraped_count,
            "imported": imported,
            "compiled_matches": int(matches.height),
            "compiled_players": int(players.height),
            "rankings_rows": ranks_rows,
            "only_ranked": bool(only_ranked),
            "since_days": since_days,
            "until_days": until_days,
        },
    )

    if upload_s3:
        upload_outputs(out_run, s3_prefix=s3_prefix_cfg)

    print(f"Run complete. Outputs: {out_run}")
    return 0


def _upload_outputs_to_s3(out_run: Path) -> None:
    # Backwards-compatible shim; delegate to utility module
    try:
        upload_outputs(out_run)
    except Exception as e:
        print(f"Upload failed: {e}")


if __name__ == "__main__":
    raise SystemExit(main())
