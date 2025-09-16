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
import logging
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
from rankings.core import DecayConfig, ExposureLogOddsConfig
from rankings.core.logging import setup_logging
from rankings.core.sentry import init_sentry
from rankings.scraping.batch import scrape_tournament_batch
from rankings.scraping.calendar_api import (
    fetch_calendar_week,
    fetch_tournament_metadata,
    is_tournament_finalized,
    iter_weeks_back,
)
from rankings.scraping.enrich import (
    apply_enrichment_cache,
    apply_enrichment_db_cache,
    enrich_appearances_team_by_match_api,
)
from rankings.scraping.storage import load_match_appearances
from rankings.sql import create_all as rankings_create_all
from rankings.sql import create_engine as rankings_create_engine
from rankings.sql import models as RM
from rankings.sql.constants import SCHEMA as RANKINGS_SCHEMA
from rankings.sql.load import load_core_tables, load_player_appearances_df
from rankings.sql.views import ensure_tournament_event_times_view
from rankings.utils.spaces_upload import upload_outputs

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # fallback if not installed; will error on load


def _timestamp() -> str:
    """Return a compact UTC timestamp string for output folders."""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _db_existing_tournament_ids(
    db_url: str | None, sslmode: Optional[str]
) -> set[int]:
    """Return set of tournament IDs currently present in the database."""
    # Ensure sslmode env is respected when using component envs (no URL provided)
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
    """Discover finalized tournament IDs from the calendar API.

    Scans ISO weeks back from today, then filters IDs whose metadata indicates
    a finalized tournament.
    """
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
    """Import newly scraped JSON files under `json_dir` into the database."""
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
    """Write a JSON manifest describing this update run to `run_dir`."""
    import json

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(data, indent=2))


def _git_commit_short() -> str | None:
    """Return short git commit hash if available, else None."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


def _get_build_version() -> str:
    """Return build version from env or git, defaulting to 'unknown'."""
    return (
        os.getenv("RANKINGS_BUILD")
        or os.getenv("GIT_COMMIT")
        or _git_commit_short()
        or "unknown"
    )


# Local Sentry init moved to rankings.core.sentry.init_sentry


def _persist_rankings(
    engine, ranks: pl.DataFrame, build_version: str, calculated_at_ms: int
) -> int:
    """Persist ranking outputs into `player_rankings` (idempotent insert)."""
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


def _persist_appearance_team_assignments(
    engine, appearances: pl.DataFrame
) -> int:
    """Upsert enriched appearance team assignments into DB.

    Expects `appearances` to contain columns [tournament_id, match_id, user_id, team_id]
    after enrichment. Performs an upsert into player_appearance_teams on the
    (tournament_id, match_id, player_id) key, setting team_id on conflict.

    Best-effort: filters to rows with non-null team_id and keys present in the
    current dataset. Attempts to filter to known player IDs to avoid FK errors.
    Returns number of rows attempted (after filtering).
    """
    try:
        if appearances is None or appearances.is_empty():
            return 0
        needed = {"tournament_id", "match_id", "user_id", "team_id"}
        if not needed.issubset(set(appearances.columns)):
            return 0
        df = (
            appearances.select(
                [
                    pl.col("tournament_id").cast(pl.Int64, strict=False),
                    pl.col("match_id").cast(pl.Int64, strict=False),
                    pl.col("user_id").cast(pl.Int64, strict=False),
                    pl.col("team_id").cast(pl.Int64, strict=False),
                ]
            )
            .drop_nulls(["tournament_id", "match_id", "user_id", "team_id"])  # type: ignore[arg-type]
            .unique(subset=["tournament_id", "match_id", "user_id"])
        )
        if df.is_empty():
            return 0

        # Filter to known players to avoid foreign key violations (best-effort)
        try:
            unique_pids = df.select("user_id").unique()["user_id"].to_list()
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
                    df = df.filter(pl.col("user_id").is_in(list(known)))
        except Exception:
            # If the filter fails for any reason, proceed without it; DB may enforce FKs
            pass

        if df.is_empty():
            return 0

        rows = [
            {
                "tournament_id": int(r["tournament_id"]),
                "match_id": int(r["match_id"]),
                "player_id": int(r["user_id"]),
                "team_id": int(r["team_id"]),
            }
            for r in df.iter_rows(named=True)
        ]
        if not rows:
            return 0

        table = RM.PlayerAppearanceTeam.__table__
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
        return len(rows)
    except Exception as e:
        logging.getLogger(__name__).warning(
            "Appearance team assignment upsert skipped due to error: %s", e
        )
        return 0


def _compute_player_stats(
    matches: pl.DataFrame,
    players: pl.DataFrame,
    appearances: pl.DataFrame | None,
) -> pl.DataFrame:
    """Compute tournament_count and last_active_ms per player for the current dataset.

    - Uses appearances if provided (who actually played); else falls back to rosters.
    - last_active_ms is derived from matches: max(last_game_finished_at, match_created_at) per tournament.
    """
    if matches is None or matches.is_empty():
        return pl.DataFrame([])

    # Derive per-tournament last activity (seconds) from matches
    ts_sec = (
        pl.coalesce(
            [
                pl.col("last_game_finished_at"),
                pl.col("match_created_at"),
            ]
        )
        .cast(pl.Float64, strict=False)
        .alias("last_ts_sec")
    )
    t_last = (
        matches.select(["tournament_id", ts_sec])
        .drop_nulls("last_ts_sec")
        .group_by("tournament_id")
        .agg(pl.col("last_ts_sec").max().alias("t_last_sec"))
    )

    # Participants frame: appearances preferred; else roster entries
    if appearances is not None and not appearances.is_empty():
        part = appearances.select(["tournament_id", "user_id"]).unique()
    else:
        part = (
            players.select(["tournament_id", "user_id"]).unique()
            if players is not None and not players.is_empty()
            else pl.DataFrame([])
        )
    if part.is_empty():
        return pl.DataFrame([])

    # Align to tournaments present in matches
    valid_tids = matches.select("tournament_id").unique()
    part = part.join(valid_tids, on="tournament_id", how="inner")

    # Join tournament last activity and compute per-player stats
    part_with_ts = part.join(t_last, on="tournament_id", how="left")
    stats = (
        part_with_ts.group_by("user_id")
        .agg(
            [
                pl.col("tournament_id").n_unique().alias("tournament_count"),
                pl.col("t_last_sec").max().alias("last_active_sec"),
            ]
        )
        .with_columns(
            (
                (pl.col("last_active_sec") * 1000.0)
                .cast(pl.Int64, strict=False)
                .alias("last_active_ms")
            )
        )
        .rename({"user_id": "player_id"})
        .select(["player_id", "tournament_count", "last_active_ms"])
    )
    return stats


def _persist_ranking_stats(
    engine, stats: pl.DataFrame, build_version: str, calculated_at_ms: int
) -> int:
    if stats is None or stats.is_empty():
        return 0
    df = stats.with_columns(
        [
            pl.lit(int(calculated_at_ms)).alias("calculated_at_ms"),
            pl.lit(build_version).alias("build_version"),
        ]
    )
    rows = [r for r in df.iter_rows(named=True)]
    table = RM.PlayerRankingStats.__table__
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


def _promote_appearances_to_roster(
    engine,
    matches: pl.DataFrame,
    players: pl.DataFrame,
    appearances: pl.DataFrame,
) -> int:
    """Promote appearance-only players into roster_entries.

    Picks a majority team per (tournament_id, user_id) from enriched appearances
    (requires team_id), excludes those already rostered, and upserts.
    Returns number of attempted inserts.
    """
    if (
        appearances is None
        or appearances.is_empty()
        or "team_id" not in appearances.columns
    ):
        return 0
    # Restrict to current matches
    try:
        key_df = matches.select(["tournament_id", "match_id"]).unique()
        apps = appearances.join(
            key_df, on=["tournament_id", "match_id"], how="inner"
        )
    except Exception:
        apps = appearances
    if apps.is_empty():
        return 0

    apps = apps.drop_nulls(["team_id"]).select(
        [
            pl.col("tournament_id").cast(pl.Int64, strict=False),
            pl.col("user_id").cast(pl.Int64, strict=False),
            pl.col("team_id").cast(pl.Int64, strict=False),
        ]
    )
    if apps.is_empty():
        return 0

    # Exclude existing roster pairs
    roster_pairs = (
        players.select(["tournament_id", "user_id"]).unique()
        if players is not None and not players.is_empty()
        else pl.DataFrame([])
    )
    if not roster_pairs.is_empty():
        apps = apps.join(
            roster_pairs, on=["tournament_id", "user_id"], how="anti"
        )
    if apps.is_empty():
        return 0

    # Majority team per (tid, uid)
    counts = (
        apps.group_by(["tournament_id", "user_id", "team_id"])
        .len()
        .rename({"len": "cnt"})
    )
    top = (
        counts.sort(
            ["tournament_id", "user_id", "cnt"], descending=[False, False, True]
        )
        .group_by(["tournament_id", "user_id"], maintain_order=True)
        .agg([pl.col("team_id").first().alias("team_id")])
        .rename({"user_id": "player_id"})
        .select(["tournament_id", "team_id", "player_id"])
        .drop_nulls(["tournament_id", "team_id", "player_id"])
        .unique()
    )
    if top.is_empty():
        return 0

    rows = [r for r in top.iter_rows(named=True)]
    if not rows:
        return 0
    table = RM.RosterEntry.__table__
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
        help=(
            "Time window in days for DB retrieval (prefer config: retrieval_days)"
        ),
    )
    parser.add_argument(
        "--until-days",
        type=int,
        default=None,
        help="Upper window bound N days back from now",
    )
    parser.add_argument(
        "--no-time-filter",
        action="store_true",
        help="Disable since/until time filters when compiling from DB",
    )
    parser.add_argument(
        "--only-ranked",
        action="store_true",
        help="Filter to tournaments marked is_ranked for ranking",
    )
    parser.add_argument(
        "--include-unranked",
        action="store_true",
        help="Do not filter by is_ranked; include all tournaments",
    )
    parser.add_argument(
        "--save-to-db",
        action="store_true",
        help="Persist rankings to DB (player_rankings)",
    )
    parser.add_argument(
        "--no-save-to-db",
        action="store_true",
        help="Do not persist rankings to DB (overrides config/--save-to-db)",
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
        "--skip-discovery",
        action="store_true",
        help=(
            "Skip calendar discovery and scraping; compile+rank from DB only."
        ),
    )

    parser.add_argument(
        "--upload-s3",
        action="store_true",
        help="Upload outputs (engine_state, manifest) to S3 using env RANKINGS_S3_URI or RANKINGS_S3_BUCKET/PREFIX",
    )

    parser.add_argument(
        "--no-ddl",
        action="store_true",
        help="Do not attempt to create schema/tables (skip DDL)",
    )
    parser.add_argument(
        "--promote-appearances",
        action="store_true",
        help="Upsert appearance-only players into roster_entries using enriched team assignments",
    )

    args = parser.parse_args(argv)

    # Initialize logging and Sentry as early as possible
    try:
        lvl = os.getenv("RANKINGS_LOG_LEVEL", "INFO")
        fmt = os.getenv("RANKINGS_LOG_FORMAT", "detailed")
        setup_logging(level=lvl, format_style=fmt)
    except Exception:
        logging.basicConfig(level=logging.INFO)
    init_sentry(context="rankings_update", release=_get_build_version())

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
    if args.include_unranked:
        only_ranked = False
    save_to_db = bool(_get("save_to_db", args.save_to_db, True))
    if args.no_save_to_db:
        save_to_db = False
    # Retrieval window (days) for DB pulls; prefer new 'retrieval_days' in config,
    # fall back to legacy 'since_days' or CLI flag
    since_days = (
        cfg.get("retrieval_days")
        if cfg.get("retrieval_days") is not None
        else _get("since_days", args.since_days, 540)
    )
    until_days = _get("until_days", args.until_days, None)
    if args.no_time_filter:
        since_days = None
        until_days = None
    sslmode = _get("sslmode", args.sslmode, None)
    write_parquet = bool(_get("write_parquet", args.write_parquet, False))
    upload_s3 = bool(_get("upload_s3", args.upload_s3, False))
    s3_prefix_cfg = cfg.get("s3_prefix")
    # Build/version can be sourced from config as a fallback before env/git
    build_version_cfg = cfg.get("build_version")
    # If Sentry is active, tag build version for correlation (best-effort)
    try:
        import sentry_sdk  # type: ignore

        bv = args.build_version or build_version_cfg or _get_build_version()
        sentry_sdk.set_tag("build_version", bv)
    except Exception:
        pass

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

    log = logging.getLogger("rankings.cli.update")
    to_fetch: list[int] = []
    if args.skip_discovery:
        log.info(
            "Skip-discovery enabled: bypassing calendar lookup and scraping."
        )
    else:
        existing = _db_existing_tournament_ids(db_url, sslmode)
        finalized = _discover_recent_finalized(int(weeks_back))
        to_fetch = [tid for tid in finalized if tid not in existing]
        log.info(
            "Discovered finalized tournaments: %d; missing in DB: %d",
            len(finalized),
            len(to_fetch),
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
        log.info(
            "Scrape result: scraped=%s failed=%s head_failed=%s",
            res.get("scraped"),
            res.get("failed"),
            res.get("failed_ids")[:10] if res.get("failed_ids") else [],
        )
        scraped_count = int(res.get("scraped") or 0)
    else:
        if args.skip_discovery:
            log.info("Skip-discovery: no scraping attempted.")
        else:
            log.info("No new finalized tournaments to scrape.")

    # Import newly scraped JSONs
    imported = 0
    if scraped_count > 0:
        imported = _import_new_payloads(db_url, scraped_dir)
        log.info(
            "Imported %d tournaments into DB from %s", imported, scraped_dir
        )

    # Compile and rank
    engine = rankings_create_engine(db_url)
    if not args.no_ddl:
        try:
            rankings_create_all(engine)
        except Exception as e:
            log.warning("Skipping DDL due to error: %s", e)

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

    # Load with a wide retrieval window to keep dataset manageable
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

    # Do not filter matches by time here.

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
        # Load appearances from DB (preferred), fallback to scraped JSONs when missing
        try:
            appearances = load_player_appearances_df(engine)
        except Exception:
            appearances = pl.DataFrame([])
        if appearances.is_empty():
            try:
                appearances = load_match_appearances(str(scraped_dir))
                if appearances.is_empty() and data_dir:
                    appearances = load_match_appearances(str(data_dir))
            except Exception:
                appearances = pl.DataFrame([])
        if not appearances.is_empty():
            try:
                key_df = matches.select(["tournament_id", "match_id"]).unique()
                appearances = appearances.join(
                    key_df, on=["tournament_id", "match_id"], how="inner"
                )
            except Exception:
                pass
            # Apply DB enrichment cache first (if present)
            try:
                appearances = apply_enrichment_db_cache(appearances, engine)
            except Exception:
                pass
            # Apply local enrichment cache next (if present), then resolve remaining via API
            try:
                cache_dir = os.getenv(
                    "RANKINGS_ENRICH_CACHE_DIR",
                    "data/enrichment/appearances_enriched",
                )
                appearances = apply_enrichment_cache(appearances, cache_dir)
            except Exception:
                pass
            # Enrich team_id for unmatched appearance rows via match API (limited calls)
            try:
                # Only trigger when necessary to minimize API calls
                needs_enrichment = False
                if "team_id" not in appearances.columns:
                    needs_enrichment = True
                else:
                    needs_enrichment = appearances.select(
                        pl.col("team_id").is_null().any()
                    ).item()  # type: ignore[arg-type]
                if needs_enrichment:
                    # Limit calls to at most 500 matches per run by default (env override)
                    max_calls_env = os.getenv("RANKINGS_ENRICH_APPS_MAX_CALLS")
                    max_calls = int(max_calls_env) if max_calls_env else 500
                    appearances = enrich_appearances_team_by_match_api(
                        appearances, matches, players, max_calls=max_calls
                    )
            except Exception as e:
                log.warning("Appearance enrichment skipped due to error: %s", e)

            # Optionally promote appearance-only players into roster_entries, then refresh players
            if args.promote_appearances:
                try:
                    inserted = _promote_appearances_to_roster(
                        engine, matches, players, appearances
                    )
                    if inserted:
                        log.info(
                            "Promoted %d appearance-based players into roster_entries",
                            inserted,
                        )
                        # Reload rosters to reflect changes
                        from rankings.sql.load import load_players_df

                        players = load_players_df(engine)
                    else:
                        log.info(
                            "No appearance-based roster promotions needed."
                        )
                except Exception as e:
                    log.warning("Roster promotion skipped due to error: %s", e)

            # Persist enriched team assignments to DB to warm the cache for future runs
            try:
                if save_to_db:
                    up_cnt = _persist_appearance_team_assignments(
                        engine, appearances
                    )
                    if up_cnt:
                        log.info(
                            "Upserted %d appearance team assignments to DB",
                            up_cnt,
                        )
            except Exception as e:
                log.warning("Appearance team assignment persist failed: %s", e)
        # Build engine config; allow overrides via YAML keys
        eng_cfg = ExposureLogOddsConfig()

        # Optional convenience: top-level 'engine_half_life_days'
        ehd = cfg.get("engine_half_life_days")
        if ehd is not None:
            try:
                eng_cfg.decay = DecayConfig(half_life_days=float(ehd))
            except Exception:
                pass

        # Structured overrides if provided
        dec = cfg.get("decay")
        if isinstance(dec, dict):
            hl = dec.get("half_life_days")
            if hl is not None:
                try:
                    eng_cfg.decay.half_life_days = float(hl)
                except Exception:
                    pass

        eng_d = cfg.get("engine")
        if isinstance(eng_d, dict):
            for k, v in eng_d.items():
                if hasattr(eng_cfg.engine, k):
                    try:
                        setattr(eng_cfg.engine, k, v)
                    except Exception:
                        pass

        pr_d = cfg.get("pagerank")
        if isinstance(pr_d, dict):
            for k, v in pr_d.items():
                if hasattr(eng_cfg.pagerank, k):
                    try:
                        setattr(eng_cfg.pagerank, k, v)
                    except Exception:
                        pass

        tt_d = cfg.get("tick_tock")
        if isinstance(tt_d, dict):
            for k, v in tt_d.items():
                if hasattr(eng_cfg.tick_tock, k):
                    try:
                        setattr(eng_cfg.tick_tock, k, v)
                    except Exception:
                        pass

        # Top-level flags
        if "lambda_mode" in cfg:
            try:
                eng_cfg.lambda_mode = str(cfg.get("lambda_mode"))
            except Exception:
                pass
        if "use_tick_tock_active" in cfg:
            try:
                eng_cfg.use_tick_tock_active = bool(
                    cfg.get("use_tick_tock_active")
                )
            except Exception:
                pass

        # Enforce the requested production settings
        eng_cfg.decay.half_life_days = 180.0
        eng_cfg.pagerank.alpha = 0.85
        eng_cfg.engine.beta = 1.0
        eng_cfg.lambda_mode = "auto"
        eng_cfg.tick_tock.convergence_tol = 0.01
        eng_cfg.tick_tock.max_ticks = 5
        eng_cfg.tick_tock.influence_method = "log_top_20_sum"
        eng_cfg.engine.score_decay_delay_days = 180
        eng_cfg.engine.score_decay_rate = 0.01
        eng_cfg.use_tick_tock_active = True

        eng = ExposureLogOddsEngine(eng_cfg)
        if appearances.is_empty():
            ranks = eng.rank_players(matches, players)
        else:
            ranks = eng.rank_players(matches, players, appearances=appearances)
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
            # Compute and persist per-player stats for eligibility display
            try:
                stats = _compute_player_stats(matches, players, appearances)
                if stats is not None and not stats.is_empty():
                    # Restrict stats to ranked IDs for this run
                    id_df = ranks.select(
                        pl.col("id").alias("player_id")
                    ).unique()
                    stats = stats.join(id_df, on="player_id", how="inner")
                    s_ins = _persist_ranking_stats(
                        engine, stats, build_version, calculated_at_ms
                    )
                    log.info(
                        "Saved %d rankings and %d stats to DB (build=%s)",
                        inserted,
                        s_ins,
                        build_version,
                    )
                else:
                    log.info(
                        "Saved %d rankings to DB (no stats derived) (build=%s)",
                        inserted,
                        build_version,
                    )
            except Exception as e:
                log.warning(
                    "Saved %d rankings to DB (stats persist failed: %s) (build=%s)",
                    inserted,
                    e,
                    build_version,
                )
            ensure_tournament_event_times_view(engine)
        log.info("Engine run complete: %d rankings written", ranks_rows)
    else:
        log.info("Skipping engine run (empty matches or players).")

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
            "retrieval_days": since_days,
            "engine_half_life_days": (
                eng_cfg.decay.half_life_days
                if not matches.is_empty() and not players.is_empty()
                else None
            ),
            "until_days": until_days,
        },
    )

    if upload_s3:
        upload_outputs(out_run, s3_prefix=s3_prefix_cfg)

    log.info("Run complete. Outputs: %s", out_run)
    return 0


def _upload_outputs_to_s3(out_run: Path) -> None:
    # Backwards-compatible shim; delegate to utility module
    try:
        upload_outputs(out_run)
    except Exception as e:
        logging.getLogger("rankings.cli.update").error("Upload failed: %s", e)


if __name__ == "__main__":
    raise SystemExit(main())
