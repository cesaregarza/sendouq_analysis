from __future__ import annotations

import argparse
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import polars as pl
from sqlalchemy.dialects.postgresql import insert as pg_insert

from rankings.algorithms import ExposureLogOddsEngine
from rankings.analysis.player_loo_report import (
    _build_engine_config,
    _load_appearances,
)
from rankings.cli import update as update_cli
from rankings.core.logging import setup_logging
from rankings.core.sentry import init_sentry
from rankings.sql import create_all as rankings_create_all
from rankings.sql import create_engine as rankings_create_engine
from rankings.sql import models as RM
from rankings.sql.load import load_core_tables


APPROX_VARIANT = "perturb_2"
EXACT_VARIANT = "exact_combined"
LOO_IMPACT_UPSERT_BATCH_SIZE = 1000


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a weekly LOO snapshot using perturb_2 shortlist selection "
            "and exact per-match recomputation for shortlisted player-match pairs."
        )
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="Optional database URL override",
    )
    parser.add_argument(
        "--since-days",
        type=int,
        default=540,
        help="Time window in days for DB retrieval",
    )
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help=(
            "Anchor timestamp (UTC). Accepts epoch seconds/ms, YYYY-MM-DD, "
            "or ISO8601."
        ),
    )
    parser.add_argument(
        "--calculated-at",
        type=str,
        default=None,
        help="Override calculated_at_ms; same accepted formats as --as-of.",
    )
    parser.add_argument(
        "--build-version",
        type=str,
        default=None,
        help="Explicit build version for the weekly ranking and LOO rows",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of positive and negative perturb_2 matches to shortlist",
    )
    parser.add_argument(
        "--player-limit",
        type=int,
        default=None,
        help="Optional cap on ranked players processed, ordered by player_rank",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum workers for perturb_2 match analysis",
    )
    parser.add_argument(
        "--max-enrich-calls",
        type=int,
        default=500,
        help="Maximum appearance team enrichment calls",
    )
    parser.add_argument(
        "--include-unranked",
        action="store_true",
        help="Include unranked tournaments in the weekly base ranking",
    )
    parser.add_argument(
        "--save-to-db",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist rankings, stats, and weekly LOO rows to Postgres",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Optional base directory for a local parquet bundle. "
            "Defaults to data/weekly_loo when --no-save-to-db is used."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level for stdout output",
    )
    return parser.parse_args(argv)


def _default_build_version(anchor_dt: datetime) -> str:
    return f"weekly-loo-{anchor_dt.date().isoformat()}"


def _slugify_path_component(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    slug = slug.strip("-._")
    return slug or "weekly-loo"


def _resolve_output_run_dir(
    *,
    base_dir: str | None,
    build_version: str,
    calculated_at_ms: int,
) -> Path:
    out_base = Path(base_dir) if base_dir else Path("data/weekly_loo")
    run_name = f"{calculated_at_ms}_{_slugify_path_component(build_version)}"
    return out_base / run_name


def _shortlist_matches(approx_df: pl.DataFrame, top_k: int) -> pl.DataFrame:
    if approx_df is None or approx_df.is_empty() or top_k <= 0:
        return pl.DataFrame([])

    positive = (
        approx_df.filter(pl.col("score_delta") > 0)
        .sort("score_delta", descending=True)
        .head(top_k)
        .with_row_index("approx_positive_rank", offset=1)
        .with_columns(
            pl.lit(None, dtype=pl.Int64).alias("approx_negative_rank")
        )
    )
    negative = (
        approx_df.filter(pl.col("score_delta") < 0)
        .sort("score_delta", descending=False)
        .head(top_k)
        .with_row_index("approx_negative_rank", offset=1)
        .with_columns(
            pl.lit(None, dtype=pl.Int64).alias("approx_positive_rank")
        )
    )

    pieces = []
    if not positive.is_empty():
        pieces.append(positive)
    if not negative.is_empty():
        pieces.append(negative)
    if not pieces:
        return pl.DataFrame([])

    shortlist = pl.concat(pieces, how="diagonal_relaxed")
    keep_cols = [
        "match_id",
        "is_win",
        "old_score",
        "new_score",
        "score_delta",
        "abs_delta",
        "approx_positive_rank",
        "approx_negative_rank",
    ]
    shortlist = shortlist.select(keep_cols)
    return shortlist.group_by("match_id", maintain_order=True).agg(
        [
            pl.col("is_win").first().alias("is_win"),
            pl.col("old_score").first().alias("old_score"),
            pl.col("new_score").first().alias("new_score"),
            pl.col("score_delta").first().alias("score_delta"),
            pl.col("abs_delta").first().alias("abs_delta"),
            pl.col("approx_positive_rank")
            .min()
            .alias("approx_positive_rank"),
            pl.col("approx_negative_rank")
            .min()
            .alias("approx_negative_rank"),
        ]
    )


def _compute_exact_row(
    loo_analyzer,
    player_id: int,
    player_rank: int | None,
    player_score: float | None,
    approx_row: dict[str, Any],
) -> dict[str, Any]:
    match_id = int(approx_row["match_id"])
    impact = loo_analyzer.impact_of_match_on_entity_variant(
        match_id,
        player_id,
        variant=EXACT_VARIANT,
        include_teleport=True,
    )
    if not impact.get("ok"):
        raise RuntimeError(
            f"Exact LOO failed for player_id={player_id} match_id={match_id}: "
            f"{impact.get('reason', 'unknown error')}"
        )

    return {
        "player_id": player_id,
        "match_id": match_id,
        "tournament_id": int(approx_row["tournament_id"]),
        "player_rank": player_rank,
        "player_score": player_score,
        "is_win": bool(approx_row["is_win"]),
        "approx_variant": APPROX_VARIANT,
        "approx_positive_rank": approx_row.get("approx_positive_rank"),
        "approx_negative_rank": approx_row.get("approx_negative_rank"),
        "approx_old_score": float(approx_row["old_score"]),
        "approx_new_score": float(approx_row["new_score"]),
        "approx_score_delta": float(approx_row["score_delta"]),
        "approx_abs_delta": float(approx_row["abs_delta"]),
        "exact_variant": EXACT_VARIANT,
        "exact_old_score": float(impact["old"]["score"]),
        "exact_new_score": float(impact["new"]["score"]),
        "exact_score_delta": float(impact["delta"]["score"]),
        "exact_abs_delta": abs(float(impact["delta"]["score"])),
    }


def _compute_weekly_loo_rows(
    ranks: pl.DataFrame,
    loo_analyzer,
    *,
    top_k: int,
    max_workers: int,
    logger: logging.Logger,
) -> pl.DataFrame:
    if ranks.is_empty():
        return pl.DataFrame([])

    match_meta = loo_analyzer.matches_df.select(
        ["match_id", "tournament_id"]
    ).unique(subset=["match_id"], keep="first")

    results: list[dict[str, Any]] = []
    total_players = ranks.height
    for idx, row in enumerate(ranks.iter_rows(named=True), start=1):
        player_id = int(row["id"])
        player_rank = (
            int(row["leaderboard_rank"])
            if row.get("leaderboard_rank") is not None
            else None
        )
        player_score = (
            float(row["score"])
            if row.get("score") is not None
            else (
                float(row["player_rank"])
                if row.get("player_rank") is not None
                else None
            )
        )

        approx_df = loo_analyzer.analyze_entity_matches_variant(
            player_id,
            variant=APPROX_VARIANT,
            limit=None,
            include_teleport=True,
            parallel=True,
            max_workers=max_workers,
        )
        shortlist = _shortlist_matches(approx_df, top_k)
        if shortlist.is_empty():
            continue

        shortlist = shortlist.join(
            match_meta,
            on="match_id",
            how="left",
            coalesce=True,
        )
        shortlist_rows = shortlist.to_dicts()

        for shortlist_row in shortlist_rows:
            results.append(
                _compute_exact_row(
                    loo_analyzer,
                    player_id,
                    player_rank,
                    player_score,
                    shortlist_row,
                )
            )

        if idx == total_players or idx % 100 == 0:
            logger.info(
                "Processed weekly LOO players: %d/%d (%d rows so far)",
                idx,
                total_players,
                len(results),
            )

    return pl.DataFrame(results) if results else pl.DataFrame([])


def _persist_weekly_loo_impacts(
    engine,
    impacts: pl.DataFrame,
    *,
    build_version: str,
    calculated_at_ms: int,
) -> int:
    if impacts is None or impacts.is_empty():
        return 0

    df = impacts.with_columns(
        [
            pl.lit(int(calculated_at_ms)).alias("calculated_at_ms"),
            pl.lit(build_version).alias("build_version"),
        ]
    )
    rows = [row for row in df.iter_rows(named=True)]
    if not rows:
        return 0

    table = RM.PlayerMatchLooImpact.__table__
    key_columns = {
        "player_id",
        "match_id",
        "calculated_at_ms",
        "build_version",
    }
    total = 0
    with engine.begin() as conn:
        for start in range(0, len(rows), LOO_IMPACT_UPSERT_BATCH_SIZE):
            batch = rows[start : start + LOO_IMPACT_UPSERT_BATCH_SIZE]
            stmt = pg_insert(table).values(batch)
            update_columns = {
                col.name: getattr(stmt.excluded, col.name)
                for col in table.c
                if col.name not in key_columns and col.name != "impact_id"
            }
            stmt = stmt.on_conflict_do_update(
                index_elements=[
                    table.c.player_id,
                    table.c.match_id,
                    table.c.calculated_at_ms,
                    table.c.build_version,
                ],
                set_=update_columns,
            )
            conn.execute(stmt)
            total += len(batch)
    return total


def _write_parquet_artifact(
    run_dir: Path,
    artifact_name: str,
    df: pl.DataFrame | None,
) -> dict[str, Any]:
    if df is None or df.width == 0:
        return {
            "path": None,
            "rows": 0,
            "columns": [],
            "written": False,
        }

    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"{artifact_name}.parquet"
    df.write_parquet(str(path))
    return {
        "path": path.name,
        "rows": int(df.height),
        "columns": list(df.columns),
        "written": True,
    }


def _write_weekly_local_bundle(
    *,
    output_dir: str | None,
    build_version: str,
    calculated_at_ms: int,
    anchor_ms: int,
    since_days: int,
    top_k: int,
    player_limit: int | None,
    max_workers: int,
    max_enrich_calls: int,
    include_unranked: bool,
    save_to_db: bool,
    matches: pl.DataFrame,
    players: pl.DataFrame,
    appearances: pl.DataFrame | None,
    rankings: pl.DataFrame,
    ranking_stats: pl.DataFrame,
    weekly_loo_impacts: pl.DataFrame,
) -> Path:
    run_dir = _resolve_output_run_dir(
        base_dir=output_dir,
        build_version=build_version,
        calculated_at_ms=calculated_at_ms,
    )
    artifacts = {
        "matches": _write_parquet_artifact(run_dir, "matches", matches),
        "players": _write_parquet_artifact(run_dir, "players", players),
        "appearances": _write_parquet_artifact(
            run_dir, "appearances", appearances
        ),
        "rankings": _write_parquet_artifact(run_dir, "rankings", rankings),
        "ranking_stats": _write_parquet_artifact(
            run_dir, "ranking_stats", ranking_stats
        ),
        "weekly_loo_impacts": _write_parquet_artifact(
            run_dir, "weekly_loo_impacts", weekly_loo_impacts
        ),
    }
    update_cli._write_manifest(
        run_dir,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "build_version": build_version,
            "anchor_ms": int(anchor_ms),
            "calculated_at_ms": int(calculated_at_ms),
            "since_days": int(since_days),
            "top_k": int(top_k),
            "player_limit": (
                int(player_limit) if player_limit is not None else None
            ),
            "max_workers": int(max_workers),
            "max_enrich_calls": int(max_enrich_calls),
            "include_unranked": bool(include_unranked),
            "save_to_db": bool(save_to_db),
            "artifacts": artifacts,
        },
    )
    return run_dir


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rankings_logger_state = update_cli._capture_rankings_logger_state()
    engine = None

    try:
        if int(args.top_k) <= 0:
            raise SystemExit("--top-k must be > 0")
        setup_logging(level=args.log_level, format_style="detailed")
    except Exception:
        logging.basicConfig(level=getattr(logging, args.log_level))
    log = logging.getLogger("rankings.cli.weekly_loo")

    anchor_ms = (
        update_cli._parse_ts_to_ms(args.as_of, end_of_day_for_date=True)
        if args.as_of
        else update_cli._dt_to_epoch_ms(datetime.now(timezone.utc))
    )
    anchor_dt = datetime.fromtimestamp(anchor_ms / 1000.0, tz=timezone.utc)
    calculated_at_ms = (
        update_cli._parse_ts_to_ms(
            args.calculated_at, end_of_day_for_date=True
        )
        if args.calculated_at
        else int(anchor_ms)
    )
    build_version = (
        args.build_version
        or os.getenv("RANKINGS_BUILD")
        or _default_build_version(anchor_dt)
    )
    write_local_bundle = bool(args.output_dir) or not bool(args.save_to_db)

    init_sentry(context="rankings_weekly_loo", release=build_version)

    engine = rankings_create_engine(args.db_url)
    try:
        if args.save_to_db:
            rankings_create_all(engine)

        since_ms = update_cli._dt_to_epoch_ms(
            anchor_dt - timedelta(days=int(args.since_days))
        )
        tables = load_core_tables(
            engine,
            since_ms=since_ms,
            until_ms=int(anchor_ms),
            only_ranked=not args.include_unranked,
        )
        matches = tables.get("matches")
        if matches is None:
            matches = pl.DataFrame([])
        players = tables.get("players")
        if players is None:
            players = pl.DataFrame([])
        if matches.is_empty() or players.is_empty():
            log.info("No matches or players available for weekly LOO run.")
            return 0

        appearances = _load_appearances(
            engine,
            matches,
            players,
            max_enrich_calls=args.max_enrich_calls,
        )

        eng = ExposureLogOddsEngine(
            _build_engine_config(),
            now_ts=float(anchor_ms) / 1000.0,
        )
        if appearances is None or appearances.is_empty():
            ranks = eng.rank_players(matches, players)
        else:
            ranks = eng.rank_players(matches, players, appearances=appearances)
        if ranks.is_empty():
            log.info("Weekly base ranking produced no rows.")
            return 0

        stats = update_cli._compute_player_stats(matches, players, appearances)
        if args.save_to_db:
            inserted_ranks = update_cli._persist_rankings(
                engine,
                ranks,
                build_version,
                calculated_at_ms,
            )
            inserted_stats = update_cli._persist_ranking_stats(
                engine,
                stats,
                build_version,
                calculated_at_ms,
            )
            log.info(
                "Persisted weekly base run: rankings=%d stats=%d build=%s",
                inserted_ranks,
                inserted_stats,
                build_version,
            )
        else:
            log.info(
                "Skipping DB persist for weekly base run: rankings=%d stats=%d build=%s",
                ranks.height,
                stats.height,
                build_version,
            )

        cohort = (
            ranks.filter(pl.col("id").is_not_null())
            .sort(["player_rank", "id"], descending=[True, False])
            .with_row_index("leaderboard_rank", offset=1)
        )
        if args.player_limit is not None and args.player_limit > 0:
            cohort = cohort.head(int(args.player_limit))
        eng.prepare_loo_analyzer()
        loo_analyzer = eng.get_loo_analyzer()
        if loo_analyzer is None:
            raise RuntimeError("Failed to prepare LOO analyzer")

        loo_rows = _compute_weekly_loo_rows(
            cohort,
            loo_analyzer,
            top_k=int(args.top_k),
            max_workers=max(1, int(args.max_workers)),
            logger=log,
        )
        if write_local_bundle:
            run_dir = _write_weekly_local_bundle(
                output_dir=args.output_dir,
                build_version=build_version,
                calculated_at_ms=calculated_at_ms,
                anchor_ms=int(anchor_ms),
                since_days=int(args.since_days),
                top_k=int(args.top_k),
                player_limit=args.player_limit,
                max_workers=max(1, int(args.max_workers)),
                max_enrich_calls=int(args.max_enrich_calls),
                include_unranked=bool(args.include_unranked),
                save_to_db=bool(args.save_to_db),
                matches=matches,
                players=players,
                appearances=appearances,
                rankings=ranks,
                ranking_stats=stats,
                weekly_loo_impacts=loo_rows,
            )
            log.info("Wrote weekly LOO local bundle to %s", run_dir)
        if args.save_to_db:
            inserted_loo = _persist_weekly_loo_impacts(
                engine,
                loo_rows,
                build_version=build_version,
                calculated_at_ms=calculated_at_ms,
            )
        else:
            inserted_loo = loo_rows.height
        log.info(
            "Weekly LOO complete: cohort=%d persisted_rows=%d build=%s",
            cohort.height,
            inserted_loo,
            build_version,
        )
        return 0
    finally:
        try:
            if engine is not None:
                engine.dispose()
        finally:
            update_cli._restore_rankings_logger_state(rankings_logger_state)


if __name__ == "__main__":
    raise SystemExit(main())
