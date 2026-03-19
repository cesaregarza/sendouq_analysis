from __future__ import annotations

"""Utilities to attribute score changes to individual matches via LOO analysis."""

import argparse
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

import polars as pl
from sqlalchemy import text as sa_text

from rankings.algorithms import ExposureLogOddsEngine
from rankings.core import ExposureLogOddsConfig
from rankings.scraping.enrich import (
    apply_enrichment_cache,
    apply_enrichment_db_cache,
    enrich_appearances_team_by_match_api,
)
from rankings.sql import create_all as rankings_create_all
from rankings.sql import create_engine as rankings_create_engine
from rankings.sql.constants import SCHEMA as RANKINGS_SCHEMA
from rankings.sql.load import load_core_tables, load_player_appearances_df


def _load_appearances(
    engine,
    matches: pl.DataFrame,
    players: pl.DataFrame,
    *,
    max_enrich_calls: int,
) -> pl.DataFrame | None:
    """Load per-match appearances and best-effort enrich team assignments."""
    try:
        appearances = load_player_appearances_df(engine)
    except Exception:
        appearances = pl.DataFrame([])

    if appearances.is_empty():
        return None

    # Restrict to the matches we actually loaded
    try:
        key_df = matches.select(["tournament_id", "match_id"]).unique()
        appearances = appearances.join(
            key_df, on=["tournament_id", "match_id"], how="inner", coalesce=True
        )
    except Exception:
        pass

    if appearances.is_empty():
        return None

    try:
        appearances = apply_enrichment_db_cache(appearances, engine)
    except Exception:
        pass

    try:
        cache_dir = os.getenv(
            "RANKINGS_ENRICH_CACHE_DIR", "data/enrichment/appearances_enriched"
        )
        appearances = apply_enrichment_cache(appearances, cache_dir)
    except Exception:
        pass

    try:
        needs_enrichment = False
        if "team_id" not in appearances.columns:
            needs_enrichment = True
        else:
            needs_enrichment = appearances.select(
                pl.col("team_id").is_null().any()
            ).item()
        if needs_enrichment and max_enrich_calls > 0:
            appearances = enrich_appearances_team_by_match_api(
                appearances,
                matches,
                players,
                max_calls=max_enrich_calls,
            )
    except Exception:
        pass

    return appearances if not appearances.is_empty() else None


def _format_tournament_date(timestamp_raw: int | None) -> str | None:
    """Convert epoch timestamp to YYYY-MM-DD, handling ms or seconds."""
    if timestamp_raw is None:
        return None
    try:
        ts_int = int(timestamp_raw)
        # Heuristic: timestamps >1e12 are in ms, otherwise seconds
        if ts_int > 1_000_000_000_000:
            ts_seconds = ts_int / 1000
        else:
            ts_seconds = ts_int
        dt = datetime.fromtimestamp(ts_seconds, tz=timezone.utc)
    except (ValueError, OSError):
        return None
    return dt.strftime("%Y-%m-%d")


def _build_engine_config() -> ExposureLogOddsConfig:
    cfg = ExposureLogOddsConfig()
    cfg.decay.half_life_days = 180.0
    cfg.pagerank.alpha = 0.85
    cfg.engine.beta = 1.0
    cfg.engine.score_decay_delay_days = 180
    cfg.engine.score_decay_rate = 0.01
    cfg.lambda_mode = "auto"
    cfg.tick_tock.convergence_tol = 0.01
    cfg.tick_tock.max_ticks = 5
    cfg.tick_tock.influence_method = "log_top_20_sum"
    cfg.use_tick_tock_active = True
    return cfg


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ExposureLogOddsEngine from the rankings database and compute "
            "leave-one-match-out impacts for a player."
        )
    )
    parser.add_argument("--player-id", type=int, default=123)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--since-days", type=int, default=540)
    parser.add_argument("--only-ranked", action="store_true", default=True)
    parser.add_argument("--max-enrich-calls", type=int, default=500)
    parser.add_argument(
        "--exclude-tournament-ids",
        type=str,
        default=None,
        help=(
            "Comma-separated tournament IDs to exclude regardless of ranked flag"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    engine = rankings_create_engine()
    rankings_create_all(engine)

    if args.since_days is None:
        since_ms: int | None = None
    else:
        since_ms = int(
            (
                datetime.now(timezone.utc)
                - timedelta(days=int(args.since_days))
            ).timestamp()
            * 1000
        )

    core_tables = load_core_tables(
        engine,
        since_ms=since_ms,
        until_ms=None,
        only_ranked=args.only_ranked,
    )
    matches = core_tables.get("matches")
    if matches is None:
        matches = pl.DataFrame([])
    players = core_tables.get("players")
    if players is None:
        players = pl.DataFrame([])

    if matches.is_empty() or players.is_empty():
        raise SystemExit("No matches or players returned from database query")

    appearances = _load_appearances(
        engine,
        matches,
        players,
        max_enrich_calls=args.max_enrich_calls,
    )

    engine_cfg = _build_engine_config()
    elo_engine = ExposureLogOddsEngine(engine_cfg)
    ranks = elo_engine.rank_players(
        matches,
        players,
        appearances=appearances,
    )

    player_row = ranks.filter(pl.col("id") == args.player_id)
    if player_row.is_empty():
        raise SystemExit(f"player_id {args.player_id} not present in rankings")

    elo_engine.prepare_loo_analyzer()
    analyzer = elo_engine.get_loo_analyzer()
    if analyzer is None:
        raise SystemExit("Failed to prepare LOO analyzer")
    impact_df = analyzer.analyze_entity_matches_variant(
        args.player_id,
        variant="exact_combined",
        limit=None,
        include_teleport=True,
        parallel=True,
    )

    if impact_df.is_empty():
        print(f"No matches found for player {args.player_id}")
        return 0

    impact_df = impact_df.filter(pl.col("abs_delta") > 0)
    if impact_df.is_empty():
        print(
            f"No matches materially changed the score for player {args.player_id}"
        )
        return 0

    matches_df = analyzer.matches_df
    meta_cols = [
        col
        for col in [
            "match_id",
            "tournament_id",
            "weight",
            "share",
            "ts",
            "winners",
            "losers",
        ]
        if col in matches_df.columns
    ]
    meta = matches_df.select(meta_cols)

    enriched = impact_df.join(meta, on="match_id", how="left", coalesce=True)

    # Attach tournament metadata
    tournament_ids = [
        int(tid)
        for tid in enriched["tournament_id"].drop_nulls().unique().to_list()
    ]
    tournament_meta_df = pl.DataFrame([])
    if tournament_ids:
        with engine.connect() as conn:
            rows = conn.execute(
                sa_text(
                    f"SELECT tournament_id, name, COALESCE(is_ranked, false) AS is_ranked, start_time_ms "
                    f"FROM {RANKINGS_SCHEMA}.tournaments "
                    "WHERE tournament_id = ANY(:ids)"
                ),
                {"ids": tournament_ids},
            ).fetchall()
        if rows:
            tournament_meta_df = pl.DataFrame(
                {
                    "tournament_id": [int(r[0]) for r in rows],
                    "tournament_name": [r[1] for r in rows],
                    "is_ranked": [bool(r[2]) for r in rows],
                    "tournament_date": [
                        _format_tournament_date(r[3]) for r in rows
                    ],
                }
            )

    if not tournament_meta_df.is_empty():
        enriched = enriched.join(
            tournament_meta_df, on="tournament_id", how="left", coalesce=True
        )
    else:
        enriched = enriched.with_columns(
            pl.lit(None).alias("tournament_name"),
            pl.lit(True).alias("is_ranked"),
            pl.lit(None).alias("tournament_date"),
        )

    ranked_enriched = enriched.filter(pl.col("is_ranked") == True)

    exclude_ids: set[int] = set()
    if args.exclude_tournament_ids:
        try:
            exclude_ids = {
                int(piece.strip())
                for piece in args.exclude_tournament_ids.split(",")
                if piece.strip()
            }
        except ValueError:
            exclude_ids = set()
    if exclude_ids:
        ranked_enriched = ranked_enriched.filter(
            ~pl.col("tournament_id").is_in(list(exclude_ids))
        )

    dropped_unranked = enriched.filter(pl.col("is_ranked") != True).height
    manual_excluded = (
        enriched.filter(pl.col("tournament_id").is_in(list(exclude_ids))).height
        if exclude_ids
        else 0
    )

    if ranked_enriched.is_empty():
        print("No ranked matches remain after exclusions.")
        return 0

    ranked_enriched = ranked_enriched.sort("abs_delta", descending=True).head(
        args.top_n
    )

    user_name_map: dict[int, str] = {}
    if "user_id" in players.columns:
        for row in players.iter_rows(named=True):
            pid = row.get("user_id")
            name = row.get("username")
            if pid is None:
                continue
            int_pid = int(pid)
            if int_pid not in user_name_map and name:
                user_name_map[int_pid] = str(name)

    match_ids = {
        int(mid)
        for mid in ranked_enriched["match_id"].drop_nulls().to_list()
    }
    match_lookup = {
        int(row["match_id"]): row
        for row in matches_df.iter_rows(named=True)
        if row.get("match_id") is not None
        and int(row["match_id"]) in match_ids
    }

    results = []
    for row in ranked_enriched.iter_rows(named=True):
        match_id = int(row["match_id"])
        match_row = match_lookup.get(match_id, {})
        winners = match_row.get("winners") or []
        losers = match_row.get("losers") or []
        is_win = bool(row.get("is_win", False))
        if is_win:
            opponent_ids = [int(pid) for pid in losers]
        else:
            opponent_ids = [int(pid) for pid in winners]
        opponent_names = [
            user_name_map.get(pid, f"Player {pid}") for pid in opponent_ids
        ]

        score_delta = float(row["score_delta"])

        results.append(
            {
                "rank": len(results) + 1,
                "match_id": match_id,
                "tournament_id": row.get("tournament_id"),
                "tournament_name": row.get("tournament_name"),
                "is_ranked": bool(row.get("is_ranked", True)),
                "is_win": is_win,
                "score_delta": score_delta,
                "score_delta_x25": score_delta * 25.0,
                "old_score": float(row["old_score"]),
                "new_score": float(row["new_score"]),
                "abs_delta": float(row["abs_delta"]),
                "keep_delta": -score_delta,
                "keep_delta_x25": -score_delta * 25.0,
                "tournament_date": row.get("tournament_date"),
                "opponent_ids": opponent_ids,
                "opponent_names": opponent_names,
                "weight": row.get("weight"),
                "share": row.get("share"),
            }
        )

    output_df = pl.DataFrame(results)

    current_score = float(player_row["score"][0])
    print(f"Player {args.player_id} current score: {current_score:.6f}")

    excluded_total = dropped_unranked + manual_excluded
    if excluded_total > 0:
        print(
            f"(Excluded {excluded_total} matches due to unranked flag or manual exclusions.)"
        )

    if output_df.is_empty():
        return 0

    display_cols = [
        "rank",
        "match_id",
        "tournament_name",
        "tournament_date",
        "is_win",
        "score_delta",
        "score_delta_x25",
        "keep_delta",
        "keep_delta_x25",
        "opponent_names",
    ]
    for display_row in output_df.select(display_cols).iter_rows(named=True):
        tournament_name = display_row.get("tournament_name") or "?"
        tournament_date = display_row.get("tournament_date") or "?"
        result_label = "Win" if display_row.get("is_win") else "Loss"
        opponents = ", ".join(display_row.get("opponent_names", []))
        print(
            f"{int(display_row['rank']):2d}. match {int(display_row['match_id'])} | "
            f"{tournament_name} | {tournament_date} | {result_label} | "
            f"Δ={float(display_row['score_delta']):+.6f} | Δ×25={float(display_row['score_delta_x25']):+.3f} | "
            f"keep={float(display_row['keep_delta']):+.6f} | keep×25={float(display_row['keep_delta_x25']):+.3f} | "
            f"vs {opponents}"
        )

    output_path = os.getenv("PLAYER_LOO_RESULTS_PATH")
    if output_path and not output_df.is_empty():
        output_df.write_parquet(output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
