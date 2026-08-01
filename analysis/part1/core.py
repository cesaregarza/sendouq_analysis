"""Shared data, rating, and statistical utilities for the Part I audit.

The public functions in this module deliberately separate two protocols:

* ``load_legacy_tables`` reproduces the repository's historical recursive loader.
* ``load_canonical_tables`` selects one deterministic, most-complete record per
  tournament before parsing.  This is the revised analysis protocol.

The distinction matters because the raw directory contains overlapping batches,
snapshots, and per-tournament files.  It must never be hidden by a generic
``drop_duplicates`` call.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
import importlib.metadata
import importlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import minimize_scalar
from scipy.sparse import csr_matrix, vstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from openskill.models import PlackettLuce

from rankings import load_scraped_tournaments, parse_tournaments_data
from rankings.algorithms import ExposureLogOddsEngine
from rankings.analysis.utils.tournament_filters import filter_ranked_tournaments
from rankings.core import ExposureLogOddsConfig, convert_matches_dataframe
from rankings.scraping.storage import extract_appearances_from_players_payload


SECONDS_PER_DAY = 86_400.0
DEFAULT_SEED = 20_260_731
DEFAULT_PLUS_DATE = "2025-11-16"
DEFAULT_PREDICTION_START = "2025-08-17"
DEFAULT_CALIBRATION_START = "2025-07-15"


@dataclass
class EvaluationData:
    """Prepared, canonical tournament data and resolved match lineups."""

    project_root: Path
    data_dir: Path
    tables: dict[str, pl.DataFrame]
    matches: pl.DataFrame
    players: pl.DataFrame
    tournaments: pl.DataFrame
    appearances: pl.DataFrame
    resolved: pd.DataFrame
    canonical_sources: pd.DataFrame
    duplicate_audit: dict[str, Any]


def utc_timestamp(value: str | datetime) -> float:
    if isinstance(value, str):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        dt = value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return float(dt.timestamp())


def utc_date(ts: float | int) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d")


def _json_files(data_dir: Path) -> list[Path]:
    per_id = list(data_dir.rglob("tournament_*.json"))
    snapshots = [
        p
        for p in data_dir.rglob("tournaments_*.json")
        if not p.name.startswith("tournaments_continuous_")
    ]
    continuous = list(data_dir.rglob("tournaments_continuous_*.json"))
    return sorted(set(per_id + snapshots + continuous))


def _entries_with_sources(data_dir: Path) -> list[tuple[dict, str, int]]:
    rows: list[tuple[dict, str, int]] = []
    for path in _json_files(data_dir):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        entries = payload if isinstance(payload, list) else [payload]
        rel = str(path.relative_to(data_dir))
        for index, entry in enumerate(entries):
            if isinstance(entry, dict):
                rows.append((entry, rel, index))
    return rows


def _record_characteristics(entry: dict) -> dict[str, Any]:
    tournament = entry.get("tournament") or {}
    data = tournament.get("data") or {}
    ctx = tournament.get("ctx") or {}
    matches = data.get("match") or []
    teams = ctx.get("teams") or []
    participants = ctx.get("participatedUsers") or []
    decided = 0
    latest_ts = 0
    for match in matches:
        one = match.get("opponent1") or {}
        two = match.get("opponent2") or {}
        if one.get("result") in {"win", "loss"} or two.get("result") in {
            "win",
            "loss",
        }:
            decided += 1
        latest_ts = max(
            latest_ts,
            int(match.get("lastGameFinishedAt") or match.get("createdAt") or 0),
        )
    appearance_rows = 0
    try:
        appearance_rows = len(
            extract_appearances_from_players_payload(
                int(ctx.get("id")), entry.get("player_matches")
            )
        )
    except Exception:
        appearance_rows = 0
    digest = sha256(
        json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "tournament_id": int(ctx["id"]),
        "match_count": len(matches),
        "decided_match_count": decided,
        "latest_match_ts": latest_ts,
        "team_count": len(teams),
        "participant_count": len(participants),
        "appearance_row_count": appearance_rows,
        "is_finalized": bool(ctx.get("isFinalized", False)),
        "sha256": digest,
    }


def load_canonical_records(
    data_dir: Path,
) -> tuple[list[dict], pd.DataFrame, dict[str, Any], pl.DataFrame]:
    """Choose one deterministic record per tournament and merge appearances.

    Selection is lexicographic by usable outcome completeness first, then by
    latest observed match time, bracket completeness, appearance coverage, and
    stable content hash.  Appearance payloads are supplemental and are safely
    unioned across every copy using their stable tournament/match/player keys.
    """

    entries = _entries_with_sources(data_dir)
    by_id: dict[int, list[tuple[dict, dict[str, Any], str, int]]] = defaultdict(list)
    all_appearance_rows: list[dict] = []
    for entry, source, index in entries:
        char = _record_characteristics(entry)
        by_id[char["tournament_id"]].append((entry, char, source, index))
        pm = entry.get("player_matches")
        if pm:
            all_appearance_rows.extend(
                extract_appearances_from_players_payload(char["tournament_id"], pm)
            )

    selected: list[dict] = []
    source_rows: list[dict[str, Any]] = []
    identical_duplicate_ids = 0
    differing_duplicate_ids = 0
    for tournament_id in sorted(by_id):
        candidates = by_id[tournament_id]
        if len(candidates) > 1:
            if len({row[1]["sha256"] for row in candidates}) == 1:
                identical_duplicate_ids += 1
            else:
                differing_duplicate_ids += 1

        def key(row: tuple[dict, dict[str, Any], str, int]) -> tuple[Any, ...]:
            c = row[1]
            return (
                c["decided_match_count"],
                c["latest_match_ts"],
                c["match_count"],
                c["appearance_row_count"],
                int(c["is_finalized"]),
                c["team_count"],
                c["participant_count"],
                c["sha256"],
            )

        chosen = max(candidates, key=key)
        selected.append(chosen[0])
        source_rows.append(
            {
                **chosen[1],
                "source_file": chosen[2],
                "source_index": chosen[3],
                "candidate_copies": len(candidates),
                "distinct_content_hashes": len(
                    {row[1]["sha256"] for row in candidates}
                ),
            }
        )

    if all_appearance_rows:
        appearances = (
            pl.DataFrame(all_appearance_rows)
            .with_columns(
                [
                    pl.col("tournament_id").cast(pl.Int64, strict=False),
                    pl.col("match_id").cast(pl.Int64, strict=False),
                    pl.col("team_id").cast(pl.Int64, strict=False),
                    pl.col("user_id").cast(pl.Int64, strict=False),
                ]
            )
            .drop_nulls(["tournament_id", "match_id", "user_id"])
            .unique(subset=["tournament_id", "match_id", "user_id"])
        )
    else:
        appearances = pl.DataFrame(
            schema={
                "tournament_id": pl.Int64,
                "match_id": pl.Int64,
                "team_id": pl.Int64,
                "user_id": pl.Int64,
            }
        )

    duplicate_audit = {
        "json_files": len(_json_files(data_dir)),
        "records_loaded": len(entries),
        "unique_tournament_ids": len(by_id),
        "duplicate_tournament_ids": sum(len(v) > 1 for v in by_id.values()),
        "extra_record_copies": sum(max(0, len(v) - 1) for v in by_id.values()),
        "identical_duplicate_ids": identical_duplicate_ids,
        "differing_duplicate_ids": differing_duplicate_ids,
        "max_copies_for_one_id": max(map(len, by_id.values()), default=0),
    }
    return selected, pd.DataFrame(source_rows), duplicate_audit, appearances


def load_legacy_tables(data_dir: Path) -> dict[str, pl.DataFrame]:
    """Historical protocol: recursive load, parser-side first-match wins."""

    return filter_ranked_tournaments(
        parse_tournaments_data(load_scraped_tournaments(str(data_dir)))
    )


def _normalize_lineup(values: Any) -> tuple[int, ...]:
    if values is None:
        return tuple()
    if hasattr(values, "to_list"):
        values = values.to_list()
    return tuple(sorted({int(v) for v in values if v is not None}))


def _resolved_matches(
    matches: pl.DataFrame,
    players: pl.DataFrame,
    appearances: pl.DataFrame,
) -> pd.DataFrame:
    valid = matches.filter(
        pl.col("winner_team_id").is_not_null()
        & pl.col("loser_team_id").is_not_null()
        & ~pl.col("is_bye").fill_null(False)
    )
    now_ts = float(
        valid.select(
            pl.coalesce([pl.col("last_game_finished_at"), pl.col("match_created_at")])
            .cast(pl.Float64)
            .max()
        ).item()
    )
    compact = convert_matches_dataframe(
        matches=valid,
        players=players,
        tournament_influence={},
        now_timestamp=now_ts,
        decay_rate=0.0,
        beta=0.0,
        appearances=appearances if not appearances.is_empty() else None,
        include_share=False,
        streaming=False,
    )
    meta = valid.select(
        [
            "tournament_id",
            "match_id",
            "team1_id",
            "team2_id",
            "winner_team_id",
            "loser_team_id",
            "last_game_finished_at",
            "match_created_at",
            "timestamp_source",
        ]
    ).unique(subset=["match_id"])
    joined = compact.join(meta, on=["tournament_id", "match_id"], how="inner")

    appearance_keys: set[tuple[int, int, int]] = set()
    if not appearances.is_empty():
        scoped = appearances
        if scoped.select(pl.col("team_id").is_null().any()).item():
            lookup = players.select(["tournament_id", "user_id", "team_id"]).unique()
            scoped = scoped.join(lookup, on=["tournament_id", "user_id"], how="left")
            if "team_id_right" in scoped.columns:
                scoped = scoped.with_columns(
                    pl.coalesce(["team_id", "team_id_right"]).alias("team_id")
                ).drop("team_id_right")
        appearance_keys = {
            (int(t), int(m), int(team))
            for t, m, team in scoped.select(["tournament_id", "match_id", "team_id"])
            .drop_nulls()
            .unique()
            .iter_rows()
        }

    pdf = joined.to_pandas()
    pdf["winners"] = pdf["winners"].map(_normalize_lineup)
    pdf["losers"] = pdf["losers"].map(_normalize_lineup)
    pdf = pdf[pdf["winners"].map(len).gt(0) & pdf["losers"].map(len).gt(0)].copy()
    pdf["ts"] = pd.to_numeric(pdf["ts"], errors="coerce")
    pdf["lineup_source"] = [
        (
            "appearance_both"
            if (
                (int(tournament_id), int(match_id), int(winner_id)) in appearance_keys
                and (int(tournament_id), int(match_id), int(loser_id))
                in appearance_keys
            )
            else "appearance_partial"
            if (
                (int(tournament_id), int(match_id), int(winner_id)) in appearance_keys
                or (int(tournament_id), int(match_id), int(loser_id)) in appearance_keys
            )
            else "roster_fallback"
        )
        for tournament_id, match_id, winner_id, loser_id in zip(
            pdf["tournament_id"],
            pdf["match_id"],
            pdf["winner_team_id"],
            pdf["loser_team_id"],
        )
    ]
    return pdf.sort_values(["ts", "tournament_id", "match_id"]).reset_index(drop=True)


def prepare_evaluation_data(
    project_root: Path,
    output_dir: Path | None = None,
) -> EvaluationData:
    data_dir = project_root / "data" / "tournaments"
    records, sources, duplicate_audit, appearances = load_canonical_records(data_dir)
    tables = filter_ranked_tournaments(parse_tournaments_data(records))
    tournaments = tables["tournaments"].unique(subset=["tournament_id"])
    matches = (
        tables["matches"]
        .unique(subset=["match_id"])
        .join(
            tournaments.select(["tournament_id", "start_time"]),
            on="tournament_id",
            how="left",
        )
        .with_columns(
            [
                pl.col("last_game_finished_at").alias("last_game_finished_at_raw"),
                pl.col("match_created_at").alias("match_created_at_raw"),
                pl.when(pl.col("last_game_finished_at").is_not_null())
                .then(pl.lit("last_game_finished_at"))
                .when(pl.col("match_created_at").is_not_null())
                .then(pl.lit("match_created_at"))
                .otherwise(pl.lit("tournament_start_imputed"))
                .alias("timestamp_source"),
                pl.coalesce(
                    [
                        pl.col("last_game_finished_at"),
                        pl.col("match_created_at"),
                        pl.col("start_time"),
                    ]
                )
                .cast(pl.Int64)
                .alias("last_game_finished_at"),
            ]
        )
        .drop("start_time")
    )
    players = tables["players"].unique(subset=["tournament_id", "team_id", "user_id"])
    ranked_ids = set(tournaments["tournament_id"].to_list())
    appearances = appearances.filter(pl.col("tournament_id").is_in(ranked_ids))
    resolved = _resolved_matches(matches, players, appearances)

    if output_dir is not None:
        audit_dir = output_dir / "audit"
        intermediate_dir = output_dir / "intermediate"
        audit_dir.mkdir(parents=True, exist_ok=True)
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        sources.to_csv(audit_dir / "canonical_tournament_sources.csv", index=False)
        (audit_dir / "raw_duplicate_audit.json").write_text(
            json.dumps(duplicate_audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tournaments.write_parquet(intermediate_dir / "tournaments.parquet")
        matches.write_parquet(intermediate_dir / "matches.parquet")
        players.write_parquet(intermediate_dir / "players.parquet")
        appearances.write_parquet(intermediate_dir / "appearances.parquet")
        resolved_for_disk = resolved.copy()
        for col in ["winners", "losers"]:
            resolved_for_disk[col] = resolved_for_disk[col].map(list)
        resolved_for_disk.to_parquet(
            intermediate_dir / "resolved_matches.parquet", index=False
        )

    return EvaluationData(
        project_root=project_root,
        data_dir=data_dir,
        tables=tables,
        matches=matches,
        players=players,
        tournaments=tournaments,
        appearances=appearances,
        resolved=resolved,
        canonical_sources=sources,
        duplicate_audit=duplicate_audit,
    )


def production_config(
    *,
    half_life_days: float | None = 180.0,
    tick_tock: bool = True,
    smoothing: bool = True,
    tournament_beta: float = 1.0,
) -> ExposureLogOddsConfig:
    cfg = ExposureLogOddsConfig()
    cfg.decay.half_life_days = (
        float(half_life_days) if half_life_days is not None else 1.0e12
    )
    cfg.pagerank.alpha = 0.85
    cfg.pagerank.tol = 1.0e-8
    cfg.pagerank.max_iter = 200
    cfg.engine.beta = float(tournament_beta)
    cfg.lambda_mode = "auto" if smoothing else "fixed"
    cfg.fixed_lambda = None if smoothing else 0.0
    cfg.tick_tock.convergence_tol = 0.01
    cfg.tick_tock.max_ticks = 5
    cfg.tick_tock.influence_method = "log_top_20_sum"
    cfg.engine.score_decay_delay_days = 180.0
    cfg.engine.score_decay_rate = 0.01
    cfg.use_tick_tock_active = bool(tick_tock)
    return cfg


def _match_ts_expr() -> pl.Expr:
    return pl.coalesce(
        [pl.col("last_game_finished_at"), pl.col("match_created_at")]
    ).cast(pl.Float64)


def slice_for_cutoff(
    data: EvaluationData,
    cutoff_ts: float,
    matches_override: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    source = data.matches if matches_override is None else matches_override
    matches = source.filter(_match_ts_expr() < float(cutoff_ts))
    tids = matches.select("tournament_id").unique()
    players = data.players.join(tids, on="tournament_id", how="inner")
    appearances = data.appearances.join(
        matches.select(["tournament_id", "match_id"]).unique(),
        on=["tournament_id", "match_id"],
        how="semi",
    )
    return matches, players, appearances


def loopr_rankings(
    data: EvaluationData,
    cutoff_ts: float,
    *,
    half_life_days: float | None = 180.0,
    tick_tock: bool = True,
    smoothing: bool = True,
    tournament_beta: float = 1.0,
    matches_override: pl.DataFrame | None = None,
) -> pd.DataFrame:
    matches, players, appearances = slice_for_cutoff(data, cutoff_ts, matches_override)
    if matches.is_empty():
        return pd.DataFrame(columns=["player_id", "score", "ordinal_rank", "exposure"])
    engine = ExposureLogOddsEngine(
        production_config(
            half_life_days=half_life_days,
            tick_tock=tick_tock,
            smoothing=smoothing,
            tournament_beta=tournament_beta,
        ),
        now_ts=float(cutoff_ts),
    )
    result = engine.rank_players(
        matches,
        players,
        appearances=appearances if not appearances.is_empty() else None,
    )
    required = {"id", "score", "player_rank", "exposure"}
    missing = required - set(result.columns)
    if missing:
        raise RuntimeError(f"LOOPR output missing columns: {sorted(missing)}")
    # Never treat player_rank as a score.  Since loopr-ranking 0.0.1 it is an
    # ordinal row number; historical local code used the same name for score.
    return result.select(
        [
            pl.col("id").cast(pl.Int64).alias("player_id"),
            pl.col("score").cast(pl.Float64),
            pl.col("player_rank").cast(pl.Int64).alias("ordinal_rank"),
            pl.col("exposure").cast(pl.Float64),
        ]
    ).to_pandas()


def score_map(frame: pd.DataFrame, column: str) -> dict[int, float]:
    return {
        int(pid): float(value)
        for pid, value in zip(frame["player_id"], frame[column])
        if pd.notna(pid) and pd.notna(value)
    }


def team_mean(
    lineup: Sequence[int], ratings: Mapping[int, float], prior: float
) -> float:
    values = [float(ratings.get(int(pid), prior)) for pid in lineup]
    return float(np.mean(values)) if values else float(prior)


def fit_temperature(
    winner_differences: Sequence[float],
    *,
    lower: float = 0.01,
    upper: float = 10.0,
) -> float:
    d = np.asarray(winner_differences, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 1.0

    def objective(alpha: float) -> float:
        # -log(sigmoid(alpha*d)) evaluated stably.
        return float(np.mean(np.logaddexp(0.0, -alpha * d)))

    result = minimize_scalar(objective, bounds=(lower, upper), method="bounded")
    return float(result.x)


def openskill_ratings(train: pd.DataFrame) -> tuple[pd.DataFrame, PlackettLuce]:
    model = PlackettLuce()
    ratings: dict[int, Any] = {}
    for row in train.sort_values(["ts", "match_id"]).itertuples(index=False):
        winners = [int(pid) for pid in row.winners]
        losers = [int(pid) for pid in row.losers]
        if not winners or not losers:
            continue
        winner_ratings = [ratings.get(pid, model.rating()) for pid in winners]
        loser_ratings = [ratings.get(pid, model.rating()) for pid in losers]
        updated = model.rate([winner_ratings, loser_ratings])
        ratings.update(zip(winners, updated[0]))
        ratings.update(zip(losers, updated[1]))
    rows = [
        {
            "player_id": pid,
            "mu": float(rating.mu),
            "sigma": float(rating.sigma),
            "ordinal": float(rating.mu - 3.0 * rating.sigma),
        }
        for pid, rating in ratings.items()
    ]
    return pd.DataFrame(rows), model


def winrate_90d_ratings(train: pd.DataFrame, cutoff_ts: float) -> pd.DataFrame:
    recent = train[train["ts"] >= cutoff_ts - 90.0 * SECONDS_PER_DAY]
    wins: Counter[int] = Counter()
    losses: Counter[int] = Counter()
    for row in recent.itertuples(index=False):
        wins.update(map(int, row.winners))
        losses.update(map(int, row.losers))
    players = sorted(set(wins) | set(losses))
    rows = []
    for pid in players:
        wins_n, losses_n = wins[pid], losses[pid]
        p = (wins_n + 1.0) / (wins_n + losses_n + 2.0)
        rows.append(
            {
                "player_id": pid,
                "winrate_90d": (
                    wins_n / (wins_n + losses_n) if wins_n + losses_n else 0.5
                ),
                "winrate_logit": math.log(p / (1.0 - p)),
                "matches_90d": wins_n + losses_n,
            }
        )
    return pd.DataFrame(rows)


def activity_ratings(train: pd.DataFrame, cutoff_ts: float) -> pd.DataFrame:
    recent = train[train["ts"] >= cutoff_ts - 90.0 * SECONDS_PER_DAY]
    matches: Counter[int] = Counter()
    tournaments: dict[int, set[int]] = defaultdict(set)
    last: dict[int, float] = {}
    for row in recent.itertuples(index=False):
        for pid in tuple(row.winners) + tuple(row.losers):
            pid = int(pid)
            matches[pid] += 1
            tournaments[pid].add(int(row.tournament_id))
            last[pid] = max(last.get(pid, 0.0), float(row.ts))
    return pd.DataFrame(
        [
            {
                "player_id": pid,
                "matches_90d": matches[pid],
                "tournaments_90d": len(tournaments[pid]),
                "days_since_match": (cutoff_ts - last[pid]) / SECONDS_PER_DAY,
            }
            for pid in sorted(matches)
        ]
    )


def fit_team_bradley_terry(
    train: pd.DataFrame,
    *,
    regularization_c: float = 1.0,
    seed: int = DEFAULT_SEED,
) -> tuple[dict[int, float], dict[str, Any]]:
    player_ids = sorted(
        {
            int(pid)
            for lineups in (train["winners"], train["losers"])
            for lineup in lineups
            for pid in lineup
        }
    )
    index = {pid: idx for idx, pid in enumerate(player_ids)}
    row_idx: list[int] = []
    col_idx: list[int] = []
    values: list[float] = []
    usable = 0
    for row in train.itertuples(index=False):
        winners = tuple(map(int, row.winners))
        losers = tuple(map(int, row.losers))
        if not winners or not losers:
            continue
        for pid in winners:
            row_idx.append(usable)
            col_idx.append(index[pid])
            values.append(1.0 / len(winners))
        for pid in losers:
            row_idx.append(usable)
            col_idx.append(index[pid])
            values.append(-1.0 / len(losers))
        usable += 1
    if usable == 0 or not player_ids:
        return {}, {"n_matches": 0, "n_players": 0, "C": regularization_c}
    x = csr_matrix(
        (values, (row_idx, col_idx)), shape=(usable, len(player_ids)), dtype=float
    )
    x_sym = vstack([x, -x], format="csr")
    y = np.r_[np.ones(usable, dtype=int), np.zeros(usable, dtype=int)]
    model = LogisticRegression(
        penalty="l2",
        C=float(regularization_c),
        fit_intercept=False,
        solver="liblinear",
        max_iter=300,
        tol=1.0e-6,
        random_state=seed,
    )
    model.fit(x_sym, y)
    coef = model.coef_.ravel()
    return (
        {pid: float(coef[idx]) for pid, idx in index.items()},
        {
            "n_matches": usable,
            "n_players": len(player_ids),
            "C": float(regularization_c),
            "iterations": int(np.max(model.n_iter_)),
        },
    )


def prior_teammate_familiarity(
    lineups: Iterable[Sequence[int]],
    prior_pairs: set[tuple[int, int]],
) -> list[float]:
    values: list[float] = []
    for lineup in lineups:
        pairs = [tuple(sorted(map(int, pair))) for pair in combinations(lineup, 2)]
        values.append(
            float(np.mean([pair in prior_pairs for pair in pairs])) if pairs else 0.0
        )
    return values


def teammate_pairs(matches: pd.DataFrame) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for lineups in (matches["winners"], matches["losers"]):
        for lineup in lineups:
            result.update(
                tuple(sorted(map(int, pair))) for pair in combinations(lineup, 2)
            )
    return result


def active_player_ids(
    resolved: pd.DataFrame,
    cutoff_ts: float,
    active_days: float = 90.0,
    exclude_tournament: int | None = None,
) -> set[int]:
    frame = resolved[resolved["ts"] < cutoff_ts]
    if exclude_tournament is not None:
        frame = frame[frame["tournament_id"] != exclude_tournament]
    latest: dict[int, float] = {}
    for row in frame.itertuples(index=False):
        for pid in tuple(row.winners) + tuple(row.losers):
            pid = int(pid)
            latest[pid] = max(latest.get(pid, 0.0), float(row.ts))
    threshold = cutoff_ts - active_days * SECONDS_PER_DAY
    return {pid for pid, ts in latest.items() if ts >= threshold}


def active_roster_player_ids(
    data: EvaluationData,
    cutoff_ts: float,
    active_days: float = 90.0,
    exclude_tournament: int | None = None,
) -> set[int]:
    """Production-compatible activity: rostered in a recent eligible event.

    Tournament rosters are used instead of only resolved match appearances so
    activity does not depend on whether a tournament happened to have the
    optional per-match appearance payload.  A tournament must contain at least
    one valid decided match in the window.
    """

    frame = data.resolved[
        (data.resolved["ts"] < cutoff_ts)
        & (data.resolved["ts"] >= cutoff_ts - active_days * SECONDS_PER_DAY)
    ]
    if exclude_tournament is not None:
        frame = frame[frame["tournament_id"] != exclude_tournament]
    tournament_ids = set(frame["tournament_id"].astype(int).unique())
    return {
        int(pid)
        for pid in data.players.filter(pl.col("tournament_id").is_in(tournament_ids))[
            "user_id"
        ].to_list()
        if pid is not None
    }


def top_n(
    rankings: pd.DataFrame,
    active_ids: set[int],
    n: int = 100,
) -> pd.DataFrame:
    result = rankings[rankings["player_id"].isin(active_ids)].copy()
    result = result.sort_values(["score", "player_id"], ascending=[False, True])
    result["rank"] = np.arange(1, len(result) + 1)
    return result.head(n).reset_index(drop=True)


def safe_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    if len(y) < 2 or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def pair_weighted_stratified_auc(
    frame: pd.DataFrame,
    score_col: str,
    strata: Sequence[str] = ("tier", "suggested"),
) -> tuple[float | None, int, list[dict[str, Any]]]:
    weighted = 0.0
    total_pairs = 0
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(list(strata), dropna=False):
        n_pass = int(group["passed"].sum())
        n_fail = int((1 - group["passed"]).sum())
        auc = safe_auc(group["passed"], group[score_col])
        pairs = n_pass * n_fail if auc is not None else 0
        if auc is not None:
            weighted += auc * pairs
            total_pairs += pairs
        if not isinstance(key, tuple):
            key = (key,)
        rows.append(
            {
                **dict(zip(strata, key)),
                "n": len(group),
                "pass": n_pass,
                "fail": n_fail,
                "pairs": pairs,
                "auc": auc,
            }
        )
    return (
        weighted / total_pairs if total_pairs else None,
        total_pairs,
        rows,
    )


def macro_tier_auc(
    frame: pd.DataFrame, score_col: str
) -> tuple[float | None, list[dict[str, Any]]]:
    rows = []
    for tier, group in frame.groupby("tier", sort=True):
        auc = safe_auc(group["passed"], group[score_col])
        rows.append(
            {
                "tier": int(tier),
                "n": len(group),
                "pass": int(group["passed"].sum()),
                "fail": int((1 - group["passed"]).sum()),
                "auc": auc,
            }
        )
    valid = [row["auc"] for row in rows if row["auc"] is not None]
    return (float(np.mean(valid)) if valid else None, rows)


def stratified_candidate_bootstrap(
    frame: pd.DataFrame,
    statistic,
    *,
    n_boot: int = 2_000,
    seed: int = DEFAULT_SEED,
    strata: Sequence[str] = ("tier", "suggested", "passed"),
) -> tuple[float | None, float | None]:
    rng = np.random.default_rng(seed)
    groups = [group for _, group in frame.groupby(list(strata), dropna=False)]
    values: list[float] = []
    for _ in range(n_boot):
        samples = [
            group.iloc[rng.integers(0, len(group), size=len(group))]
            for group in groups
            if len(group)
        ]
        sample = pd.concat(samples, ignore_index=True)
        value = statistic(sample)
        if value is not None and np.isfinite(value):
            values.append(float(value))
    if not values:
        return None, None
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def tournament_cluster_bootstrap_difference(
    predictions: pd.DataFrame,
    method: str,
    baseline: str = "LOOPR",
    *,
    n_boot: int = 5_000,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float, float]:
    wide = predictions.pivot_table(
        index=["tournament_id", "match_id"],
        columns="method",
        values="log_loss",
        aggfunc="first",
    ).dropna(subset=[method, baseline])
    wide["difference"] = wide[method] - wide[baseline]
    point = float(wide["difference"].mean())
    clusters = {
        int(tid): group["difference"].to_numpy()
        for tid, group in wide.reset_index().groupby("tournament_id")
    }
    tids = np.array(sorted(clusters), dtype=int)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        sampled = rng.choice(tids, size=len(tids), replace=True)
        diffs = np.concatenate([clusters[int(tid)] for tid in sampled])
        values.append(float(diffs.mean()))
    return point, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def package_versions(repo_root: Path) -> dict[str, Any]:
    packages = {
        "loopr-ranking": "loopr",
        "openskill": "openskill",
        "numpy": "numpy",
        "pandas": "pandas",
        "polars": "polars",
        "pyarrow": "pyarrow",
        "scipy": "scipy",
        "scikit-learn": "sklearn",
        "statsmodels": "statsmodels",
        "matplotlib": "matplotlib",
        "networkx": "networkx",
    }
    versions: dict[str, str | None] = {}
    for package, module_name in packages.items():
        try:
            module = importlib.import_module(module_name)
            versions[package] = getattr(module, "__version__", None)
            if versions[package] is None:
                versions[package] = importlib.metadata.version(package)
        except (ImportError, importlib.metadata.PackageNotFoundError):
            versions[package] = None
    try:
        git_commit = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        git_commit = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": git_commit,
        "packages": versions,
    }


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
