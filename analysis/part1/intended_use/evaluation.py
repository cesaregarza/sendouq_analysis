"""Comparative intended-use evaluation for public player rankings.

This module is deliberately additive.  It reads the frozen Part I artifacts and
writes only beneath ``analysis/part1/intended_use/output``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.part1.core import (
    DEFAULT_PLUS_DATE,
    DEFAULT_PREDICTION_START,
    DEFAULT_SEED,
    EvaluationData,
    SECONDS_PER_DAY,
    _json_files,
    active_roster_player_ids,
    fit_team_bradley_terry,
    loopr_rankings,
    openskill_ratings,
    safe_auc,
    score_map,
    slice_for_cutoff,
    utc_date,
    utc_timestamp,
    winrate_90d_ratings,
)


METHODS = (
    "LOOPR",
    "OpenSkill μ",
    "OpenSkill ordinal",
    "Bradley–Terry (L2)",
    "90-day win rate",
)
PREDICTION_METHODS = METHODS + ("Chance (0.5)",)
SUPPORT_THRESHOLDS = (1, 2, 4, 8, 16, 32)
BOOTSTRAP_REPETITIONS = 2_000


@dataclass(frozen=True)
class RatingMaps:
    """Player score maps and neutral priors for one temporal snapshot."""

    maps: dict[str, dict[int, float]]
    priors: dict[str, float]
    metadata: dict[str, Any]


def _tuple_lineup(value: Any) -> tuple[int, ...]:
    if value is None:
        return tuple()
    if hasattr(value, "to_list"):
        value = value.to_list()
    return tuple(
        sorted({int(player_id) for player_id in value if player_id is not None})
    )


def load_frozen_evaluation(
    repository_root: Path,
    frozen_output: Path,
) -> EvaluationData:
    """Load the frozen canonical intermediates without reparsing raw JSON."""

    intermediate = frozen_output / "intermediate"
    resolved = pd.read_parquet(intermediate / "resolved_matches.parquet")
    resolved["winners"] = resolved["winners"].map(_tuple_lineup)
    resolved["losers"] = resolved["losers"].map(_tuple_lineup)
    canonical_sources = pd.read_csv(
        frozen_output / "audit" / "canonical_tournament_sources.csv"
    )
    duplicate_audit = json.loads(
        (frozen_output / "audit" / "raw_duplicate_audit.json").read_text(
            encoding="utf-8"
        )
    )
    matches = pl.read_parquet(intermediate / "matches.parquet")
    players = pl.read_parquet(intermediate / "players.parquet")
    tournaments = pl.read_parquet(intermediate / "tournaments.parquet")
    appearances = pl.read_parquet(intermediate / "appearances.parquet")
    return EvaluationData(
        project_root=repository_root,
        data_dir=repository_root / "data" / "tournaments",
        tables={},
        matches=matches,
        players=players,
        tournaments=tournaments,
        appearances=appearances,
        resolved=resolved.sort_values(["ts", "tournament_id", "match_id"]).reset_index(
            drop=True
        ),
        canonical_sources=canonical_sources,
        duplicate_audit=duplicate_audit,
    )


def _lineup_rows(
    frame: pd.DataFrame,
) -> Iterable[tuple[tuple[int, ...], tuple[int, ...]]]:
    for row in frame.itertuples(index=False):
        winners = _tuple_lineup(row.winners)
        losers = _tuple_lineup(row.losers)
        if winners and losers:
            yield winners, losers


def graph_metrics_for_players(
    frame: pd.DataFrame,
    player_ids: Iterable[int],
    *,
    compute_path_metrics: bool = True,
) -> pd.DataFrame:
    """Compute outcome-free support metrics in the prior opponent graph."""

    requested = sorted({int(player_id) for player_id in player_ids})
    graph = nx.Graph()
    opponent_counts: dict[int, Counter[int]] = defaultdict(Counter)
    teammate_sets: dict[int, set[int]] = defaultdict(set)
    lineup_counts: dict[int, Counter[tuple[int, ...]]] = defaultdict(Counter)
    prior_matches: Counter[int] = Counter()

    for winners, losers in _lineup_rows(frame):
        all_players = winners + losers
        graph.add_nodes_from(all_players)
        prior_matches.update(all_players)
        for lineup in (winners, losers):
            for player_id in lineup:
                teammate_sets[player_id].update(
                    teammate for teammate in lineup if teammate != player_id
                )
                lineup_counts[player_id][lineup] += 1
        for winner in winners:
            for loser in losers:
                graph.add_edge(winner, loser)
                opponent_counts[winner][loser] += 1
                opponent_counts[loser][winner] += 1

    if graph.number_of_nodes():
        core_numbers = nx.core_number(graph)
        component_map: dict[int, int] = {}
        components = sorted(nx.connected_components(graph), key=len, reverse=True)
        for component in components:
            size = len(component)
            component_map.update({int(player_id): size for player_id in component})
        largest_size = len(components[0]) if components else 0
        bridges = (
            {tuple(sorted(edge)) for edge in nx.bridges(graph)}
            if compute_path_metrics
            else set()
        )
    else:
        core_numbers = {}
        component_map = {}
        largest_size = 0
        bridges = set()

    rows: list[dict[str, Any]] = []
    for player_id in requested:
        counts = opponent_counts[player_id]
        total_opponent_exposure = sum(counts.values())
        effective_opponents = (
            total_opponent_exposure**2 / sum(value**2 for value in counts.values())
            if total_opponent_exposure
            else 0.0
        )
        lineups = lineup_counts[player_id]
        total_lineup_exposure = sum(lineups.values())
        effective_lineups = (
            total_lineup_exposure**2 / sum(value**2 for value in lineups.values())
            if total_lineup_exposure
            else 0.0
        )
        degree = int(graph.degree(player_id)) if player_id in graph else 0
        bridge_incident = (
            sum(
                tuple(sorted((player_id, neighbor))) in bridges
                for neighbor in graph[player_id]
            )
            if compute_path_metrics and player_id in graph
            else 0
        )
        nonbridge_edge_share = (
            float((degree - bridge_incident) / degree)
            if compute_path_metrics and degree
            else np.nan
        )

        two_hop_counts: Counter[int] = Counter()
        if compute_path_metrics and player_id in graph:
            for neighbor in graph[player_id]:
                for destination in graph[neighbor]:
                    if destination != player_id:
                        two_hop_counts[int(destination)] += 1
        two_hop_paths = sum(two_hop_counts.values())
        two_hop_redundancy = (
            1.0 - len(two_hop_counts) / two_hop_paths
            if two_hop_paths >= 2
            else 0.0
            if compute_path_metrics
            else np.nan
        )
        component_size = int(component_map.get(player_id, 0))
        rows.append(
            {
                "player_id": player_id,
                "prior_matches": int(prior_matches[player_id]),
                "unique_opponents": int(len(counts)),
                "effective_opponents": float(effective_opponents),
                "core_number": int(core_numbers.get(player_id, 0)),
                "component_size": component_size,
                "in_largest_component": bool(
                    largest_size > 0 and component_size == largest_size
                ),
                "nonbridge_edge_share": nonbridge_edge_share,
                "two_hop_redundancy": float(two_hop_redundancy),
                "unique_teammates": int(len(teammate_sets[player_id])),
                "effective_lineups": float(effective_lineups),
            }
        )
    return pd.DataFrame(rows)


def graph_support_for_prediction_events(
    data: EvaluationData,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate pre-event support for every held-out match participant."""

    events = (
        predictions.loc[
            predictions["is_report_window"],
            ["tournament_id", "cutoff_ts", "is_report_window"],
        ]
        .drop_duplicates("tournament_id")
        .sort_values(["cutoff_ts", "tournament_id"])
    )
    player_rows: list[pd.DataFrame] = []
    match_rows: list[dict[str, Any]] = []
    for position, event in enumerate(events.itertuples(index=False), start=1):
        tournament_id = int(event.tournament_id)
        cutoff_ts = float(event.cutoff_ts)
        test = data.resolved[data.resolved["tournament_id"] == tournament_id]
        participants = sorted(
            {
                int(player_id)
                for lineups in (test["winners"], test["losers"])
                for lineup in lineups
                for player_id in lineup
            }
        )
        train = data.resolved[
            (data.resolved["ts"] < cutoff_ts)
            & (data.resolved["tournament_id"] != tournament_id)
        ]
        metrics = graph_metrics_for_players(
            train, participants, compute_path_metrics=False
        )
        metrics["tournament_id"] = tournament_id
        metrics["cutoff_ts"] = cutoff_ts
        metrics["is_report_window"] = bool(event.is_report_window)
        player_rows.append(metrics)
        lookup = metrics.set_index("player_id").to_dict("index")
        for match in test.itertuples(index=False):
            lineup = tuple(match.winners) + tuple(match.losers)
            cores = [
                int(lookup.get(int(player_id), {}).get("core_number", 0))
                for player_id in lineup
            ]
            effective = [
                float(lookup.get(int(player_id), {}).get("effective_opponents", 0.0))
                for player_id in lineup
            ]
            unique_opponents = [
                int(lookup.get(int(player_id), {}).get("unique_opponents", 0))
                for player_id in lineup
            ]
            match_rows.append(
                {
                    "tournament_id": tournament_id,
                    "match_id": int(match.match_id),
                    "participants": len(lineup),
                    "minimum_core_number": int(min(cores, default=0)),
                    "median_core_number": float(np.median(cores)) if cores else 0.0,
                    "minimum_effective_opponents": float(min(effective, default=0.0)),
                    "minimum_unique_opponents": int(min(unique_opponents, default=0)),
                    "is_report_window": bool(event.is_report_window),
                }
            )
        print(
            f"support {position}/{len(events)}: tournament {tournament_id} "
            f"({len(participants)} players)",
            flush=True,
        )
    return pd.concat(player_rows, ignore_index=True), pd.DataFrame(match_rows)


def _cluster_mean_interval(
    values: pd.DataFrame,
    value_column: str,
    *,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float, float]:
    event_values = values.groupby("tournament_id")[value_column].mean().dropna()
    observed = float(event_values.mean())
    if len(event_values) < 2:
        return observed, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    array = event_values.to_numpy(dtype=float)
    draws = rng.choice(array, size=(repetitions, len(array)), replace=True).mean(axis=1)
    return observed, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def prediction_by_graph_support(
    predictions: pd.DataFrame,
    match_support: pd.DataFrame,
    *,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize generalization and relative performance over nested cores."""

    frame = predictions[predictions["is_report_window"]].merge(
        match_support.drop(columns=["is_report_window"], errors="ignore"),
        on=["tournament_id", "match_id"],
        how="inner",
        validate="many_to_one",
    )
    total_matches = frame[["tournament_id", "match_id"]].drop_duplicates().shape[0]
    rows: list[dict[str, Any]] = []
    for threshold in SUPPORT_THRESHOLDS:
        scoped = frame[frame["minimum_core_number"] >= threshold].copy()
        coverage_matches = scoped[["tournament_id", "match_id"]].drop_duplicates()
        n_matches = len(coverage_matches)
        n_tournaments = int(coverage_matches["tournament_id"].nunique())
        if n_matches < 200 or n_tournaments < 5:
            break
        loopr = scoped[scoped["method"] == "LOOPR"][
            [
                "tournament_id",
                "match_id",
                "log_loss",
                "accuracy",
            ]
        ].rename(columns={"log_loss": "loopr_log_loss", "accuracy": "loopr_accuracy"})
        for method in PREDICTION_METHODS:
            group = scoped[scoped["method"] == method].copy()
            event = group.groupby("tournament_id", as_index=False).agg(
                event_log_loss=("log_loss", "mean"),
                event_accuracy=("accuracy", "mean"),
            )
            if method == "LOOPR":
                loss_difference = loss_low = loss_high = 0.0
                accuracy_difference = accuracy_low = accuracy_high = 0.0
            else:
                paired = group.merge(
                    loopr,
                    on=["tournament_id", "match_id"],
                    how="inner",
                    validate="one_to_one",
                )
                paired["loss_difference"] = (
                    paired["log_loss"] - paired["loopr_log_loss"]
                )
                paired["accuracy_difference"] = (
                    paired["accuracy"] - paired["loopr_accuracy"]
                )
                loss_difference, loss_low, loss_high = _cluster_mean_interval(
                    paired,
                    "loss_difference",
                    repetitions=repetitions,
                    seed=seed + threshold,
                )
                accuracy_difference, accuracy_low, accuracy_high = (
                    _cluster_mean_interval(
                        paired,
                        "accuracy_difference",
                        repetitions=repetitions,
                        seed=seed + 1_000 + threshold,
                    )
                )
            rows.append(
                {
                    "minimum_core_threshold": threshold,
                    "method": method,
                    "matches": n_matches,
                    "tournaments": n_tournaments,
                    "match_coverage": n_matches / total_matches,
                    "match_micro_log_loss": float(group["log_loss"].mean()),
                    "tournament_macro_log_loss": float(event["event_log_loss"].mean()),
                    "ordering_accuracy": float(group["accuracy"].mean()),
                    "brier_score": float(group["brier"].mean()),
                    "method_minus_loopr_log_loss": loss_difference,
                    "log_loss_difference_ci_low": loss_low,
                    "log_loss_difference_ci_high": loss_high,
                    "method_minus_loopr_accuracy": accuracy_difference,
                    "accuracy_difference_ci_low": accuracy_low,
                    "accuracy_difference_ci_high": accuracy_high,
                }
            )

    wide = frame.pivot_table(
        index=["tournament_id", "match_id", "minimum_core_number"],
        columns="method",
        values="log_loss",
        aggfunc="first",
    ).reset_index()
    slope_rows: list[dict[str, Any]] = []
    x_all = np.log2(1.0 + wide["minimum_core_number"].to_numpy(dtype=float))
    tournament_ids = wide["tournament_id"].to_numpy(dtype=int)
    unique_events = np.unique(tournament_ids)
    rng = np.random.default_rng(seed + 20_000)
    for method in METHODS[1:]:
        valid = wide[[method, "LOOPR"]].notna().all(axis=1).to_numpy()
        x = x_all[valid]
        y = (wide.loc[valid, method] - wide.loc[valid, "LOOPR"]).to_numpy(dtype=float)
        tids = tournament_ids[valid]
        observed = (
            float(np.polyfit(x, y, 1)[0]) if len(np.unique(x)) > 1 else float("nan")
        )
        draws: list[float] = []
        events = np.unique(tids)
        for _ in range(repetitions):
            sampled = rng.choice(events, size=len(events), replace=True)
            indices = np.concatenate(
                [np.flatnonzero(tids == event) for event in sampled]
            )
            if len(np.unique(x[indices])) > 1:
                draws.append(float(np.polyfit(x[indices], y[indices], 1)[0]))
        slope_rows.append(
            {
                "method": method,
                "paired_loss_difference": "method minus LOOPR",
                "support_predictor": "log2(1 + minimum participant core)",
                "slope": observed,
                "ci_low": float(np.quantile(draws, 0.025)) if draws else np.nan,
                "ci_high": float(np.quantile(draws, 0.975)) if draws else np.nan,
                "matches": int(valid.sum()),
                "tournaments": int(len(unique_events)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(slope_rows)


def prediction_in_disjoint_support_bands(
    predictions: pd.DataFrame,
    match_support: pd.DataFrame,
    *,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Summarize the predeclared, mutually exclusive graph-support bands."""

    frame = predictions[predictions["is_report_window"]].merge(
        match_support.drop(columns=["is_report_window"], errors="ignore"),
        on=["tournament_id", "match_id"],
        how="inner",
        validate="many_to_one",
    )
    bands = (
        ("core 0", 0, 0),
        ("core 1–7", 1, 7),
        ("core 8–15", 8, 15),
        ("core 16–31", 16, 31),
        ("core 32+", 32, None),
    )
    total_matches = frame[["tournament_id", "match_id"]].drop_duplicates().shape[0]
    rows: list[dict[str, Any]] = []
    for band_index, (label, lower, upper) in enumerate(bands):
        mask = frame["minimum_core_number"] >= lower
        if upper is not None:
            mask &= frame["minimum_core_number"] <= upper
        scoped = frame[mask].copy()
        unique_matches = scoped[["tournament_id", "match_id"]].drop_duplicates()
        n_matches = int(len(unique_matches))
        n_tournaments = int(unique_matches["tournament_id"].nunique())
        loopr = scoped[scoped["method"] == "LOOPR"][
            ["tournament_id", "match_id", "log_loss", "accuracy"]
        ].rename(columns={"log_loss": "loopr_log_loss", "accuracy": "loopr_accuracy"})
        for method in PREDICTION_METHODS:
            group = scoped[scoped["method"] == method].copy()
            event = group.groupby("tournament_id", as_index=False).agg(
                event_log_loss=("log_loss", "mean"),
                event_accuracy=("accuracy", "mean"),
            )
            if method == "LOOPR":
                loss_difference = loss_low = loss_high = 0.0
                accuracy_difference = accuracy_low = accuracy_high = 0.0
            else:
                paired = group.merge(
                    loopr,
                    on=["tournament_id", "match_id"],
                    how="inner",
                    validate="one_to_one",
                )
                paired["loss_difference"] = (
                    paired["log_loss"] - paired["loopr_log_loss"]
                )
                paired["accuracy_difference"] = (
                    paired["accuracy"] - paired["loopr_accuracy"]
                )
                loss_difference, loss_low, loss_high = _cluster_mean_interval(
                    paired,
                    "loss_difference",
                    repetitions=repetitions,
                    seed=seed + 50_000 + band_index,
                )
                accuracy_difference, accuracy_low, accuracy_high = (
                    _cluster_mean_interval(
                        paired,
                        "accuracy_difference",
                        repetitions=repetitions,
                        seed=seed + 60_000 + band_index,
                    )
                )
            rows.append(
                {
                    "support_band": label,
                    "minimum_core": lower,
                    "maximum_core": upper,
                    "method": method,
                    "matches": n_matches,
                    "tournaments": n_tournaments,
                    "match_coverage": n_matches / total_matches,
                    "match_micro_log_loss": float(group["log_loss"].mean())
                    if len(group)
                    else np.nan,
                    "tournament_macro_log_loss": float(event["event_log_loss"].mean())
                    if len(event)
                    else np.nan,
                    "ordering_accuracy": float(group["accuracy"].mean())
                    if len(group)
                    else np.nan,
                    "brier_score": float(group["brier"].mean())
                    if len(group)
                    else np.nan,
                    "method_minus_loopr_log_loss": loss_difference,
                    "log_loss_difference_ci_low": loss_low,
                    "log_loss_difference_ci_high": loss_high,
                    "method_minus_loopr_accuracy": accuracy_difference,
                    "accuracy_difference_ci_low": accuracy_low,
                    "accuracy_difference_ci_high": accuracy_high,
                    "support_warning": (
                        "descriptive only: fewer than 200 matches or five tournaments"
                        if n_matches < 200 or n_tournaments < 5
                        else ""
                    ),
                }
            )
    return pd.DataFrame(rows)


def fit_rating_maps(
    data: EvaluationData,
    cutoff_ts: float,
    *,
    exclude_tournament: int | None = None,
) -> RatingMaps:
    """Fit all five score systems at one strict temporal cutoff."""

    train = data.resolved[data.resolved["ts"] < float(cutoff_ts)].copy()
    matches_override = data.matches
    if exclude_tournament is not None:
        train = train[train["tournament_id"] != int(exclude_tournament)]
        matches_override = matches_override.filter(
            pl.col("tournament_id") != int(exclude_tournament)
        )

    loopr = loopr_rankings(data, cutoff_ts, matches_override=matches_override)
    openskill, openskill_model = openskill_ratings(train)
    bradley_terry, bt_metadata = fit_team_bradley_terry(
        train,
        regularization_c=1.0,
        seed=DEFAULT_SEED,
    )
    winrate = winrate_90d_ratings(train, cutoff_ts)
    default_rating = openskill_model.rating()
    maps = {
        "LOOPR": score_map(loopr, "score"),
        "OpenSkill μ": score_map(openskill, "mu") if not openskill.empty else {},
        "OpenSkill ordinal": (
            score_map(openskill, "ordinal") if not openskill.empty else {}
        ),
        "Bradley–Terry (L2)": bradley_terry,
        "90-day win rate": (
            score_map(winrate, "winrate_logit") if not winrate.empty else {}
        ),
    }
    priors = {
        "LOOPR": 0.0,
        "OpenSkill μ": float(default_rating.mu),
        "OpenSkill ordinal": float(default_rating.mu - 3.0 * default_rating.sigma),
        "Bradley–Terry (L2)": 0.0,
        "90-day win rate": 0.0,
    }
    metadata = {
        "cutoff_ts": float(cutoff_ts),
        "cutoff_date": utc_date(cutoff_ts - 1.0),
        "training_matches": int(len(train)),
        "training_tournaments": int(train["tournament_id"].nunique()),
        "excluded_tournament": exclude_tournament,
        "bt": bt_metadata,
    }
    return RatingMaps(maps=maps, priors=priors, metadata=metadata)


def rankings_on_universe(
    ratings: RatingMaps,
    player_ids: Iterable[int],
) -> pd.DataFrame:
    """Return deterministic ranks and within-system standardized scores."""

    universe = sorted({int(player_id) for player_id in player_ids})
    rows: list[pd.DataFrame] = []
    for method in METHODS:
        mapping = ratings.maps[method]
        prior = ratings.priors[method]
        frame = pd.DataFrame(
            {
                "player_id": universe,
                "score": [
                    float(mapping.get(player_id, prior)) for player_id in universe
                ],
            }
        )
        mean = float(frame["score"].mean()) if len(frame) else 0.0
        std = float(frame["score"].std(ddof=0)) if len(frame) else 0.0
        frame["standardized_score"] = (frame["score"] - mean) / std if std > 0 else 0.0
        frame = frame.sort_values(
            ["score", "player_id"], ascending=[False, True]
        ).reset_index(drop=True)
        frame["rank"] = np.arange(1, len(frame) + 1)
        frame["percentile"] = (
            1.0 - (frame["rank"] - 1) / max(1, len(frame) - 1)
            if len(frame) > 1
            else 1.0
        )
        frame["method"] = method
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _extract_stage_shape(entry: dict[str, Any]) -> tuple[str | None, int | None, int]:
    tournament = entry.get("tournament") or {}
    ctx = tournament.get("ctx") or {}
    data = tournament.get("data") or {}
    settings = ctx.get("settings") or {}
    progression = settings.get("bracketProgression") or []
    if progression:
        first = progression[0] or {}
        stage_type = first.get("type")
        stage_settings = first.get("settings") or {}
        group_count = stage_settings.get("groupCount")
        return (
            str(stage_type) if stage_type else None,
            int(group_count) if group_count is not None else None,
            len(progression),
        )

    stages = sorted(
        data.get("stage") or [],
        key=lambda stage: (stage.get("number") is None, stage.get("number") or 0),
    )
    if not stages:
        return None, None, 0
    first = stages[0]
    stage_type = first.get("type")
    stage_settings = first.get("settings") or {}
    swiss = stage_settings.get("swiss") or {}
    group_count = swiss.get("groupCount") or stage_settings.get("groupCount")
    return (
        str(stage_type) if stage_type else None,
        int(group_count) if group_count is not None else None,
        len(stages),
    )


def _team_roster(team: dict[str, Any]) -> tuple[int, ...]:
    active = team.get("activeRosterUserIds")
    if active:
        return _tuple_lineup(active)
    return _tuple_lineup(
        member.get("userId")
        for member in (team.get("members") or [])
        if member.get("userId") is not None
    )


def audit_historical_seeds(
    data: EvaluationData,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Audit whether stored seeds can represent an organizer strength order."""

    canonical = data.canonical_sources.copy()
    canonical["source_index"] = canonical["source_index"].astype(int)
    canonical_lookup = {
        (str(row.source_file), int(row.source_index)): int(row.tournament_id)
        for row in canonical.itertuples(index=False)
    }
    eligible_ids = set(data.tournaments["tournament_id"].to_list())
    seed_history: dict[tuple[int, int], set[int]] = defaultdict(set)
    seed_snapshot_counts: Counter[int] = Counter()
    canonical_entries: dict[int, dict[str, Any]] = {}

    for path in _json_files(data.data_dir):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        entries = payload if isinstance(payload, list) else [payload]
        relative = str(path.relative_to(data.data_dir))
        for source_index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            tournament = entry.get("tournament") or {}
            ctx = tournament.get("ctx") or {}
            tournament_id = ctx.get("id")
            if tournament_id is None:
                continue
            tournament_id = int(tournament_id)
            teams = ctx.get("teams") or []
            if any(team.get("seed") is not None for team in teams):
                seed_snapshot_counts[tournament_id] += 1
            for team in teams:
                if team.get("id") is None or team.get("seed") is None:
                    continue
                seed_history[(tournament_id, int(team["id"]))].add(int(team["seed"]))
            expected = canonical_lookup.get((relative, source_index))
            if expected == tournament_id:
                canonical_entries[tournament_id] = entry

    tournament_rows: list[dict[str, Any]] = []
    team_rows: list[dict[str, Any]] = []
    for tournament_id, entry in sorted(canonical_entries.items()):
        tournament = entry.get("tournament") or {}
        ctx = tournament.get("ctx") or {}
        teams = ctx.get("teams") or []
        stage_type, group_count, progression_count = _extract_stage_shape(entry)
        extracted: list[dict[str, Any]] = []
        for team in teams:
            team_id = team.get("id")
            if team_id is None:
                continue
            persistent_team = team.get("team") or {}
            roster = _team_roster(team)
            seed = team.get("seed")
            ordinal = team.get("avgSeedingSkillOrdinal")
            created_at = team.get("createdAt")
            row = {
                "tournament_id": tournament_id,
                "team_id": int(team_id),
                "persistent_team_id": (
                    int(persistent_team["id"])
                    if persistent_team.get("id") is not None
                    else np.nan
                ),
                "team_name": team.get("name"),
                "seed": int(seed) if seed is not None else np.nan,
                "created_at": float(created_at) if created_at is not None else np.nan,
                "avg_seeding_skill_ordinal": (
                    float(ordinal) if ordinal is not None else np.nan
                ),
                "roster": roster,
                "roster_size": len(roster),
                "seed_values_across_snapshots": sorted(
                    seed_history[(tournament_id, int(team_id))]
                ),
                "seed_changed_across_snapshots": len(
                    seed_history[(tournament_id, int(team_id))]
                )
                > 1,
            }
            extracted.append(row)
            team_rows.append(row)

        team_frame = pd.DataFrame(
            extracted,
            columns=[
                "tournament_id",
                "team_id",
                "persistent_team_id",
                "team_name",
                "seed",
                "created_at",
                "avg_seeding_skill_ordinal",
                "roster",
                "roster_size",
                "seed_values_across_snapshots",
                "seed_changed_across_snapshots",
            ],
        )
        seeds = pd.to_numeric(team_frame.get("seed"), errors="coerce")
        n_teams = len(team_frame)
        complete_seed = bool(
            n_teams
            and seeds.notna().all()
            and seeds.nunique() == n_teams
            and sorted(seeds.astype(int).tolist()) == list(range(1, n_teams + 1))
        )
        registration_complete = bool(n_teams and team_frame["created_at"].notna().all())
        ordinal_complete = bool(
            n_teams and team_frame["avg_seeding_skill_ordinal"].notna().all()
        )
        if complete_seed and registration_complete:
            registration_order = team_frame.sort_values(["created_at", "team_id"])[
                "team_id"
            ].tolist()
            seed_order = team_frame.sort_values(["seed", "team_id"])["team_id"].tolist()
            registration_equivalent = registration_order == seed_order
            registration_rank = {
                team_id: rank
                for rank, team_id in enumerate(registration_order, start=1)
            }
            registration_rho = float(
                spearmanr(
                    team_frame["seed"],
                    team_frame["team_id"].map(registration_rank),
                ).statistic
            )
        else:
            registration_equivalent = False
            registration_rho = np.nan
        if complete_seed and ordinal_complete:
            ordinal_order = team_frame.sort_values(
                ["avg_seeding_skill_ordinal", "team_id"],
                ascending=[False, True],
            )["team_id"].tolist()
            seed_order = team_frame.sort_values(["seed", "team_id"])["team_id"].tolist()
            ordinal_equivalent = ordinal_order == seed_order
            ordinal_rank = {
                team_id: rank for rank, team_id in enumerate(ordinal_order, start=1)
            }
            ordinal_rho = float(
                spearmanr(
                    team_frame["seed"],
                    team_frame["team_id"].map(ordinal_rank),
                ).statistic
            )
        else:
            ordinal_equivalent = False
            ordinal_rho = np.nan

        changed_teams = int(team_frame["seed_changed_across_snapshots"].sum())
        seeds_stable = changed_teams == 0
        simple_order_format = bool(
            stage_type in {"single_elimination", "double_elimination"}
            or (stage_type == "swiss" and (group_count in {None, 1}))
        )
        all_rosters_at_least_four = bool(
            n_teams and (team_frame["roster_size"] >= 4).all()
        )
        all_rosters_exactly_four = bool(
            n_teams and (team_frame["roster_size"] == 4).all()
        )
        structural_eligibility = bool(
            tournament_id in eligible_ids
            and n_teams >= 8
            and complete_seed
            and seeds_stable
            and simple_order_format
            and all_rosters_at_least_four
        )
        model_seed_eligibility = structural_eligibility and all_rosters_exactly_four
        if registration_equivalent and ordinal_equivalent:
            seed_pattern = "registration-and-ordinal-equivalent"
        elif registration_equivalent:
            seed_pattern = "registration-order-equivalent"
        elif ordinal_equivalent:
            seed_pattern = "automatic-ordinal-equivalent"
        else:
            seed_pattern = "non-default-like"
        settings = ctx.get("settings") or {}
        tournament_rows.append(
            {
                "tournament_id": tournament_id,
                "name": ctx.get("name"),
                "start_time": ctx.get("startTime"),
                "start_date": utc_date(ctx.get("startTime"))
                if ctx.get("startTime")
                else None,
                "is_ranked_raw": bool(settings.get("isRanked", False)),
                "eligible_ranked_protocol": tournament_id in eligible_ids,
                "n_teams": n_teams,
                "n_seeded_teams": int(seeds.notna().sum()),
                "complete_contiguous_seed_permutation": complete_seed,
                "seed_snapshots": int(seed_snapshot_counts[tournament_id]),
                "teams_changing_seed_across_snapshots": changed_teams,
                "seeds_stable_across_snapshots": seeds_stable,
                "first_stage_type": stage_type,
                "first_stage_group_count": group_count,
                "progression_stage_count": progression_count,
                "simple_one_dimensional_seed_format": simple_order_format,
                "all_rosters_at_least_four": all_rosters_at_least_four,
                "all_rosters_exactly_four": all_rosters_exactly_four,
                "registration_order_complete": registration_complete,
                "registration_order_equivalent": registration_equivalent,
                "seed_registration_spearman": registration_rho,
                "automatic_ordinal_complete": ordinal_complete,
                "automatic_ordinal_equivalent": ordinal_equivalent,
                "seed_automatic_ordinal_spearman": ordinal_rho,
                "seed_pattern": seed_pattern,
                "structurally_eligible_seed_comparison": structural_eligibility,
                "model_seed_comparison_eligible": model_seed_eligibility,
                "provenance": "unknown: no manual/random/copied provenance field",
            }
        )

    tournament_frame = pd.DataFrame(tournament_rows)
    team_frame = pd.DataFrame(team_rows)
    team_frame = team_frame.merge(
        tournament_frame[["tournament_id", "start_time", "n_teams"]],
        on="tournament_id",
        how="left",
        validate="many_to_one",
    )
    team_frame["relative_seed"] = np.where(
        team_frame["n_teams"] > 1,
        (team_frame["seed"] - 1.0) / (team_frame["n_teams"] - 1.0),
        np.nan,
    )
    team_frame = team_frame.sort_values(
        ["start_time", "tournament_id", "team_id"]
    ).reset_index(drop=True)

    def add_prior_seed_columns(
        frame: pd.DataFrame, key: str, prefix: str, valid: pd.Series
    ) -> pd.DataFrame:
        working = frame.copy()
        eligible = working[valid].copy()
        grouped = eligible.groupby(key, sort=False, dropna=True)
        eligible[f"prior_{prefix}_tournament_id"] = grouped["tournament_id"].shift(1)
        eligible[f"prior_{prefix}_seed"] = grouped["seed"].shift(1)
        eligible[f"prior_{prefix}_relative_seed"] = grouped["relative_seed"].shift(1)
        columns = [
            f"prior_{prefix}_tournament_id",
            f"prior_{prefix}_seed",
            f"prior_{prefix}_relative_seed",
        ]
        working = working.join(eligible[columns], how="left")
        working[f"repeated_{prefix}"] = working[f"prior_{prefix}_tournament_id"].notna()
        working[f"same_integer_seed_as_prior_{prefix}"] = np.where(
            working[f"repeated_{prefix}"],
            working["seed"] == working[f"prior_{prefix}_seed"],
            np.nan,
        )
        working[f"abs_relative_seed_change_from_prior_{prefix}"] = (
            working["relative_seed"] - working[f"prior_{prefix}_relative_seed"]
        ).abs()
        return working

    team_frame = add_prior_seed_columns(
        team_frame,
        "roster",
        "exact_roster",
        team_frame["roster"].map(len) >= 4,
    )
    team_frame = add_prior_seed_columns(
        team_frame,
        "persistent_team_id",
        "persistent_team",
        team_frame["persistent_team_id"].notna(),
    )
    team_lookup = team_frame.set_index(["tournament_id", "team_id"])
    match_rows: list[dict[str, Any]] = []
    for match in data.resolved.itertuples(index=False):
        winner_key = (int(match.tournament_id), int(match.winner_team_id))
        loser_key = (int(match.tournament_id), int(match.loser_team_id))
        if winner_key not in team_lookup.index or loser_key not in team_lookup.index:
            continue
        winner = team_lookup.loc[winner_key]
        loser = team_lookup.loc[loser_key]
        if isinstance(winner, pd.DataFrame) or isinstance(loser, pd.DataFrame):
            continue
        winner_seed = winner["seed"]
        loser_seed = loser["seed"]
        if pd.isna(winner_seed) or pd.isna(loser_seed) or winner_seed == loser_seed:
            continue
        higher_seed_won = float(winner_seed < loser_seed)
        ordinal_winner = winner["avg_seeding_skill_ordinal"]
        ordinal_loser = loser["avg_seeding_skill_ordinal"]
        match_rows.append(
            {
                "tournament_id": int(match.tournament_id),
                "match_id": int(match.match_id),
                "winner_seed": int(winner_seed),
                "loser_seed": int(loser_seed),
                "higher_seed_won": higher_seed_won,
                "best_seed_in_match": int(min(winner_seed, loser_seed)),
                "seed_band": (
                    "top 4"
                    if min(winner_seed, loser_seed) <= 4
                    else "5-8"
                    if min(winner_seed, loser_seed) <= 8
                    else "9+"
                ),
                "automatic_ordinal_winner_higher": (
                    float(ordinal_winner > ordinal_loser)
                    if pd.notna(ordinal_winner)
                    and pd.notna(ordinal_loser)
                    and ordinal_winner != ordinal_loser
                    else np.nan
                ),
            }
        )
    seed_matches = pd.DataFrame(match_rows)
    eligible_seed_matches = seed_matches.merge(
        tournament_frame[
            ["tournament_id", "structurally_eligible_seed_comparison", "seed_pattern"]
        ],
        on="tournament_id",
        how="left",
    )
    scoped = eligible_seed_matches[
        eligible_seed_matches["structurally_eligible_seed_comparison"].fillna(False)
    ]
    if len(scoped):
        macro, macro_low, macro_high = _cluster_mean_interval(
            scoped,
            "higher_seed_won",
            repetitions=BOOTSTRAP_REPETITIONS,
            seed=DEFAULT_SEED,
        )
    else:
        macro = macro_low = macro_high = np.nan
    scoped_tournaments = tournament_frame[
        tournament_frame["structurally_eligible_seed_comparison"]
    ]
    exact_repeats = team_frame[
        team_frame["tournament_id"].isin(scoped_tournaments["tournament_id"])
        & team_frame["repeated_exact_roster"]
    ]
    persistent_repeats = team_frame[
        team_frame["tournament_id"].isin(scoped_tournaments["tournament_id"])
        & team_frame["repeated_persistent_team"]
    ]
    summary = {
        "canonical_tournaments": int(len(tournament_frame)),
        "ranked_protocol_tournaments": int(
            tournament_frame["eligible_ranked_protocol"].sum()
        ),
        "tournaments_with_any_seed": int(
            (tournament_frame["n_seeded_teams"] > 0).sum()
        ),
        "complete_seed_permutations": int(
            tournament_frame["complete_contiguous_seed_permutation"].sum()
        ),
        "tournaments_with_seed_changes_across_snapshots": int(
            (tournament_frame["teams_changing_seed_across_snapshots"] > 0).sum()
        ),
        "structurally_eligible_seed_comparison": int(
            tournament_frame["structurally_eligible_seed_comparison"].sum()
        ),
        "model_seed_comparison_eligible": int(
            tournament_frame["model_seed_comparison_eligible"].sum()
        ),
        "seed_pattern_counts_structurally_eligible": {
            str(key): int(value)
            for key, value in tournament_frame.loc[
                tournament_frame["structurally_eligible_seed_comparison"],
                "seed_pattern",
            ]
            .value_counts()
            .items()
        },
        "registration_order_equivalent_structurally_eligible": int(
            scoped_tournaments["registration_order_equivalent"].sum()
        ),
        "median_seed_registration_spearman_structurally_eligible": float(
            scoped_tournaments["seed_registration_spearman"].median()
        ),
        "automatic_ordinal_equivalent_structurally_eligible": int(
            scoped_tournaments["automatic_ordinal_equivalent"].sum()
        ),
        "median_seed_automatic_ordinal_spearman_structurally_eligible": float(
            scoped_tournaments["seed_automatic_ordinal_spearman"].median()
        ),
        "prior_exact_roster_comparisons_structurally_eligible": int(len(exact_repeats)),
        "same_integer_seed_as_prior_exact_roster_fraction": float(
            exact_repeats["same_integer_seed_as_prior_exact_roster"].mean()
        )
        if len(exact_repeats)
        else None,
        "median_abs_relative_seed_change_from_prior_exact_roster": float(
            exact_repeats["abs_relative_seed_change_from_prior_exact_roster"].median()
        )
        if len(exact_repeats)
        else None,
        "prior_persistent_team_comparisons_structurally_eligible": int(
            len(persistent_repeats)
        ),
        "same_integer_seed_as_prior_persistent_team_fraction": float(
            persistent_repeats["same_integer_seed_as_prior_persistent_team"].mean()
        )
        if len(persistent_repeats)
        else None,
        "median_abs_relative_seed_change_from_prior_persistent_team": float(
            persistent_repeats[
                "abs_relative_seed_change_from_prior_persistent_team"
            ].median()
        )
        if len(persistent_repeats)
        else None,
        "eligible_seed_match_micro_higher_seed_win_rate": float(
            scoped["higher_seed_won"].mean()
        )
        if len(scoped)
        else None,
        "eligible_seed_tournament_macro_higher_seed_win_rate": macro,
        "eligible_seed_tournament_macro_ci": [macro_low, macro_high],
        "eligible_seed_matches": int(len(scoped)),
        "eligible_seed_tournaments_with_matches": int(
            scoped["tournament_id"].nunique()
        ),
        "provenance_limit": (
            "team.seed has no repository field identifying manual, random, copied, "
            "or automatic assignment"
        ),
    }
    return tournament_frame, team_frame, seed_matches, summary


def summarize_seed_outcome_bands(
    seed_matches: pd.DataFrame,
    tournaments: pd.DataFrame,
) -> pd.DataFrame:
    """Test whether stored seed information is confined to the top positions."""

    scoped = seed_matches.merge(
        tournaments[["tournament_id", "structurally_eligible_seed_comparison"]],
        on="tournament_id",
        how="inner",
        validate="many_to_one",
    )
    scoped = scoped[scoped["structurally_eligible_seed_comparison"]].copy()
    rows: list[dict[str, Any]] = []
    for band in ("top 4", "5-8", "9+"):
        group = scoped[scoped["seed_band"] == band]
        macro, low, high = _cluster_mean_interval(
            group,
            "higher_seed_won",
            repetitions=BOOTSTRAP_REPETITIONS,
            seed=DEFAULT_SEED + 70_000 + len(rows),
        )
        rows.append(
            {
                "best_seed_in_match_band": band,
                "matches": int(len(group)),
                "tournaments": int(group["tournament_id"].nunique()),
                "match_micro_higher_seed_win_rate": float(
                    group["higher_seed_won"].mean()
                ),
                "tournament_macro_higher_seed_win_rate": macro,
                "macro_ci_low": low,
                "macro_ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def _pair_concordance(model_difference: float, seed_preference: int) -> float:
    if not np.isfinite(model_difference) or model_difference == 0:
        return 0.5
    return float(int(np.sign(model_difference)) == int(seed_preference))


def evaluate_models_against_seeds(
    data: EvaluationData,
    seed_tournaments: pd.DataFrame,
    seed_teams: pd.DataFrame,
    *,
    start_date: str = DEFAULT_PREDICTION_START,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare pre-event team score orderings with audited stored seeds."""

    start_ts = utc_timestamp(start_date)
    eligible = seed_tournaments[
        seed_tournaments["model_seed_comparison_eligible"]
        & (pd.to_numeric(seed_tournaments["start_time"], errors="coerce") >= start_ts)
    ].sort_values(["start_time", "tournament_id"])
    tournament_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    disagreement_rows: list[dict[str, Any]] = []

    for position, event in enumerate(eligible.itertuples(index=False), start=1):
        tournament_id = int(event.tournament_id)
        cutoff_ts = float(event.start_time)
        train = data.resolved[
            (data.resolved["ts"] < cutoff_ts)
            & (data.resolved["tournament_id"] != tournament_id)
        ]
        if train["tournament_id"].nunique() < 50:
            continue
        teams = seed_teams[seed_teams["tournament_id"] == tournament_id].copy()
        teams["roster"] = teams["roster"].map(_tuple_lineup)
        if teams.empty or not (teams["roster"].map(len) == 4).all():
            continue
        roster_players = sorted(
            {player_id for roster in teams["roster"] for player_id in roster}
        )
        support = graph_metrics_for_players(
            train, roster_players, compute_path_metrics=False
        )
        support_lookup = support.set_index("player_id")["core_number"].to_dict()
        ratings = fit_rating_maps(
            data,
            cutoff_ts,
            exclude_tournament=tournament_id,
        )
        team_records: list[dict[str, Any]] = []
        for team in teams.itertuples(index=False):
            roster = _tuple_lineup(team.roster)
            record = {
                "team_id": int(team.team_id),
                "team_name": team.team_name,
                "seed": int(team.seed),
                "minimum_core_number": int(
                    min(
                        (support_lookup.get(player_id, 0) for player_id in roster),
                        default=0,
                    )
                ),
            }
            for method in METHODS:
                mapping = ratings.maps[method]
                prior = ratings.priors[method]
                record[method] = float(
                    np.mean([mapping.get(player_id, prior) for player_id in roster])
                )
            team_records.append(record)
        scored = pd.DataFrame(team_records).sort_values("seed").reset_index(drop=True)

        for method in METHODS:
            tau = kendalltau(-scored["seed"], scored[method]).statistic
            all_pair_values: list[float] = []
            top_pair_values: list[float] = []
            for left_index, right_index in combinations(range(len(scored)), 2):
                left = scored.iloc[left_index]
                right = scored.iloc[right_index]
                seed_preference = int(np.sign(float(right["seed"] - left["seed"])))
                model_difference = float(left[method] - right[method])
                concordance = _pair_concordance(model_difference, seed_preference)
                pair_support = int(
                    min(left["minimum_core_number"], right["minimum_core_number"])
                )
                all_pair_values.append(concordance)
                if max(int(left["seed"]), int(right["seed"])) <= 8:
                    top_pair_values.append(concordance)
                pair_rows.append(
                    {
                        "tournament_id": tournament_id,
                        "tournament_name": event.name,
                        "seed_pattern": event.seed_pattern,
                        "method": method,
                        "left_team_id": int(left["team_id"]),
                        "right_team_id": int(right["team_id"]),
                        "left_seed": int(left["seed"]),
                        "right_seed": int(right["seed"]),
                        "minimum_core_number": pair_support,
                        "model_difference": model_difference,
                        "concordance": concordance,
                        "top_eight_pair": bool(
                            max(int(left["seed"]), int(right["seed"])) <= 8
                        ),
                    }
                )
            tournament_rows.append(
                {
                    "tournament_id": tournament_id,
                    "tournament_name": event.name,
                    "start_date": event.start_date,
                    "seed_pattern": event.seed_pattern,
                    "teams": int(len(scored)),
                    "method": method,
                    "kendall_tau_b": float(tau) if np.isfinite(tau) else np.nan,
                    "all_pair_concordance": float(np.mean(all_pair_values)),
                    "top_eight_pair_concordance": (
                        float(np.mean(top_pair_values)) if top_pair_values else np.nan
                    ),
                    "all_pairs": int(len(all_pair_values)),
                    "top_eight_pairs": int(len(top_pair_values)),
                }
            )

        loopr_scores = scored.set_index("team_id")["LOOPR"].to_dict()
        for baseline in METHODS[1:]:
            baseline_scores = scored.set_index("team_id")[baseline].to_dict()
            for left_index, right_index in combinations(range(len(scored)), 2):
                left = scored.iloc[left_index]
                right = scored.iloc[right_index]
                left_id, right_id = int(left["team_id"]), int(right["team_id"])
                loopr_sign = int(
                    np.sign(loopr_scores[left_id] - loopr_scores[right_id])
                )
                baseline_sign = int(
                    np.sign(baseline_scores[left_id] - baseline_scores[right_id])
                )
                if loopr_sign == 0 or baseline_sign == 0 or loopr_sign == baseline_sign:
                    continue
                seed_sign = int(np.sign(float(right["seed"] - left["seed"])))
                disagreement_rows.append(
                    {
                        "tournament_id": tournament_id,
                        "tournament_name": event.name,
                        "seed_pattern": event.seed_pattern,
                        "baseline": baseline,
                        "left_team_id": left_id,
                        "right_team_id": right_id,
                        "minimum_core_number": int(
                            min(
                                left["minimum_core_number"],
                                right["minimum_core_number"],
                            )
                        ),
                        "seed_supports_loopr": float(seed_sign == loopr_sign),
                        "seed_supports_baseline": float(seed_sign == baseline_sign),
                    }
                )
        print(
            f"seed models {position}/{len(eligible)}: {tournament_id} {event.name}",
            flush=True,
        )
    return (
        pd.DataFrame(tournament_rows),
        pd.DataFrame(pair_rows),
        pd.DataFrame(disagreement_rows),
    )


def summarize_seed_model_agreement(
    tournament_results: pd.DataFrame,
    pair_results: pd.DataFrame,
    disagreements: pd.DataFrame,
    *,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create tournament-macro seed agreement and disagreement summaries."""

    summary_rows: list[dict[str, Any]] = []
    for seed_pattern in ["all", *sorted(tournament_results["seed_pattern"].unique())]:
        tournament_scope = (
            tournament_results
            if seed_pattern == "all"
            else tournament_results[tournament_results["seed_pattern"] == seed_pattern]
        )
        pair_scope = (
            pair_results
            if seed_pattern == "all"
            else pair_results[pair_results["seed_pattern"] == seed_pattern]
        )
        for threshold in (0, *SUPPORT_THRESHOLDS):
            scoped_pairs = pair_scope[pair_scope["minimum_core_number"] >= threshold]
            if scoped_pairs.empty:
                continue
            n_tournaments = int(scoped_pairs["tournament_id"].nunique())
            if threshold > 0 and (len(scoped_pairs) < 200 or n_tournaments < 5):
                break
            for method in METHODS:
                method_pairs = scoped_pairs[scoped_pairs["method"] == method]
                if method_pairs.empty:
                    continue
                concordance, low, high = _cluster_mean_interval(
                    method_pairs,
                    "concordance",
                    repetitions=repetitions,
                    seed=seed + threshold,
                )
                event_method = tournament_scope[tournament_scope["method"] == method]
                summary_rows.append(
                    {
                        "seed_pattern": seed_pattern,
                        "minimum_core_threshold": threshold,
                        "method": method,
                        "tournaments": n_tournaments,
                        "pairs": int(len(method_pairs)),
                        "pair_concordance_tournament_macro": concordance,
                        "pair_concordance_ci_low": low,
                        "pair_concordance_ci_high": high,
                        "kendall_tau_b_tournament_macro": float(
                            event_method["kendall_tau_b"].mean()
                        ),
                        "top_eight_pair_concordance_tournament_macro": float(
                            event_method["top_eight_pair_concordance"].mean()
                        ),
                    }
                )

    disagreement_summary: list[dict[str, Any]] = []
    if not disagreements.empty:
        for seed_pattern in ["all", *sorted(disagreements["seed_pattern"].unique())]:
            base_scope = (
                disagreements
                if seed_pattern == "all"
                else disagreements[disagreements["seed_pattern"] == seed_pattern]
            )
            for threshold in (0, *SUPPORT_THRESHOLDS):
                scoped = base_scope[base_scope["minimum_core_number"] >= threshold]
                if scoped.empty:
                    continue
                n_tournaments = int(scoped["tournament_id"].nunique())
                if threshold > 0 and (len(scoped) < 100 or n_tournaments < 5):
                    break
                for baseline in METHODS[1:]:
                    group = scoped[scoped["baseline"] == baseline].copy()
                    if group.empty:
                        continue
                    group["loopr_minus_baseline"] = (
                        group["seed_supports_loopr"] - group["seed_supports_baseline"]
                    )
                    difference, low, high = _cluster_mean_interval(
                        group,
                        "loopr_minus_baseline",
                        repetitions=repetitions,
                        seed=seed + 5_000 + threshold,
                    )
                    disagreement_summary.append(
                        {
                            "seed_pattern": seed_pattern,
                            "minimum_core_threshold": threshold,
                            "baseline": baseline,
                            "tournaments": int(group["tournament_id"].nunique()),
                            "disagreement_pairs": int(len(group)),
                            "seed_supports_loopr_share": float(
                                group["seed_supports_loopr"].mean()
                            ),
                            "seed_supports_baseline_share": float(
                                group["seed_supports_baseline"].mean()
                            ),
                            "loopr_minus_baseline": difference,
                            "ci_low": low,
                            "ci_high": high,
                        }
                    )
    return pd.DataFrame(summary_rows), pd.DataFrame(disagreement_summary)


def _auc_with_ties(pass_scores: Sequence[float], fail_scores: Sequence[float]) -> float:
    pass_array = np.asarray(pass_scores, dtype=float)
    fail_array = np.asarray(fail_scores, dtype=float)
    comparisons = pass_array[:, None] - fail_array[None, :]
    return float(np.mean((comparisons > 0) + 0.5 * (comparisons == 0)))


def _macro_tier_auc(frame: pd.DataFrame, score_column: str) -> float | None:
    aucs: list[float] = []
    for _, group in frame.groupby("tier"):
        auc = safe_auc(group["passed"], group[score_column])
        if auc is not None:
            aucs.append(float(auc))
    return float(np.mean(aucs)) if aucs else None


def _oof_logistic_probabilities(
    design: pd.DataFrame,
    labels: np.ndarray,
    *,
    folds: int = 10,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    if design.shape[1] == 0:
        probabilities = np.zeros(len(labels), dtype=float)
        for train_index, test_index in splitter.split(np.zeros(len(labels)), labels):
            probabilities[test_index] = float(labels[train_index].mean())
        return probabilities
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=1_000,
            random_state=seed,
        ),
    )
    return cross_val_predict(
        model,
        design.to_numpy(dtype=float),
        labels,
        cv=splitter,
        method="predict_proba",
    )[:, 1]


def decompose_plus_agreement(
    data: EvaluationData,
    plus_candidates: pd.DataFrame,
    *,
    vote_date: str = DEFAULT_PLUS_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Explain pooled Plus AUC using pair types and pre-existing baselines."""

    cutoff_ts = utc_timestamp(vote_date)
    frame = plus_candidates.copy()
    train = data.resolved[data.resolved["ts"] < cutoff_ts]
    support = graph_metrics_for_players(train, frame["player_id"].astype(int))
    support_columns = [
        "player_id",
        "prior_matches",
        "unique_opponents",
        "effective_opponents",
        "core_number",
        "component_size",
        "in_largest_component",
        "nonbridge_edge_share",
        "two_hop_redundancy",
        "unique_teammates",
        "effective_lineups",
    ]
    frame = frame.merge(support[support_columns], on="player_id", how="left")
    numeric_support = [column for column in support_columns if column != "player_id"]
    for column in numeric_support:
        if column == "in_largest_component":
            frame[column] = frame[column].fillna(False).astype(int)
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    passers = frame[frame["passed"] == 1]
    failers = frame[frame["passed"] == 0]
    pair_rows: list[dict[str, Any]] = []
    for passer in passers.itertuples(index=False):
        for failer in failers.itertuples(index=False):
            score_difference = float(passer.score - failer.score)
            pair_rows.append(
                {
                    "passer_id": int(passer.player_id),
                    "failer_id": int(failer.player_id),
                    "same_tier": bool(passer.tier == failer.tier),
                    "same_status": bool(passer.suggested == failer.suggested),
                    "passer_tier": int(passer.tier),
                    "failer_tier": int(failer.tier),
                    "passer_suggested": bool(passer.suggested),
                    "failer_suggested": bool(failer.suggested),
                    "minimum_core_number": int(
                        min(passer.core_number, failer.core_number)
                    ),
                    "loopr_concordance": float(
                        1.0
                        if score_difference > 0
                        else 0.5
                        if score_difference == 0
                        else 0.0
                    ),
                }
            )
    pairs = pd.DataFrame(pair_rows)
    decomposition_rows: list[dict[str, Any]] = []
    total_pairs = len(pairs)
    for same_tier in (True, False):
        for same_status in (True, False):
            group = pairs[
                (pairs["same_tier"] == same_tier)
                & (pairs["same_status"] == same_status)
            ]
            if group.empty:
                continue
            decomposition_rows.append(
                {
                    "pair_type": (
                        ("same tier" if same_tier else "different tier")
                        + ", "
                        + ("same status" if same_status else "different status")
                    ),
                    "same_tier": same_tier,
                    "same_status": same_status,
                    "pairs": int(len(group)),
                    "share_of_all_passer_failer_pairs": float(len(group) / total_pairs),
                    "loopr_concordance": float(group["loopr_concordance"].mean()),
                    "contribution_to_pooled_auc": float(
                        group["loopr_concordance"].sum() / total_pairs
                    ),
                }
            )
    decomposition = pd.DataFrame(decomposition_rows)

    tier_dummies = pd.get_dummies(frame["tier"].astype(str), prefix="tier", dtype=float)
    feature_frame = pd.DataFrame(index=frame.index)
    feature_frame = pd.concat([feature_frame, tier_dummies], axis=1)
    feature_frame["suggested"] = frame["suggested"].astype(float)
    feature_frame["log_matches_90d"] = np.log1p(
        pd.to_numeric(frame["matches_90d_activity"], errors="coerce").fillna(0.0)
    )
    feature_frame["log_tournaments_90d"] = np.log1p(
        pd.to_numeric(frame["tournaments_90d"], errors="coerce").fillna(0.0)
    )
    feature_frame["days_since_match"] = pd.to_numeric(
        frame["days_since_match"], errors="coerce"
    ).fillna(365.0)
    feature_frame["log_prior_matches"] = np.log1p(frame["prior_matches"])
    feature_frame["log_unique_opponents"] = np.log1p(frame["unique_opponents"])
    feature_frame["log_effective_opponents"] = np.log1p(frame["effective_opponents"])
    feature_frame["core_number"] = frame["core_number"].astype(float)
    feature_frame["nonbridge_edge_share"] = frame["nonbridge_edge_share"].astype(float)
    feature_frame["loopr_score"] = frame["score"].astype(float)
    tier_columns = list(tier_dummies.columns)
    activity_columns = [
        "log_matches_90d",
        "log_tournaments_90d",
        "days_since_match",
    ]
    graph_columns = [
        "log_prior_matches",
        "log_unique_opponents",
        "log_effective_opponents",
        "core_number",
        "nonbridge_edge_share",
    ]
    specifications = {
        "intercept only": [],
        "tier only": tier_columns,
        "candidate status only": ["suggested"],
        "tier + candidate status": tier_columns + ["suggested"],
        "activity only": activity_columns,
        "graph support only": graph_columns,
        "institution + activity + support": (
            tier_columns + ["suggested"] + activity_columns + graph_columns
        ),
        "institution + activity + support + LOOPR": (
            tier_columns
            + ["suggested"]
            + activity_columns
            + graph_columns
            + ["loopr_score"]
        ),
        "LOOPR only": ["loopr_score"],
    }
    labels = frame["passed"].to_numpy(dtype=int)
    baseline_rows: list[dict[str, Any]] = []
    for name, columns in specifications.items():
        probabilities = np.clip(
            _oof_logistic_probabilities(feature_frame[columns], labels),
            1e-6,
            1.0 - 1e-6,
        )
        baseline_rows.append(
            {
                "model": name,
                "features": ", ".join(columns)
                if columns
                else "fold training pass rate",
                "candidates": int(len(frame)),
                "out_of_fold_auc": float(roc_auc_score(labels, probabilities)),
                "out_of_fold_log_loss": float(log_loss(labels, probabilities)),
            }
        )
    baselines = pd.DataFrame(baseline_rows)
    base_loss = float(
        baselines.loc[
            baselines["model"] == "institution + activity + support",
            "out_of_fold_log_loss",
        ].iloc[0]
    )
    baselines["log_loss_improvement_vs_institution_activity_support"] = (
        base_loss - baselines["out_of_fold_log_loss"]
    )

    support_rows: list[dict[str, Any]] = []
    for threshold in (0, *SUPPORT_THRESHOLDS):
        scoped = frame[frame["core_number"] >= threshold]
        if threshold > 0 and (len(scoped) < 40 or scoped["passed"].nunique() < 2):
            break
        pooled_auc = safe_auc(scoped["passed"], scoped["score"])
        support_rows.append(
            {
                "minimum_player_core": threshold,
                "candidates": int(len(scoped)),
                "pass": int(scoped["passed"].sum()),
                "fail": int((1 - scoped["passed"]).sum()),
                "pooled_auc": pooled_auc,
                "institutional_macro_tier_auc": _macro_tier_auc(scoped, "score"),
                "incumbents": int((~scoped["suggested"]).sum()),
                "suggested": int(scoped["suggested"].sum()),
            }
        )
    return (
        frame,
        pairs,
        decomposition,
        pd.concat(
            [
                baselines.assign(result_type="cross_validated_baseline"),
                pd.DataFrame(support_rows).assign(result_type="support_stratification"),
            ],
            ignore_index=True,
            sort=False,
        ),
    )


def pooled_outcome_shuffle_null(
    data: EvaluationData,
    votes_common: pd.DataFrame,
    *,
    vote_date: str = DEFAULT_PLUS_DATE,
    repetitions: int = 200,
    seed: int = DEFAULT_SEED,
    checkpoint_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Recompute the outcome-shuffle null for literal all-candidate pooled AUC."""

    cutoff_ts = utc_timestamp(vote_date)
    matches, _, _ = slice_for_cutoff(data, cutoff_ts)
    observed = safe_auc(votes_common["passed"], votes_common["score"])
    rng = np.random.default_rng(seed)
    flips_by_rep = rng.integers(0, 2, size=(repetitions, matches.height), dtype=np.int8)
    rows: list[dict[str, Any]] = []
    completed: set[int] = set()
    if checkpoint_path is not None and checkpoint_path.exists():
        old = pd.read_csv(checkpoint_path)
        rows.extend(old.to_dict("records"))
        completed = set(old["repetition"].astype(int))
    for repetition in range(repetitions):
        if repetition in completed:
            continue
        flip = flips_by_rep[repetition].astype(bool)
        shuffled = matches.with_columns(
            [
                pl.when(pl.Series("flip", flip))
                .then(pl.col("loser_team_id"))
                .otherwise(pl.col("winner_team_id"))
                .alias("winner_team_id"),
                pl.when(pl.Series("flip2", flip))
                .then(pl.col("winner_team_id"))
                .otherwise(pl.col("loser_team_id"))
                .alias("loser_team_id"),
            ]
        )
        rankings = loopr_rankings(data, cutoff_ts, matches_override=shuffled)
        merged = votes_common.drop(columns=["score"], errors="ignore").merge(
            rankings[["player_id", "score"]], on="player_id", how="left"
        )
        valid = merged.dropna(subset=["score"])
        rows.append(
            {
                "repetition": repetition,
                "pooled_auc": safe_auc(valid["passed"], valid["score"]),
                "score_coverage": int(len(valid)),
                "flipped_matches": int(flip.sum()),
            }
        )
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).sort_values("repetition").to_csv(
                checkpoint_path, index=False
            )
        if repetition % 10 == 0 or repetition == repetitions - 1:
            print(f"pooled outcome shuffle {repetition + 1}/{repetitions}", flush=True)
    result = pd.DataFrame(rows).sort_values("repetition").reset_index(drop=True)
    summary = {
        "repetitions": repetitions,
        "seed": seed,
        "null_procedure": (
            "independent fair-coin result reversal for every pre-vote match; "
            "participation, teams, tournaments, timestamps, and volume preserved"
        ),
        "observed_pooled_auc": observed,
        "null_pooled_mean": float(result["pooled_auc"].mean()),
        "null_pooled_q025": float(result["pooled_auc"].quantile(0.025)),
        "null_pooled_q975": float(result["pooled_auc"].quantile(0.975)),
        "null_pooled_exceedance_p": float(
            (1 + (result["pooled_auc"] >= observed).sum()) / (1 + len(result))
        ),
        "candidate_graph_support": (
            "all 310 evaluated candidates are in at least the pre-vote 32-core, "
            "so graph-support stratification is identical to the full candidate set"
        ),
    }
    return result, summary


def _month_end_cutoffs(data: EvaluationData) -> list[float]:
    start = pd.Timestamp(data.resolved["ts"].min(), unit="s", tz="UTC")
    end = pd.Timestamp(data.resolved["ts"].max(), unit="s", tz="UTC")
    months = pd.date_range(
        start=start.normalize(), end=end.normalize(), freq="M", tz="UTC"
    )
    cutoffs = [float((month + pd.Timedelta(days=1)).timestamp()) for month in months]
    if not cutoffs or cutoffs[-1] <= float(data.resolved["ts"].max()):
        cutoffs.append(float(data.resolved["ts"].max()) + 1.0)
    return cutoffs


def _rerank_subset(frame: pd.DataFrame, threshold: int) -> pd.DataFrame:
    scoped = frame[frame["core_number"] >= threshold].copy()
    scoped = scoped.sort_values(
        ["score", "player_id"], ascending=[False, True]
    ).reset_index(drop=True)
    scoped["rank"] = np.arange(1, len(scoped) + 1)
    return scoped


def comparative_monthly_stability(
    data: EvaluationData,
    *,
    minimum_date: str = "2025-01-01",
    active_days: float = 90.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate adjacent-month stability for all systems and dense universes."""

    snapshot_rows: list[pd.DataFrame] = []
    cutoffs = [
        cutoff
        for cutoff in _month_end_cutoffs(data)
        if cutoff >= utc_timestamp(minimum_date)
    ]
    for position, cutoff_ts in enumerate(cutoffs, start=1):
        active = active_roster_player_ids(data, cutoff_ts, active_days)
        train = data.resolved[data.resolved["ts"] < cutoff_ts]
        support = graph_metrics_for_players(train, active, compute_path_metrics=False)[
            ["player_id", "core_number"]
        ]
        ratings = fit_rating_maps(data, cutoff_ts)
        rankings = rankings_on_universe(ratings, active).merge(
            support,
            on="player_id",
            how="left",
        )
        rankings["core_number"] = rankings["core_number"].fillna(0).astype(int)
        for threshold in (0, 8, 16):
            for method in METHODS:
                method_frame = rankings[rankings["method"] == method]
                ranked = _rerank_subset(method_frame, threshold).head(100).copy()
                ranked["support_threshold"] = threshold
                ranked["cutoff_ts"] = cutoff_ts
                ranked["snapshot_date"] = utc_date(cutoff_ts - 1.0)
                snapshot_rows.append(ranked)
        print(
            f"comparative stability {position}/{len(cutoffs)}: "
            f"{utc_date(cutoff_ts - 1.0)}",
            flush=True,
        )

    snapshots = pd.concat(snapshot_rows, ignore_index=True)
    transition_rows: list[dict[str, Any]] = []
    for (method, threshold), group in snapshots.groupby(
        ["method", "support_threshold"]
    ):
        dates = list(dict.fromkeys(group["snapshot_date"]))
        for prior_date, next_date in zip(dates[:-1], dates[1:]):
            prior = group[group["snapshot_date"] == prior_date]
            following = group[group["snapshot_date"] == next_date]
            merged = prior[["player_id", "rank"]].merge(
                following[["player_id", "rank"]],
                on="player_id",
                suffixes=("_a", "_b"),
            )
            movement = (merged["rank_a"] - merged["rank_b"]).abs()
            rho = (
                spearmanr(merged["rank_a"], merged["rank_b"]).statistic
                if len(merged) >= 2
                else np.nan
            )
            transition_rows.append(
                {
                    "method": method,
                    "support_threshold": int(threshold),
                    "month_a": prior_date,
                    "month_b": next_date,
                    "top100_size_a": int(len(prior)),
                    "top100_size_b": int(len(following)),
                    "top100_retained": int(len(merged)),
                    "top100_retention": float(
                        len(merged) / max(1, min(100, len(prior)))
                    ),
                    "spearman_on_retained": float(rho),
                    "median_absolute_rank_movement": float(movement.median())
                    if len(movement)
                    else np.nan,
                    "p90_absolute_rank_movement": float(movement.quantile(0.90))
                    if len(movement)
                    else np.nan,
                }
            )
    return snapshots, pd.DataFrame(transition_rows)


def comparative_important_tournament_removal(
    data: EvaluationData,
    *,
    active_days: float = 90.0,
    event_count: int = 25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove the largest active-8-core events and refit every system."""

    cutoff_ts = float(data.resolved["ts"].max()) + 1.0
    active = active_roster_player_ids(data, cutoff_ts, active_days)
    train = data.resolved[data.resolved["ts"] < cutoff_ts]
    support = graph_metrics_for_players(train, active, compute_path_metrics=False)[
        ["player_id", "core_number"]
    ]
    core_lookup = support.set_index("player_id")["core_number"].to_dict()
    active_core8 = {
        player_id for player_id in active if int(core_lookup.get(player_id, 0)) >= 8
    }
    importance_rows: list[dict[str, Any]] = []
    for tournament_id, group in train.groupby("tournament_id"):
        appearances = 0
        for row in group.itertuples(index=False):
            appearances += sum(
                int(player_id) in active_core8
                for player_id in tuple(row.winners) + tuple(row.losers)
            )
        importance_rows.append(
            {
                "tournament_id": int(tournament_id),
                "active_core8_player_match_appearances": int(appearances),
                "valid_matches": int(len(group)),
            }
        )
    importance = pd.DataFrame(importance_rows).sort_values(
        [
            "active_core8_player_match_appearances",
            "valid_matches",
            "tournament_id",
        ],
        ascending=[False, False, True],
    )
    important = importance.head(event_count).copy()
    tournament_names = dict(
        zip(
            data.tournaments["tournament_id"].to_list(),
            data.tournaments["name"].to_list(),
        )
    )
    important["tournament_name"] = important["tournament_id"].map(tournament_names)

    full_ratings = fit_rating_maps(data, cutoff_ts)
    full_rankings = rankings_on_universe(full_ratings, active).merge(
        support,
        on="player_id",
        how="left",
    )
    full_rankings["core_number"] = full_rankings["core_number"].fillna(0).astype(int)
    result_rows: list[dict[str, Any]] = []
    for position, event in enumerate(important.itertuples(index=False), start=1):
        tournament_id = int(event.tournament_id)
        reduced_ratings = fit_rating_maps(
            data,
            cutoff_ts,
            exclude_tournament=tournament_id,
        )
        reduced_rankings = rankings_on_universe(reduced_ratings, active).merge(
            support,
            on="player_id",
            how="left",
        )
        reduced_rankings["core_number"] = (
            reduced_rankings["core_number"].fillna(0).astype(int)
        )
        for threshold in (0, 8, 16):
            for method in METHODS:
                full = _rerank_subset(
                    full_rankings[full_rankings["method"] == method], threshold
                )
                reduced = _rerank_subset(
                    reduced_rankings[reduced_rankings["method"] == method], threshold
                )
                full_top = full.head(100)
                reduced_top = reduced.head(100)
                reduced_rank = dict(
                    zip(reduced["player_id"].astype(int), reduced["rank"].astype(int))
                )
                original_ids = list(full_top["player_id"].astype(int))
                retained = set(original_ids) & set(reduced_top["player_id"].astype(int))
                movement = np.asarray(
                    [
                        abs(
                            int(
                                full_top.loc[
                                    full_top["player_id"] == player_id, "rank"
                                ].iloc[0]
                            )
                            - reduced_rank[player_id]
                        )
                        for player_id in original_ids
                        if player_id in reduced_rank
                    ],
                    dtype=float,
                )
                result_rows.append(
                    {
                        "removed_tournament_id": tournament_id,
                        "removed_tournament_name": event.tournament_name,
                        "active_core8_player_match_appearances": int(
                            event.active_core8_player_match_appearances
                        ),
                        "valid_matches": int(event.valid_matches),
                        "method": method,
                        "support_threshold": threshold,
                        "top100_denominator": int(len(full_top)),
                        "top100_retained": int(len(retained)),
                        "top100_retention": float(
                            len(retained) / max(1, len(full_top))
                        ),
                        "median_abs_rank_change_original_top100": float(
                            np.median(movement)
                        )
                        if len(movement)
                        else np.nan,
                        "p90_abs_rank_change_original_top100": float(
                            np.quantile(movement, 0.90)
                        )
                        if len(movement)
                        else np.nan,
                        "largest_abs_rank_change_original_top100": float(
                            np.max(movement)
                        )
                        if len(movement)
                        else np.nan,
                    }
                )
        print(
            f"comparative removal {position}/{len(important)}: "
            f"{tournament_id} {event.tournament_name}",
            flush=True,
        )
    return important, pd.DataFrame(result_rows)


def _stable_lineup_hash(lineup: Sequence[int]) -> str:
    return sha256(
        ",".join(str(int(player_id)) for player_id in sorted(lineup)).encode("utf-8")
    ).hexdigest()


def select_attack_lineups(
    data: EvaluationData,
    *,
    active_days: float = 90.0,
    target_count: int = 20,
) -> tuple[
    list[tuple[int, ...]], list[tuple[int, ...]], list[tuple[int, ...]], pd.DataFrame
]:
    """Select target and opponent lineups without using evaluated model scores."""

    cutoff_ts = float(data.resolved["ts"].max()) + 1.0
    active = active_roster_player_ids(data, cutoff_ts, active_days)
    train = data.resolved[data.resolved["ts"] < cutoff_ts]
    recent = train[train["ts"] >= cutoff_ts - active_days * SECONDS_PER_DAY]
    support = graph_metrics_for_players(
        train, active, compute_path_metrics=False
    ).set_index("player_id")
    player_matches: Counter[int] = Counter()
    lineup_matches: Counter[tuple[int, ...]] = Counter()
    lineup_wins: Counter[tuple[int, ...]] = Counter()
    lineup_losses: Counter[tuple[int, ...]] = Counter()
    for row in train.itertuples(index=False):
        player_matches.update(map(int, tuple(row.winners) + tuple(row.losers)))
    for row in recent.itertuples(index=False):
        winner = _tuple_lineup(row.winners)
        loser = _tuple_lineup(row.losers)
        lineup_matches[winner] += 1
        lineup_matches[loser] += 1
        lineup_wins[winner] += 1
        lineup_losses[loser] += 1

    rows: list[dict[str, Any]] = []
    for lineup, matches in lineup_matches.items():
        if len(lineup) != 4:
            continue
        win_rate = lineup_wins[lineup] / matches
        rows.append(
            {
                "lineup": lineup,
                "lineup_hash": _stable_lineup_hash(lineup),
                "matches_90d": int(matches),
                "win_rate_90d": float(win_rate),
                "minimum_core_number": int(
                    min(
                        (
                            support.loc[player_id, "core_number"]
                            if player_id in support.index
                            else 0
                            for player_id in lineup
                        ),
                        default=0,
                    )
                ),
                "minimum_prior_matches": int(
                    min((player_matches[player_id] for player_id in lineup), default=0)
                ),
            }
        )
    lineups = pd.DataFrame(rows)
    weak_candidates = lineups[
        (lineups["matches_90d"] >= 10) & (lineups["win_rate_90d"] <= 0.25)
    ].sort_values("lineup_hash")
    strong_candidates = lineups[
        (lineups["matches_90d"] >= 10) & (lineups["win_rate_90d"] >= 0.75)
    ].sort_values("lineup_hash")
    weak = list(weak_candidates["lineup"])
    strong = list(strong_candidates["lineup"])
    opponent_players = {player_id for lineup in weak + strong for player_id in lineup}

    target_candidates = lineups[
        (lineups["matches_90d"] >= 3)
        & (lineups["minimum_core_number"] >= 8)
        & (lineups["minimum_prior_matches"] >= 20)
    ].sort_values("lineup_hash")
    targets: list[tuple[int, ...]] = []
    used_players: set[int] = set(opponent_players)
    for lineup in target_candidates["lineup"]:
        if set(lineup) & used_players:
            continue
        targets.append(lineup)
        used_players.update(lineup)
        if len(targets) >= target_count:
            break

    lineups["role"] = "not selected"
    lineups.loc[lineups["lineup"].isin(targets), "role"] = "target"
    lineups.loc[lineups["lineup"].isin(weak), "role"] = "weak opponent"
    lineups.loc[lineups["lineup"].isin(strong), "role"] = "strong opponent"
    return targets, weak, strong, lineups


def _empty_row(schema: Mapping[str, pl.DataType]) -> dict[str, Any]:
    return {column: None for column in schema}


def add_synthetic_event(
    data: EvaluationData,
    battles: Sequence[tuple[tuple[int, ...], tuple[int, ...]]],
    *,
    cutoff_ts: float,
    tournament_id: int,
) -> EvaluationData:
    """Append one controlled event with the supplied winner/loser lineups."""

    if not battles:
        return data
    unique_lineups = sorted(
        {lineup for battle in battles for lineup in battle}, key=_stable_lineup_hash
    )
    team_ids = {
        lineup: -10_000_000 - index
        for index, lineup in enumerate(unique_lineups, start=1)
    }
    timestamp = int(math.floor(cutoff_ts - 1.0))
    match_rows: list[dict[str, Any]] = []
    resolved_rows: list[dict[str, Any]] = []
    appearance_rows: list[dict[str, Any]] = []
    for index, (winners, losers) in enumerate(battles, start=1):
        match_id = -100_000_000 - abs(int(tournament_id)) * 100_000 - index
        winner_team_id = team_ids[winners]
        loser_team_id = team_ids[losers]
        row = _empty_row(data.matches.schema)
        row.update(
            {
                "tournament_id": int(tournament_id),
                "stage_id": int(tournament_id),
                "group_id": int(tournament_id),
                "round_id": int(tournament_id),
                "match_id": int(match_id),
                "match_number": index,
                "status": 2,
                "last_game_finished_at": timestamp,
                "match_created_at": timestamp,
                "team1_id": winner_team_id,
                "team1_position": 1,
                "team1_score": 1,
                "team1_result": "win",
                "team2_id": loser_team_id,
                "team2_position": 2,
                "team2_score": 0,
                "team2_result": "loss",
                "winner_team_id": winner_team_id,
                "loser_team_id": loser_team_id,
                "score_diff": 1,
                "total_games": 1,
                "is_bye": False,
                "last_game_finished_at_raw": timestamp,
                "match_created_at_raw": timestamp,
                "timestamp_source": "controlled_intervention",
            }
        )
        match_rows.append(row)
        resolved_rows.append(
            {
                "match_id": match_id,
                "tournament_id": int(tournament_id),
                "winners": winners,
                "losers": losers,
                "weight": 1.0,
                "ts": float(timestamp),
                "team1_id": winner_team_id,
                "team2_id": loser_team_id,
                "winner_team_id": winner_team_id,
                "loser_team_id": loser_team_id,
                "last_game_finished_at": timestamp,
                "match_created_at": float(timestamp),
                "timestamp_source": "controlled_intervention",
                "lineup_source": "controlled_intervention",
            }
        )
        for team_id, lineup in (
            (winner_team_id, winners),
            (loser_team_id, losers),
        ):
            appearance_rows.extend(
                {
                    "tournament_id": int(tournament_id),
                    "match_id": match_id,
                    "team_id": team_id,
                    "user_id": player_id,
                }
                for player_id in lineup
            )

    player_rows: list[dict[str, Any]] = []
    for lineup, team_id in team_ids.items():
        for player_id in lineup:
            row = _empty_row(data.players.schema)
            row.update(
                {
                    "tournament_id": int(tournament_id),
                    "team_id": team_id,
                    "user_id": int(player_id),
                    "username": f"controlled-{player_id}",
                    "is_owner": False,
                    "roster_created_at": timestamp,
                }
            )
            player_rows.append(row)

    new_matches = pl.DataFrame(match_rows, schema=data.matches.schema, strict=False)
    new_players = pl.DataFrame(player_rows, schema=data.players.schema, strict=False)
    new_appearances = pl.DataFrame(
        appearance_rows, schema=data.appearances.schema, strict=False
    ).unique()
    resolved = pd.concat(
        [data.resolved, pd.DataFrame(resolved_rows)], ignore_index=True
    ).sort_values(["ts", "tournament_id", "match_id"])
    return replace(
        data,
        matches=pl.concat([data.matches, new_matches]),
        players=pl.concat([data.players, new_players]).unique(
            subset=["tournament_id", "team_id", "user_id"]
        ),
        appearances=pl.concat([data.appearances, new_appearances]).unique(
            subset=["tournament_id", "match_id", "user_id"]
        ),
        resolved=resolved.reset_index(drop=True),
    )


def _target_change_rows(
    baseline: pd.DataFrame,
    attacked: pd.DataFrame,
    target_players: Iterable[int],
    **labels: Any,
) -> list[dict[str, Any]]:
    targets = {int(player_id) for player_id in target_players}
    merged = baseline[baseline["player_id"].isin(targets)].merge(
        attacked[attacked["player_id"].isin(targets)],
        on=["player_id", "method"],
        suffixes=("_before", "_after"),
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        rows.append(
            {
                **labels,
                "method": row.method,
                "player_id": int(row.player_id),
                "rank_before": int(row.rank_before),
                "rank_after": int(row.rank_after),
                "rank_gain": int(row.rank_before - row.rank_after),
                "percentile_gain": float(row.percentile_after - row.percentile_before),
                "standardized_score_change": float(
                    row.standardized_score_after - row.standardized_score_before
                ),
                "entered_top100": bool(row.rank_before > 100 and row.rank_after <= 100),
            }
        )
    return rows


def adversarial_weak_win_and_schedule_tests(
    data: EvaluationData,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run weak-win farming, strong controls, and equal-record schedule tests."""

    cutoff_ts = float(data.resolved["ts"].max()) + 1.0
    active = active_roster_player_ids(data, cutoff_ts, 90.0)
    targets, weak, strong, selection = select_attack_lineups(data)
    if not targets or not weak or not strong:
        return selection, pd.DataFrame(), pd.DataFrame()
    target_players = {player_id for lineup in targets for player_id in lineup}
    baseline = rankings_on_universe(fit_rating_maps(data, cutoff_ts), active)
    attack_rows: list[dict[str, Any]] = []
    scenario_index = 0
    for opponent_class, opponents in (("weak", weak), ("strong", strong)):
        for wins_per_target in (1, 5, 20, 50):
            battles: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
            for index, target in enumerate(targets):
                opponent = opponents[index % len(opponents)]
                battles.extend([(target, opponent)] * wins_per_target)
            scenario_index += 1
            modified = add_synthetic_event(
                data,
                battles,
                cutoff_ts=cutoff_ts,
                tournament_id=-20_000 - scenario_index,
            )
            attacked = rankings_on_universe(
                fit_rating_maps(modified, cutoff_ts), active
            )
            attack_rows.extend(
                _target_change_rows(
                    baseline,
                    attacked,
                    target_players,
                    attack="repeated wins",
                    opponent_class=opponent_class,
                    wins_per_target=wins_per_target,
                    target_lineups=len(targets),
                    opponent_lineups=len(opponents),
                )
            )
            print(
                f"farming attack {opponent_class} {wins_per_target} wins",
                flush=True,
            )

    # Same 5-5 record against strong versus weak opposition.
    schedule_battles: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    schedule_group_by_player: dict[int, str] = {}
    split = len(targets) // 2
    for index, target in enumerate(targets):
        schedule_class = "strong" if index < split else "weak"
        opponents = strong if schedule_class == "strong" else weak
        opponent = opponents[index % len(opponents)]
        schedule_battles.extend([(target, opponent)] * 5)
        schedule_battles.extend([(opponent, target)] * 5)
        schedule_group_by_player.update(
            {player_id: schedule_class for player_id in target}
        )
    scheduled = add_synthetic_event(
        data,
        schedule_battles,
        cutoff_ts=cutoff_ts,
        tournament_id=-30_000,
    )
    scheduled_rankings = rankings_on_universe(
        fit_rating_maps(scheduled, cutoff_ts), active
    )
    schedule_changes = pd.DataFrame(
        _target_change_rows(
            baseline,
            scheduled_rankings,
            target_players,
            attack="equal 5-5 schedule",
        )
    )
    schedule_changes["opponent_class"] = schedule_changes["player_id"].map(
        schedule_group_by_player
    )
    schedule_summary = (
        schedule_changes.groupby(["method", "opponent_class"], as_index=False)
        .agg(
            target_players=("player_id", "nunique"),
            median_rank_gain=("rank_gain", "median"),
            median_standardized_score_change=("standardized_score_change", "median"),
        )
        .sort_values(["method", "opponent_class"])
    )
    strong_change = schedule_summary[schedule_summary["opponent_class"] == "strong"][
        ["method", "median_standardized_score_change"]
    ].rename(columns={"median_standardized_score_change": "strong_schedule_change"})
    weak_change = schedule_summary[schedule_summary["opponent_class"] == "weak"][
        ["method", "median_standardized_score_change"]
    ].rename(columns={"median_standardized_score_change": "weak_schedule_change"})
    schedule_summary = schedule_summary.merge(strong_change, on="method").merge(
        weak_change, on="method"
    )
    schedule_summary["strong_minus_weak_standardized_change"] = (
        schedule_summary["strong_schedule_change"]
        - schedule_summary["weak_schedule_change"]
    )
    return selection, pd.DataFrame(attack_rows), schedule_summary


def adversarial_transitive_chain_test(
    data: EvaluationData,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add A-over-B and B-over-C chains with no A-C match for every system."""

    cutoff_ts = float(data.resolved["ts"].max()) + 1.0
    active = active_roster_player_ids(data, cutoff_ts, 90.0)
    targets, _, _, selection = select_attack_lineups(data)
    chain_count = len(targets) // 3
    chains = [
        (targets[3 * index], targets[3 * index + 1], targets[3 * index + 2])
        for index in range(chain_count)
    ]
    baseline = rankings_on_universe(fit_rating_maps(data, cutoff_ts), active)
    rows: list[dict[str, Any]] = []
    for repetitions in (1, 5, 20):
        battles: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for lineup_a, lineup_b, lineup_c in chains:
            battles.extend([(lineup_a, lineup_b)] * repetitions)
            battles.extend([(lineup_b, lineup_c)] * repetitions)
        modified = add_synthetic_event(
            data,
            battles,
            cutoff_ts=cutoff_ts,
            tournament_id=-40_000 - repetitions,
        )
        attacked = rankings_on_universe(fit_rating_maps(modified, cutoff_ts), active)
        for method in METHODS:
            before = baseline[baseline["method"] == method].set_index("player_id")
            after = attacked[attacked["method"] == method].set_index("player_id")
            for chain_index, (lineup_a, lineup_b, lineup_c) in enumerate(
                chains, start=1
            ):
                before_a = float(
                    before.loc[list(lineup_a), "standardized_score"].mean()
                )
                before_c = float(
                    before.loc[list(lineup_c), "standardized_score"].mean()
                )
                after_a = float(after.loc[list(lineup_a), "standardized_score"].mean())
                after_c = float(after.loc[list(lineup_c), "standardized_score"].mean())
                rows.append(
                    {
                        "method": method,
                        "repetitions_per_edge": repetitions,
                        "chain_index": chain_index,
                        "lineup_a_hash": _stable_lineup_hash(lineup_a),
                        "lineup_b_hash": _stable_lineup_hash(lineup_b),
                        "lineup_c_hash": _stable_lineup_hash(lineup_c),
                        "a_minus_c_before": before_a - before_c,
                        "a_minus_c_after": after_a - after_c,
                        "induced_a_minus_c_gap_change": (after_a - after_c)
                        - (before_a - before_c),
                        "induced_direction_positive": (after_a - after_c)
                        - (before_a - before_c)
                        > 0,
                        "a_rank_change": float(
                            before.loc[list(lineup_a), "rank"].mean()
                            - after.loc[list(lineup_a), "rank"].mean()
                        ),
                        "c_rank_change": float(
                            before.loc[list(lineup_c), "rank"].mean()
                            - after.loc[list(lineup_c), "rank"].mean()
                        ),
                    }
                )
        print(f"transitive chain {repetitions} repetitions", flush=True)
    return selection, pd.DataFrame(rows)


def duplicate_target_evidence(
    data: EvaluationData,
    target_lineups: Sequence[tuple[int, ...]],
    *,
    copies: int,
) -> EvaluationData:
    """Duplicate each target lineup's last ten observed matches."""

    if copies <= 0:
        return data
    candidate_ids: list[int] = []
    for lineup in target_lineups:
        exact = data.resolved[
            data.resolved.apply(
                lambda row: (
                    _tuple_lineup(row["winners"]) == lineup
                    or _tuple_lineup(row["losers"]) == lineup
                ),
                axis=1,
            )
        ].sort_values(["ts", "match_id"])
        candidate_ids.extend(exact.tail(10)["match_id"].astype(int).tolist())
    match_ids = sorted(set(candidate_ids))
    if not match_ids:
        return data
    raw_source = data.matches.filter(pl.col("match_id").is_in(match_ids))
    resolved_source = data.resolved[data.resolved["match_id"].isin(match_ids)]
    appearance_source = data.appearances.filter(pl.col("match_id").is_in(match_ids))
    raw_copies: list[pl.DataFrame] = []
    resolved_copies: list[pd.DataFrame] = []
    appearance_copies: list[pl.DataFrame] = []
    for copy_index in range(1, copies + 1):
        offset = 200_000_000 + copy_index * 1_000_000
        id_map = {
            match_id: -(offset + position)
            for position, match_id in enumerate(match_ids, start=1)
        }
        raw_copies.append(
            raw_source.with_columns(
                pl.col("match_id")
                .replace_strict(id_map, default=pl.col("match_id"))
                .alias("match_id")
            )
        )
        copied = resolved_source.copy()
        copied["match_id"] = copied["match_id"].map(id_map)
        copied["lineup_source"] = "controlled_volume_duplicate"
        resolved_copies.append(copied)
        appearance_copies.append(
            appearance_source.with_columns(
                pl.col("match_id")
                .replace_strict(id_map, default=pl.col("match_id"))
                .alias("match_id")
            )
        )
    return replace(
        data,
        matches=pl.concat([data.matches, *raw_copies]),
        appearances=pl.concat([data.appearances, *appearance_copies]).unique(
            subset=["tournament_id", "match_id", "user_id"]
        ),
        resolved=pd.concat(
            [data.resolved, *resolved_copies], ignore_index=True
        ).sort_values(["ts", "tournament_id", "match_id"]),
    )


def adversarial_volume_test(data: EvaluationData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Duplicate observed target evidence to 2x and 5x volume."""

    cutoff_ts = float(data.resolved["ts"].max()) + 1.0
    active = active_roster_player_ids(data, cutoff_ts, 90.0)
    targets, _, _, selection = select_attack_lineups(data)
    target_players = {player_id for lineup in targets for player_id in lineup}
    baseline = rankings_on_universe(fit_rating_maps(data, cutoff_ts), active)
    rows: list[dict[str, Any]] = []
    for multiplier, copies in ((2, 1), (5, 4)):
        modified = duplicate_target_evidence(data, targets, copies=copies)
        attacked = rankings_on_universe(fit_rating_maps(modified, cutoff_ts), active)
        rows.extend(
            _target_change_rows(
                baseline,
                attacked,
                target_players,
                attack="observed evidence duplication",
                volume_multiplier=multiplier,
                target_lineups=len(targets),
            )
        )
        print(f"volume attack {multiplier}x", flush=True)
    return selection, pd.DataFrame(rows)


def adversarial_thinning_test(
    data: EvaluationData,
    *,
    hash_seeds: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Thin target-player history and refit all systems."""

    cutoff_ts = float(data.resolved["ts"].max()) + 1.0
    active = active_roster_player_ids(data, cutoff_ts, 90.0)
    targets, _, _, selection = select_attack_lineups(data)
    target_players = {player_id for lineup in targets for player_id in lineup}
    baseline_ratings = fit_rating_maps(data, cutoff_ts)
    baseline = rankings_on_universe(baseline_ratings, active)
    baseline_top = {
        method: set(
            baseline[baseline["method"] == method].head(100)["player_id"].astype(int)
        )
        for method in METHODS
    }
    incident = data.resolved[
        data.resolved.apply(
            lambda row: bool(
                set(_tuple_lineup(row["winners"]) + _tuple_lineup(row["losers"]))
                & target_players
            ),
            axis=1,
        )
    ][["tournament_id", "match_id"]]
    incident_keys = {
        (int(row.tournament_id), int(row.match_id))
        for row in incident.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for fraction in (0.10, 0.25, 0.50):
        for hash_seed in range(hash_seeds):
            removed = {
                key
                for key in incident_keys
                if int(
                    sha256(f"{hash_seed}:{key[0]}:{key[1]}".encode()).hexdigest()[:16],
                    16,
                )
                / float(16**16)
                < fraction
            }
            removed_match_ids = {match_id for _, match_id in removed}
            reduced = replace(
                data,
                matches=data.matches.filter(
                    ~pl.col("match_id").is_in(removed_match_ids)
                ),
                appearances=data.appearances.filter(
                    ~pl.col("match_id").is_in(removed_match_ids)
                ),
                resolved=data.resolved[
                    ~data.resolved["match_id"].isin(removed_match_ids)
                ].copy(),
            )
            attacked = rankings_on_universe(fit_rating_maps(reduced, cutoff_ts), active)
            changes = pd.DataFrame(
                _target_change_rows(
                    baseline,
                    attacked,
                    target_players,
                    attack="evidence thinning",
                    removal_fraction=fraction,
                    hash_seed=hash_seed,
                    removed_matches=len(removed_match_ids),
                )
            )
            for method in METHODS:
                attacked_method = attacked[attacked["method"] == method]
                retained = baseline_top[method] & set(
                    attacked_method.head(100)["player_id"].astype(int)
                )
                method_changes = changes[changes["method"] == method].copy()
                method_changes["full_top100_retention"] = len(retained) / 100.0
                rows.extend(method_changes.to_dict("records"))
            print(
                f"thinning {fraction:.0%} seed {hash_seed + 1}/{hash_seeds}",
                flush=True,
            )
    return selection, pd.DataFrame(rows)
