"""The substantive analyses for the Part I evaluation package."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
from scipy.special import expit
from scipy.stats import spearmanr

from rankings.algorithms import ExposureLogOddsEngine

from .core import (
    DEFAULT_CALIBRATION_START,
    DEFAULT_PLUS_DATE,
    DEFAULT_PREDICTION_START,
    DEFAULT_SEED,
    EvaluationData,
    active_roster_player_ids,
    activity_ratings,
    fit_team_bradley_terry,
    fit_temperature,
    loopr_rankings,
    macro_tier_auc,
    openskill_ratings,
    pair_weighted_stratified_auc,
    prior_teammate_familiarity,
    production_config,
    score_map,
    seed_everything,
    slice_for_cutoff,
    team_mean,
    teammate_pairs,
    top_n,
    tournament_cluster_bootstrap_difference,
    utc_date,
    utc_timestamp,
    winrate_90d_ratings,
)


METHOD_ORDER = [
    "LOOPR",
    "Bradley–Terry (L2)",
    "OpenSkill μ",
    "OpenSkill ordinal",
    "90-day win rate",
    "Chance (0.5)",
]


def _iqr(values: Sequence[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    return tuple(map(float, np.quantile(array, [0.25, 0.5, 0.75])))


def data_audit(data: EvaluationData) -> dict[str, Any]:
    resolved = data.resolved
    player_ids = {
        int(pid)
        for lineups in (resolved["winners"], resolved["losers"])
        for lineup in lineups
        for pid in lineup
    }
    sources = resolved["lineup_source"].value_counts().to_dict()
    timestamp_sources = resolved["timestamp_source"].value_counts().to_dict()
    output = {
        "canonical_raw_records": int(len(data.canonical_sources)),
        "eligible_tournaments": int(data.tournaments.height),
        "eligible_decided_matches": int(len(resolved)),
        "eligible_unique_players": int(len(player_ids)),
        "eligible_unique_roster_players": int(data.players["user_id"].n_unique()),
        "date_start": utc_date(float(resolved["ts"].min())),
        "date_end": utc_date(float(resolved["ts"].max())),
        "lineup_sources": {str(k): int(v) for k, v in sources.items()},
        "timestamp_sources": {str(k): int(v) for k, v in timestamp_sources.items()},
        "timestamp_imputed_matches": int(
            (resolved["timestamp_source"] == "tournament_start_imputed").sum()
        ),
        "appearance_any_matches": int(
            (resolved["lineup_source"] != "roster_fallback").sum()
        ),
        "appearance_both_matches": int(
            (resolved["lineup_source"] == "appearance_both").sum()
        ),
        "roster_fallback_matches": int(
            (resolved["lineup_source"] == "roster_fallback").sum()
        ),
        "winner_lineup_size": {
            "median": float(resolved["winners"].map(len).median()),
            "q1": float(resolved["winners"].map(len).quantile(0.25)),
            "q3": float(resolved["winners"].map(len).quantile(0.75)),
        },
        "loser_lineup_size": {
            "median": float(resolved["losers"].map(len).median()),
            "q1": float(resolved["losers"].map(len).quantile(0.25)),
            "q3": float(resolved["losers"].map(len).quantile(0.75)),
        },
        "duplicate_source_audit": data.duplicate_audit,
    }
    return output


def _plus_frame(path: Path) -> pd.DataFrame:
    votes = pd.read_csv(path)
    required = {"id", "pass_or_fail", "tier", "suggested"}
    missing = required - set(votes.columns)
    if missing:
        raise ValueError(f"Plus file missing columns: {sorted(missing)}")
    votes = votes.copy()
    votes["player_id"] = pd.to_numeric(votes["id"], errors="coerce").astype("Int64")
    votes["tier"] = pd.to_numeric(votes["tier"], errors="coerce").astype("Int64")
    votes["passed"] = (
        votes["pass_or_fail"].astype(str).str.lower().eq("pass").astype(int)
    )
    votes["suggested"] = (
        votes["suggested"].astype(str).str.lower().isin({"1", "true", "t", "yes"})
    )
    return votes.dropna(subset=["player_id", "tier"]).copy()


def _pair_auc(pass_scores: np.ndarray, fail_scores: np.ndarray) -> float:
    """AUC as the share of passer-failer pairs ordered correctly."""

    comparison = pass_scores[:, None] - fail_scores[None, :]
    return float(np.mean(comparison > 0) + 0.5 * np.mean(comparison == 0))


def _plus_bootstrap_intervals(
    frame: pd.DataFrame,
    score_columns: Mapping[str, str],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Efficient fixed-cell bootstrap for all Plus methods at once."""

    tiers = sorted(frame["tier"].astype(int).unique())
    statuses = [False, True]
    labels = [0, 1]
    cell_indices: dict[tuple[int, bool, int], np.ndarray] = {}
    for tier in tiers:
        for status in statuses:
            for label in labels:
                mask = (
                    frame["tier"].astype(int).eq(tier)
                    & frame["suggested"].astype(bool).eq(status)
                    & frame["passed"].astype(int).eq(label)
                )
                cell_indices[(tier, status, label)] = np.flatnonzero(mask.to_numpy())

    scores = frame[list(score_columns.values())].to_numpy(dtype=float)
    methods = list(score_columns)
    macro_draws = {method: np.empty(n_boot, dtype=float) for method in methods}
    strat_draws = {method: np.empty(n_boot, dtype=float) for method in methods}
    tier_draws = {
        method: {tier: np.full(n_boot, np.nan, dtype=float) for tier in tiers}
        for method in methods
    }
    cell_draws = {
        method: {
            (tier, status): np.full(n_boot, np.nan, dtype=float)
            for tier in tiers
            for status in statuses
        }
        for method in methods
    }
    rng = np.random.default_rng(seed)
    for draw in range(n_boot):
        sampled = {
            key: rng.choice(indices, size=len(indices), replace=True)
            for key, indices in cell_indices.items()
            if len(indices)
        }
        for method_index, method in enumerate(methods):
            tier_aucs = []
            weighted = 0.0
            total_pairs = 0
            for tier in tiers:
                pass_parts = [
                    scores[sampled[(tier, status, 1)], method_index]
                    for status in statuses
                    if (tier, status, 1) in sampled
                ]
                fail_parts = [
                    scores[sampled[(tier, status, 0)], method_index]
                    for status in statuses
                    if (tier, status, 0) in sampled
                ]
                if pass_parts and fail_parts:
                    tier_auc = _pair_auc(
                        np.concatenate(pass_parts), np.concatenate(fail_parts)
                    )
                    tier_aucs.append(tier_auc)
                    tier_draws[method][tier][draw] = tier_auc
                for status in statuses:
                    pass_key = (tier, status, 1)
                    fail_key = (tier, status, 0)
                    if pass_key not in sampled or fail_key not in sampled:
                        continue
                    pass_scores = scores[sampled[pass_key], method_index]
                    fail_scores = scores[sampled[fail_key], method_index]
                    pairs = len(pass_scores) * len(fail_scores)
                    cell_auc = _pair_auc(pass_scores, fail_scores)
                    cell_draws[method][(tier, status)][draw] = cell_auc
                    weighted += cell_auc * pairs
                    total_pairs += pairs
            macro_draws[method][draw] = float(np.mean(tier_aucs))
            strat_draws[method][draw] = weighted / total_pairs
    result = {
        method: {
            "macro": (
                float(np.quantile(macro_draws[method], 0.025)),
                float(np.quantile(macro_draws[method], 0.975)),
            ),
            "stratified": (
                float(np.quantile(strat_draws[method], 0.025)),
                float(np.quantile(strat_draws[method], 0.975)),
            ),
            "tiers": {
                str(tier): (
                    float(np.nanquantile(draws, 0.025)),
                    float(np.nanquantile(draws, 0.975)),
                )
                for tier, draws in tier_draws[method].items()
                if np.isfinite(draws).any()
            },
            "strata": {
                f"{tier}|{int(status)}": (
                    float(np.nanquantile(draws, 0.025)),
                    float(np.nanquantile(draws, 0.975)),
                )
                for (tier, status), draws in cell_draws[method].items()
                if np.isfinite(draws).any()
            },
        }
        for method in methods
    }
    loopr_macro = macro_draws["LOOPR"]
    loopr_strat = strat_draws["LOOPR"]
    for method in methods:
        macro_delta = loopr_macro - macro_draws[method]
        strat_delta = loopr_strat - strat_draws[method]
        result[method]["loopr_minus_method_macro"] = (
            float(np.quantile(macro_delta, 0.025)),
            float(np.quantile(macro_delta, 0.975)),
        )
        result[method]["loopr_minus_method_stratified"] = (
            float(np.quantile(strat_delta, 0.025)),
            float(np.quantile(strat_delta, 0.975)),
        )
    return result


def connectivity_and_mixing(
    data: EvaluationData,
    plus_path: Path,
    *,
    cutoff_ts: float | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Measure what the player-opponent graph can and cannot distinguish."""

    if cutoff_ts is None:
        cutoff_ts = float(data.resolved["ts"].max()) + 1.0
    frame = data.resolved[data.resolved["ts"] < cutoff_ts].copy()
    graph = nx.DiGraph()
    teammates: dict[int, set[int]] = defaultdict(set)
    opponents: dict[int, set[int]] = defaultdict(set)
    lineup_counts: dict[int, Counter[tuple[int, ...]]] = defaultdict(Counter)
    teammate_match_counts: dict[int, Counter[int]] = defaultdict(Counter)
    matches_played: Counter[int] = Counter()

    for row in frame.itertuples(index=False):
        winners = tuple(map(int, row.winners))
        losers = tuple(map(int, row.losers))
        graph.add_nodes_from(winners)
        graph.add_nodes_from(losers)
        graph.add_edges_from((loser, winner) for loser in losers for winner in winners)
        for lineup, other in ((winners, losers), (losers, winners)):
            lineup_key = tuple(sorted(lineup))
            for pid in lineup:
                matches_played[pid] += 1
                lineup_counts[pid][lineup_key] += 1
                mates = set(lineup) - {pid}
                teammates[pid].update(mates)
                teammate_match_counts[pid].update(mates)
                opponents[pid].update(other)

    components = sorted(nx.weakly_connected_components(graph), key=len, reverse=True)
    largest = components[0] if components else set()
    player_rows = []
    for pid in sorted(graph.nodes):
        count = matches_played[pid]
        most_common_lineup = lineup_counts[pid].most_common(1)
        max_lineup_share = (
            most_common_lineup[0][1] / count if most_common_lineup and count else np.nan
        )
        max_teammate_share = (
            max(teammate_match_counts[pid].values()) / count
            if teammate_match_counts[pid] and count
            else np.nan
        )
        player_rows.append(
            {
                "player_id": pid,
                "matches": count,
                "unique_teammates": len(teammates[pid]),
                "unique_opponents": len(opponents[pid]),
                "distinct_lineups": len(lineup_counts[pid]),
                "most_common_lineup_share": max_lineup_share,
                "most_common_teammate_share": max_teammate_share,
                "in_largest_component": pid in largest,
                "persistent_lineup_80pct": count >= 5 and max_lineup_share >= 0.80,
                "near_inseparable_teammate_90pct": count >= 5
                and max_teammate_share >= 0.90,
            }
        )
    players = pd.DataFrame(player_rows)

    rankings = loopr_rankings(data, cutoff_ts)
    active = active_roster_player_ids(data, cutoff_ts)
    top100 = top_n(rankings, active, 100)
    votes = _plus_frame(plus_path)

    teammate_q1, teammate_median, teammate_q3 = _iqr(players["unique_teammates"])
    opponent_q1, opponent_median, opponent_q3 = _iqr(players["unique_opponents"])
    lineup_q1, lineup_median, lineup_q3 = _iqr(players["distinct_lineups"])
    share_q1, share_median, share_q3 = _iqr(players["most_common_lineup_share"])
    meaningful = [len(component) for component in components if len(component) >= 4]
    summary = {
        "cutoff_date": utc_date(cutoff_ts),
        "eligible_players": int(len(players)),
        "largest_weak_component_n": int(len(largest)),
        "largest_weak_component_pct": float(len(largest) / len(players)),
        "top100_in_largest_n": int(top100["player_id"].isin(largest).sum()),
        "top100_in_largest_pct": float(top100["player_id"].isin(largest).mean()),
        "plus_candidates_in_graph_n": int(
            votes["player_id"].isin(set(graph.nodes)).sum()
        ),
        "plus_candidates_total": int(len(votes)),
        "plus_candidates_in_largest_n": int(votes["player_id"].isin(largest).sum()),
        "plus_candidates_in_largest_pct_all_candidates": float(
            votes["player_id"].isin(largest).mean()
        ),
        "weak_components": int(len(components)),
        "meaningful_components_ge4": int(len(meaningful)),
        "meaningful_component_sizes": meaningful,
        "unique_teammates": {
            "median": teammate_median,
            "q1": teammate_q1,
            "q3": teammate_q3,
        },
        "unique_opponents": {
            "median": opponent_median,
            "q1": opponent_q1,
            "q3": opponent_q3,
        },
        "distinct_lineups": {
            "median": lineup_median,
            "q1": lineup_q1,
            "q3": lineup_q3,
        },
        "most_common_lineup_share": {
            "median": share_median,
            "q1": share_q1,
            "q3": share_q3,
        },
        "persistent_lineup_80pct_n": int(players["persistent_lineup_80pct"].sum()),
        "persistent_lineup_80pct_pct": float(players["persistent_lineup_80pct"].mean()),
        "near_inseparable_teammate_90pct_n": int(
            players["near_inseparable_teammate_90pct"].sum()
        ),
        "near_inseparable_teammate_90pct_pct": float(
            players["near_inseparable_teammate_90pct"].mean()
        ),
        "persistent_threshold_denominator": "players with >=1 match; flags require >=5 matches",
    }
    component_rows = pd.DataFrame(
        [
            {
                "component_rank": i + 1,
                "size": len(component),
                "share": len(component) / len(players),
            }
            for i, component in enumerate(components)
        ]
    )
    return summary, players, component_rows


def _tournament_schedule(data: EvaluationData, start_ts: float) -> pd.DataFrame:
    tournaments = data.tournaments.select(
        ["tournament_id", "name", "start_time"]
    ).to_pandas()
    tournaments = tournaments.drop_duplicates("tournament_id")
    counts = data.resolved.groupby("tournament_id").size().rename("n_matches")
    tournaments = tournaments.join(counts, on="tournament_id", how="inner")
    tournaments = tournaments[tournaments["start_time"] >= start_ts]
    return tournaments.sort_values(["start_time", "tournament_id"]).reset_index(
        drop=True
    )


def walk_forward_prediction(
    data: EvaluationData,
    *,
    calibration_start: str = DEFAULT_CALIBRATION_START,
    report_start: str = DEFAULT_PREDICTION_START,
    regularization_c: float = 1.0,
    seed: int = DEFAULT_SEED,
    checkpoint_path: Path | None = None,
) -> pd.DataFrame:
    """Strict tournament walk-forward predictions with prior-only calibration."""

    seed_everything(seed)
    calibration_start_ts = utc_timestamp(calibration_start)
    report_start_ts = utc_timestamp(report_start)
    schedule = _tournament_schedule(data, calibration_start_ts)
    calibration_history: dict[str, list[tuple[int, float]]] = defaultdict(list)
    records: list[dict[str, Any]] = []

    completed: set[int] = set()
    if checkpoint_path is not None and checkpoint_path.exists():
        old = pd.read_parquet(checkpoint_path)
        records.extend(old.to_dict("records"))
        completed = set(old["tournament_id"].astype(int).unique())
        # A checkpoint can only be resumed safely when it includes warm-up rows.
        for method, group in old.groupby("method"):
            if "raw_difference" in group:
                calibration_history[method].extend(
                    zip(
                        group["tournament_id"].astype(int),
                        group["raw_difference"].astype(float),
                    )
                )

    for index, event in schedule.iterrows():
        tid = int(event["tournament_id"])
        if tid in completed:
            continue
        test = data.resolved[data.resolved["tournament_id"] == tid].copy()
        if test.empty:
            continue
        cutoff_ts = min(float(event["start_time"]), float(test["ts"].min()))
        train = data.resolved[
            (data.resolved["ts"] < cutoff_ts) & (data.resolved["tournament_id"] != tid)
        ].copy()
        if train["tournament_id"].nunique() < 50:
            continue

        match_override = data.matches.filter(pl.col("tournament_id") != tid)
        loopr = loopr_rankings(
            data,
            cutoff_ts,
            matches_override=match_override,
        )
        loopr_map = score_map(loopr, "score")

        os_frame, os_model = openskill_ratings(train)
        mu_map = score_map(os_frame, "mu") if not os_frame.empty else {}
        ordinal_map = score_map(os_frame, "ordinal") if not os_frame.empty else {}
        os_default = os_model.rating()
        mu_prior = float(os_default.mu)
        ordinal_prior = float(os_default.mu - 3.0 * os_default.sigma)

        winrate = winrate_90d_ratings(train, cutoff_ts)
        winrate_map = score_map(winrate, "winrate_logit") if not winrate.empty else {}
        bt_map, bt_meta = fit_team_bradley_terry(
            train, regularization_c=regularization_c, seed=seed
        )

        rating_specs: dict[str, tuple[Mapping[int, float], float]] = {
            "LOOPR": (loopr_map, 0.0),
            "Bradley–Terry (L2)": (bt_map, 0.0),
            "OpenSkill μ": (mu_map, mu_prior),
            "OpenSkill ordinal": (ordinal_map, ordinal_prior),
            "90-day win rate": (winrate_map, 0.0),
        }
        prior_pairs = teammate_pairs(train)
        winner_fam = prior_teammate_familiarity(test["winners"], prior_pairs)
        loser_fam = prior_teammate_familiarity(test["losers"], prior_pairs)
        familiarity = (np.asarray(winner_fam) + np.asarray(loser_fam)) / 2.0

        event_rows: list[dict[str, Any]] = []
        for method in METHOD_ORDER:
            prior_oof = [value for _, value in calibration_history[method]]
            prior_tournaments = {t for t, _ in calibration_history[method]}
            alpha = (
                fit_temperature(prior_oof)
                if len(prior_oof) >= 200 and len(prior_tournaments) >= 5
                else 1.0
            )
            if method == "Chance (0.5)":
                differences = np.zeros(len(test), dtype=float)
                alpha = 1.0
            else:
                mapping, prior = rating_specs[method]
                differences = np.array(
                    [
                        team_mean(row.winners, mapping, prior)
                        - team_mean(row.losers, mapping, prior)
                        for row in test.itertuples(index=False)
                    ],
                    dtype=float,
                )
            p_win = expit(alpha * differences)
            for position, (row, raw_diff, probability) in enumerate(
                zip(test.itertuples(index=False), differences, p_win)
            ):
                team1_won = int(row.team1_id) == int(row.winner_team_id)
                p_team1 = float(probability if team1_won else 1.0 - probability)
                fam = float(familiarity[position])
                event_rows.append(
                    {
                        "tournament_id": tid,
                        "tournament_name": event["name"],
                        "tournament_start_ts": float(event["start_time"]),
                        "cutoff_ts": cutoff_ts,
                        "match_id": int(row.match_id),
                        "method": method,
                        "alpha": float(alpha),
                        "raw_difference": float(raw_diff),
                        "p_actual_winner": float(probability),
                        "log_loss": float(-math.log(np.clip(probability, 1e-15, 1.0))),
                        "brier": float((1.0 - probability) ** 2),
                        "accuracy": float(raw_diff > 0.0)
                        if method != "Chance (0.5)"
                        else 0.5,
                        "p_team1": p_team1,
                        "team1_won": int(team1_won),
                        "lineup_familiarity": fam,
                        "familiarity_group": (
                            "mostly novel"
                            if fam < 0.25
                            else "mixed"
                            if fam < 0.75
                            else "familiar"
                        ),
                        "is_report_window": bool(
                            float(event["start_time"]) >= report_start_ts
                        ),
                        "bt_players": int(bt_meta.get("n_players", 0)),
                    }
                )
            calibration_history[method].extend((tid, float(d)) for d in differences)

        records.extend(event_rows)
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(records).to_parquet(checkpoint_path, index=False)
        print(
            f"walk-forward {index + 1}/{len(schedule)}: {tid} {event['name']} "
            f"({len(test)} matches, train={len(train)})",
            flush=True,
        )
    return pd.DataFrame(records)


def prediction_summary(
    predictions: pd.DataFrame,
    *,
    n_boot: int = 5_000,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = predictions[predictions["is_report_window"]].copy()
    rows = []
    tournament_method = frame.groupby(["method", "tournament_id"], as_index=False).agg(
        log_loss=("log_loss", "mean"),
        brier=("brier", "mean"),
        accuracy=("accuracy", "mean"),
        n_matches=("match_id", "nunique"),
    )
    for method in METHOD_ORDER:
        group = frame[frame["method"] == method]
        per_event = tournament_method[tournament_method["method"] == method]
        if group.empty:
            continue
        if method == "LOOPR":
            diff, diff_lo, diff_hi = 0.0, 0.0, 0.0
            tournaments_better = 0
            median_event_diff = 0.0
        else:
            diff, diff_lo, diff_hi = tournament_cluster_bootstrap_difference(
                frame, method, "LOOPR", n_boot=n_boot, seed=seed
            )
            wide = tournament_method.pivot(
                index="tournament_id", columns="method", values="log_loss"
            ).dropna(subset=[method, "LOOPR"])
            event_diff = wide[method] - wide["LOOPR"]
            tournaments_better = int((event_diff < 0).sum())
            median_event_diff = float(event_diff.median())
        rows.append(
            {
                "method": method,
                "matches": int(group["match_id"].nunique()),
                "tournaments": int(group["tournament_id"].nunique()),
                "match_micro_log_loss": float(group["log_loss"].mean()),
                "tournament_macro_log_loss": float(per_event["log_loss"].mean()),
                "brier_score": float(group["brier"].mean()),
                "accuracy": float(group["accuracy"].mean()),
                "paired_log_loss_difference_vs_loopr": diff,
                "paired_difference_ci_low": diff_lo,
                "paired_difference_ci_high": diff_hi,
                "tournaments_outperforming_loopr": tournaments_better,
                "median_per_tournament_difference": median_event_diff,
            }
        )
    summary = pd.DataFrame(rows)

    loopr = frame[frame["method"] == "LOOPR"].copy()
    loopr["calibration_bin"] = pd.qcut(
        loopr["p_team1"], q=5, labels=False, duplicates="drop"
    )
    calibration = loopr.groupby("calibration_bin", as_index=False).agg(
        mean_predicted_probability=("p_team1", "mean"),
        observed_win_rate=("team1_won", "mean"),
        n_matches=("match_id", "count"),
        min_probability=("p_team1", "min"),
        max_probability=("p_team1", "max"),
    )

    per_tournament = tournament_method.pivot(
        index="tournament_id", columns="method", values="log_loss"
    ).reset_index()
    familiarity = frame.groupby(["method", "familiarity_group"], as_index=False).agg(
        matches=("match_id", "count"),
        tournaments=("tournament_id", "nunique"),
        log_loss=("log_loss", "mean"),
        brier=("brier", "mean"),
        accuracy=("accuracy", "mean"),
        median_familiarity=("lineup_familiarity", "median"),
    )
    return summary, calibration, per_tournament, familiarity


def plus_analysis(
    data: EvaluationData,
    votes_path: Path,
    *,
    vote_date: str = DEFAULT_PLUS_DATE,
    active_days: float = 90.0,
    n_boot: int = 2_000,
    seed: int = DEFAULT_SEED,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cutoff_ts = utc_timestamp(vote_date)
    votes = _plus_frame(votes_path)
    history = data.resolved[data.resolved["ts"] < cutoff_ts].copy()
    active = active_roster_player_ids(data, cutoff_ts, active_days)

    loopr = loopr_rankings(data, cutoff_ts)
    os_frame, _ = openskill_ratings(history)
    winrate = winrate_90d_ratings(history, cutoff_ts)
    activity = activity_ratings(history, cutoff_ts)
    ratings = (
        loopr[["player_id", "score"]]
        .merge(os_frame, on="player_id", how="outer")
        .merge(winrate, on="player_id", how="outer")
        .merge(activity, on="player_id", how="outer", suffixes=("", "_activity"))
    )
    merged = votes.merge(ratings, on="player_id", how="left")
    merged["active_90d"] = merged["player_id"].astype(int).isin(active)

    score_columns = {
        "LOOPR": "score",
        "OpenSkill μ": "mu",
        "OpenSkill ordinal": "ordinal",
        "90-day win rate": "winrate_90d",
        "Matches in prior 90d": "matches_90d_activity",
        "Tournaments in prior 90d": "tournaments_90d",
    }
    native_coverage = {
        method: int(merged.loc[merged["active_90d"], column].notna().sum())
        for method, column in score_columns.items()
    }
    # Keep every roster-active candidate and apply the same population-neutral
    # unseen-player rules used in prediction.  This prevents optional appearance
    # coverage from silently choosing the evaluated candidates.
    merged["score"] = merged["score"].fillna(0.0)
    merged["mu"] = merged["mu"].fillna(25.0)
    merged["ordinal"] = merged["ordinal"].fillna(0.0)
    merged["winrate_90d"] = merged["winrate_90d"].fillna(0.5)
    merged["matches_90d_activity"] = merged["matches_90d_activity"].fillna(0.0)
    merged["tournaments_90d"] = merged["tournaments_90d"].fillna(0.0)
    common = merged[merged["active_90d"]].copy()

    intervals = _plus_bootstrap_intervals(
        common, score_columns, n_boot=n_boot, seed=seed
    )

    method_rows: list[dict[str, Any]] = []
    stratum_rows: list[dict[str, Any]] = []
    for method, column in score_columns.items():
        macro, tier_rows = macro_tier_auc(common, column)
        stratified, pair_count, strata = pair_weighted_stratified_auc(common, column)
        macro_ci = intervals[method]["macro"]
        strat_ci = intervals[method]["stratified"]
        method_rows.append(
            {
                "method": method,
                "n": len(common),
                "institutional_macro_auc": macro,
                "institutional_ci_low": macro_ci[0],
                "institutional_ci_high": macro_ci[1],
                "like_for_like_stratified_auc": stratified,
                "stratified_ci_low": strat_ci[0],
                "stratified_ci_high": strat_ci[1],
                "eligible_passer_failer_pairs": pair_count,
                "loopr_minus_method_macro_auc": (
                    macro_tier_auc(common, "score")[0] - macro
                ),
                "loopr_minus_method_macro_ci_low": intervals[method][
                    "loopr_minus_method_macro"
                ][0],
                "loopr_minus_method_macro_ci_high": intervals[method][
                    "loopr_minus_method_macro"
                ][1],
                "loopr_minus_method_stratified_auc": (
                    pair_weighted_stratified_auc(common, "score")[0] - stratified
                ),
                "loopr_minus_method_stratified_ci_low": intervals[method][
                    "loopr_minus_method_stratified"
                ][0],
                "loopr_minus_method_stratified_ci_high": intervals[method][
                    "loopr_minus_method_stratified"
                ][1],
            }
        )
        for row in tier_rows:
            ci = intervals[method]["tiers"].get(str(int(row["tier"])), (None, None))
            stratum_rows.append(
                {
                    "method": method,
                    "comparison": "tier",
                    **row,
                    "ci_low": ci[0],
                    "ci_high": ci[1],
                }
            )
        for row in strata:
            key = f"{int(row['tier'])}|{int(bool(row['suggested']))}"
            ci = intervals[method]["strata"].get(key, (None, None))
            stratum_rows.append(
                {
                    "method": method,
                    "comparison": "tier_status",
                    **row,
                    "ci_low": ci[0],
                    "ci_high": ci[1],
                }
            )

    summary = {
        "vote_date": vote_date,
        "raw_candidates": int(len(votes)),
        "pass": int(votes["passed"].sum()),
        "fail": int((1 - votes["passed"]).sum()),
        "suggested": int(votes["suggested"].sum()),
        "incumbent": int((~votes["suggested"]).sum()),
        "active_90d_candidates": int(merged["active_90d"].sum()),
        "common_evaluation_candidates": int(len(common)),
        "coverage_of_all_candidates": float(len(common) / len(votes)),
        "coverage_of_active_candidates": float(
            len(common) / max(1, int(merged["active_90d"].sum()))
        ),
        "native_score_coverage_before_neutral_fill": native_coverage,
        "unseen_candidate_treatment": {
            "LOOPR": 0.0,
            "OpenSkill_mu": 25.0,
            "OpenSkill_ordinal": 0.0,
            "winrate_90d": 0.5,
            "matches_or_tournaments_90d": 0.0,
        },
        "automatic_sendouq_admissions": "excluded by source: CSV is derived from PlusVotingResult peer-vote rows",
        "vote_percentage_available": False,
        "historical_rule": "pass at >=60% approval (mean +/-1 score >=0.2)",
    }
    return summary, pd.DataFrame(method_rows), pd.DataFrame(stratum_rows), common


def outcome_shuffle_null(
    data: EvaluationData,
    votes_common: pd.DataFrame,
    *,
    vote_date: str = DEFAULT_PLUS_DATE,
    repetitions: int = 200,
    seed: int = DEFAULT_SEED,
    checkpoint_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Flip each historical result with p=.5 while preserving participation."""

    cutoff_ts = utc_timestamp(vote_date)
    matches, _, _ = slice_for_cutoff(data, cutoff_ts)
    observed_macro = macro_tier_auc(votes_common, "score")[0]
    observed_stratified = pair_weighted_stratified_auc(votes_common, "score")[0]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    completed: set[int] = set()
    if checkpoint_path is not None and checkpoint_path.exists():
        old = pd.read_csv(checkpoint_path)
        rows.extend(old.to_dict("records"))
        completed = set(old["repetition"].astype(int))

    # Advance the generator consistently when resuming.
    flips_by_rep = rng.integers(0, 2, size=(repetitions, matches.height), dtype=np.int8)
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
        rankings = loopr_rankings(
            data,
            cutoff_ts,
            matches_override=shuffled,
        )
        merged = votes_common.drop(columns=["score"], errors="ignore").merge(
            rankings[["player_id", "score"]], on="player_id", how="left"
        )
        macro = macro_tier_auc(merged.dropna(subset=["score"]), "score")[0]
        stratified = pair_weighted_stratified_auc(
            merged.dropna(subset=["score"]), "score"
        )[0]
        rows.append(
            {
                "repetition": repetition,
                "macro_auc": macro,
                "stratified_auc": stratified,
                "flipped_matches": int(flip.sum()),
            }
        )
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).sort_values("repetition").to_csv(
                checkpoint_path, index=False
            )
        if repetition % 10 == 0 or repetition == repetitions - 1:
            print(f"outcome shuffle {repetition + 1}/{repetitions}", flush=True)

    result = pd.DataFrame(rows).sort_values("repetition").reset_index(drop=True)
    summary = {
        "repetitions": repetitions,
        "seed": seed,
        "null_procedure": "independent fair-coin result reversal for every pre-vote match",
        "observed_macro_auc": observed_macro,
        "null_macro_mean": float(result["macro_auc"].mean()),
        "null_macro_q025": float(result["macro_auc"].quantile(0.025)),
        "null_macro_q975": float(result["macro_auc"].quantile(0.975)),
        "null_macro_exceedance_p": float(
            (1 + (result["macro_auc"] >= observed_macro).sum()) / (1 + len(result))
        ),
        "observed_stratified_auc": observed_stratified,
        "null_stratified_mean": float(result["stratified_auc"].mean()),
        "null_stratified_q025": float(result["stratified_auc"].quantile(0.025)),
        "null_stratified_q975": float(result["stratified_auc"].quantile(0.975)),
        "null_stratified_exceedance_p": float(
            (1 + (result["stratified_auc"] >= observed_stratified).sum())
            / (1 + len(result))
        ),
    }
    return result, summary


def month_end_cutoffs(data: EvaluationData) -> list[float]:
    start = pd.Timestamp(data.resolved["ts"].min(), unit="s", tz="UTC")
    end = pd.Timestamp(data.resolved["ts"].max(), unit="s", tz="UTC")
    months = pd.date_range(
        start=start.normalize(), end=end.normalize(), freq="M", tz="UTC"
    )
    cutoffs = [float((month + pd.Timedelta(days=1)).timestamp()) for month in months]
    if not cutoffs or cutoffs[-1] <= float(data.resolved["ts"].max()):
        cutoffs.append(float(data.resolved["ts"].max()) + 1.0)
    return cutoffs


def stability_analysis(
    data: EvaluationData,
    *,
    half_life_days: float | None = 180.0,
    active_days: float = 90.0,
    minimum_date: str = "2025-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    snapshots: list[pd.DataFrame] = []
    for cutoff in month_end_cutoffs(data):
        if cutoff < utc_timestamp(minimum_date):
            continue
        rankings = loopr_rankings(data, cutoff, half_life_days=half_life_days)
        active = active_roster_player_ids(data, cutoff, active_days)
        top = top_n(rankings, active, 100)
        top["cutoff_ts"] = cutoff
        top["snapshot_date"] = utc_date(cutoff - 1.0)
        snapshots.append(top)
        print(f"stability snapshot {utc_date(cutoff - 1.0)}", flush=True)
    all_snapshots = pd.concat(snapshots, ignore_index=True)
    transition_rows = []
    dates = list(dict.fromkeys(all_snapshots["snapshot_date"]))
    for prior_date, next_date in zip(dates[:-1], dates[1:]):
        prior = all_snapshots[all_snapshots["snapshot_date"] == prior_date]
        following = all_snapshots[all_snapshots["snapshot_date"] == next_date]
        merged = prior[["player_id", "rank"]].merge(
            following[["player_id", "rank"]],
            on="player_id",
            suffixes=("_a", "_b"),
        )
        movement = (merged["rank_a"] - merged["rank_b"]).abs()
        rho = spearmanr(merged["rank_a"], merged["rank_b"]).statistic
        transition_rows.append(
            {
                "month_a": prior_date,
                "month_b": next_date,
                "top100_retained": int(len(merged)),
                "top100_retention": float(len(merged) / 100.0),
                "spearman_on_retained": float(rho),
                "median_absolute_rank_movement": float(movement.median()),
                "p90_absolute_rank_movement": float(movement.quantile(0.90)),
            }
        )
    return pd.DataFrame(transition_rows), all_snapshots


def leave_one_tournament_out(
    data: EvaluationData,
    *,
    cutoff_ts: float | None = None,
    active_days: float = 90.0,
    checkpoint_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if cutoff_ts is None:
        cutoff_ts = float(data.resolved["ts"].max()) + 1.0
    full = loopr_rankings(data, cutoff_ts)
    full_active = active_roster_player_ids(data, cutoff_ts, active_days)
    full_top = top_n(full, full_active, 100)
    full_rank = dict(
        zip(full_top["player_id"].astype(int), full_top["rank"].astype(int))
    )
    exposure = score_map(full.rename(columns={"exposure": "value"}), "value")
    full_match_counts: Counter[int] = Counter()
    for row in data.resolved[data.resolved["ts"] < cutoff_ts].itertuples(index=False):
        for player_id in tuple(row.winners) + tuple(row.losers):
            full_match_counts[int(player_id)] += 1

    tournament_names = dict(
        zip(
            data.tournaments["tournament_id"].to_list(),
            data.tournaments["name"].to_list(),
        )
    )
    tids = sorted(
        int(tid)
        for tid in data.resolved.loc[
            data.resolved["ts"] < cutoff_ts, "tournament_id"
        ].unique()
    )
    rows: list[dict[str, Any]] = []
    completed: set[int] = set()
    if checkpoint_path is not None and checkpoint_path.exists():
        old = pd.read_csv(checkpoint_path)
        rows.extend(old.to_dict("records"))
        completed = set(old["removed_tournament_id"].astype(int))

    for index, tid in enumerate(tids):
        if tid in completed:
            continue
        matches_override = data.matches.filter(pl.col("tournament_id") != tid)
        reduced = loopr_rankings(
            data,
            cutoff_ts,
            matches_override=matches_override,
        )
        reduced_active = active_roster_player_ids(
            data, cutoff_ts, active_days, exclude_tournament=tid
        )
        reduced_ranked = (
            reduced[reduced["player_id"].isin(reduced_active)]
            .sort_values(["score", "player_id"], ascending=[False, True])
            .reset_index(drop=True)
        )
        reduced_ranked["rank"] = np.arange(1, len(reduced_ranked) + 1)
        reduced_rank = dict(
            zip(
                reduced_ranked["player_id"].astype(int),
                reduced_ranked["rank"].astype(int),
            )
        )
        reduced_top_ids = set(reduced_ranked.head(100)["player_id"].astype(int))
        retained = set(full_rank) & reduced_top_ids
        still_ranked = [pid for pid in full_rank if pid in reduced_rank]
        no_longer_rank_eligible = [pid for pid in full_rank if pid not in reduced_rank]
        movement = np.array(
            [abs(full_rank[pid] - reduced_rank[pid]) for pid in still_ranked],
            dtype=float,
        )
        most_affected_pid = (
            max(
                still_ranked,
                key=lambda pid: abs(full_rank[pid] - reduced_rank[pid]),
            )
            if still_ranked
            else None
        )
        largest = (
            float(abs(full_rank[most_affected_pid] - reduced_rank[most_affected_pid]))
            if most_affected_pid is not None
            else None
        )
        ineligible_pid = (
            min(no_longer_rank_eligible, key=lambda pid: full_rank[pid])
            if no_longer_rank_eligible
            else None
        )
        rows.append(
            {
                "removed_tournament_id": tid,
                "removed_tournament_name": tournament_names.get(tid),
                "top100_retained": len(retained),
                "top100_retention": len(retained) / 100.0,
                "original_top100_still_ranked": len(still_ranked),
                "original_top100_no_longer_rank_eligible": len(no_longer_rank_eligible),
                "median_abs_rank_change_original_top100": float(np.median(movement))
                if len(movement)
                else None,
                "p90_abs_rank_change_original_top100": float(
                    np.quantile(movement, 0.90)
                )
                if len(movement)
                else None,
                "p95_abs_rank_change_original_top100": float(
                    np.quantile(movement, 0.95)
                )
                if len(movement)
                else None,
                "largest_abs_rank_change_original_top100": largest,
                "most_affected_player_id": most_affected_pid,
                "most_affected_full_rank": full_rank.get(most_affected_pid),
                "most_affected_reduced_rank": reduced_rank.get(most_affected_pid),
                "most_affected_full_exposure": exposure.get(most_affected_pid),
                "most_affected_full_matches": full_match_counts.get(most_affected_pid),
                "highest_ranked_newly_ineligible_player_id": ineligible_pid,
                "highest_ranked_newly_ineligible_full_rank": full_rank.get(
                    ineligible_pid
                ),
                "highest_ranked_newly_ineligible_full_exposure": exposure.get(
                    ineligible_pid
                ),
                "highest_ranked_newly_ineligible_full_matches": full_match_counts.get(
                    ineligible_pid
                ),
            }
        )
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
        if index % 10 == 0 or index == len(tids) - 1:
            print(f"LOTO {index + 1}/{len(tids)}", flush=True)

    result = pd.DataFrame(rows)
    worst = result.sort_values(
        ["top100_retention", "p95_abs_rank_change_original_top100"],
        ascending=[True, False],
    ).iloc[0]
    top100_match_q25 = float(
        np.quantile([full_match_counts[pid] for pid in full_rank], 0.25)
    )
    top100_exposure_q25 = float(np.quantile([exposure[pid] for pid in full_rank], 0.25))
    eligibility_events = result[result["original_top100_no_longer_rank_eligible"] > 0]
    largest_shared = result.loc[
        result["largest_abs_rank_change_original_top100"].idxmax()
    ]
    shared_movements = []
    for _, row in result.iterrows():
        # The per-event p90 is retained for the table; this aggregate answers
        # how large the upper tail usually was across removal experiments.
        shared_movements.append(float(row["p90_abs_rank_change_original_top100"]))
    summary = {
        "cutoff_date": utc_date(cutoff_ts),
        "eligible_tournaments_removed": int(len(result)),
        "median_top100_retention": float(result["top100_retention"].median()),
        "worst_top100_retention": float(result["top100_retention"].min()),
        "median_of_median_abs_rank_changes": float(
            result["median_abs_rank_change_original_top100"].median()
        ),
        "p90_of_all_tournament_p90_rank_changes": float(
            np.quantile(shared_movements, 0.90)
        ),
        "largest_observed_shared_player_rank_shift": float(
            result["largest_abs_rank_change_original_top100"].max()
        ),
        "worst_retention_tournament_id": int(worst["removed_tournament_id"]),
        "worst_retention_tournament_name": worst["removed_tournament_name"],
        "largest_shared_shift_tournament_id": int(
            largest_shared["removed_tournament_id"]
        ),
        "largest_shared_shift_tournament_name": largest_shared[
            "removed_tournament_name"
        ],
        "largest_shared_shift_player_id": int(
            largest_shared["most_affected_player_id"]
        ),
        "largest_shared_shift_player_full_matches": int(
            largest_shared["most_affected_full_matches"]
        ),
        "largest_shared_shift_player_low_match_exposure": bool(
            float(largest_shared["most_affected_full_matches"]) <= top100_match_q25
        ),
        "top100_match_count_q25": top100_match_q25,
        "top100_exposure_q25": top100_exposure_q25,
        "events_changing_rank_eligibility_for_at_least_one_original_top100_player": int(
            len(eligibility_events)
        ),
        "maximum_original_top100_players_made_rank_ineligible": int(
            result["original_top100_no_longer_rank_eligible"].max()
        ),
        "rank_eligibility_is_not_rank_movement": "players absent from the reduced active/ranked population are counted separately, not assigned a fictitious bottom rank; absence can reflect the 90-day active-player rule rather than disappearance from the rating graph",
    }
    return result, summary


def half_life_tradeoff(
    data: EvaluationData,
    base_predictions: pd.DataFrame,
    votes_common: pd.DataFrame,
    *,
    vote_date: str = DEFAULT_PLUS_DATE,
    settings: Sequence[float | None] = (90.0, 120.0, 180.0, 270.0, 365.0, None),
    seed: int = DEFAULT_SEED,
    checkpoint_path: Path | None = None,
) -> pd.DataFrame:
    """LOOPR-only half-life sweep using the same report tournaments."""

    report_events = (
        base_predictions[base_predictions["is_report_window"]][
            ["tournament_id", "tournament_start_ts"]
        ]
        .drop_duplicates()
        .sort_values(["tournament_start_ts", "tournament_id"])
    )
    calibration_events = (
        base_predictions[["tournament_id", "tournament_start_ts"]]
        .drop_duplicates()
        .sort_values(["tournament_start_ts", "tournament_id"])
    )
    report_ids = set(report_events["tournament_id"].astype(int))
    # Use every calibration-warmup event retained by the unified walk-forward
    # harness.  Truncating this history changes the fitted temperature and makes
    # a nominally identical 180-day setting disagree with the primary result.
    calibration_events = calibration_events.sort_values(
        ["tournament_start_ts", "tournament_id"]
    ).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    completed: set[str] = set()
    if checkpoint_path is not None and checkpoint_path.exists():
        old = pd.read_parquet(checkpoint_path)
        rows.extend(old.to_dict("records"))
        completed = set(old["setting"].astype(str))

    for setting in settings:
        label = "no_decay" if setting is None else f"{int(setting)}d"
        if label in completed:
            continue
        history: list[float] = []
        predictions: list[dict[str, Any]] = []
        for index, event in calibration_events.iterrows():
            tid = int(event["tournament_id"])
            test = data.resolved[data.resolved["tournament_id"] == tid]
            cutoff = min(float(event["tournament_start_ts"]), float(test["ts"].min()))
            rankings = loopr_rankings(
                data,
                cutoff,
                half_life_days=setting,
                matches_override=data.matches.filter(pl.col("tournament_id") != tid),
            )
            mapping = score_map(rankings, "score")
            differences = np.array(
                [
                    team_mean(row.winners, mapping, 0.0)
                    - team_mean(row.losers, mapping, 0.0)
                    for row in test.itertuples(index=False)
                ]
            )
            alpha = fit_temperature(history) if len(history) >= 200 else 1.0
            probabilities = expit(alpha * differences)
            if tid in report_ids:
                predictions.extend(
                    {
                        "tournament_id": tid,
                        "match_id": int(match_id),
                        "log_loss": float(-math.log(np.clip(p, 1e-15, 1.0))),
                    }
                    for match_id, p in zip(test["match_id"], probabilities)
                )
            history.extend(map(float, differences))
            if index % 20 == 0:
                print(
                    f"half-life {label}: {index + 1}/{len(calibration_events)}",
                    flush=True,
                )

        pred = pd.DataFrame(predictions)
        per_event = pred.groupby("tournament_id")["log_loss"].mean()
        transitions, _ = stability_analysis(
            data, half_life_days=setting, minimum_date="2025-07-31"
        )
        plus_rank = loopr_rankings(
            data, utc_timestamp(vote_date), half_life_days=setting
        )
        plus = (
            votes_common.drop(columns=["score"], errors="ignore")
            .merge(plus_rank[["player_id", "score"]], on="player_id", how="left")
            .dropna(subset=["score"])
        )
        macro = macro_tier_auc(plus, "score")[0]
        strat = pair_weighted_stratified_auc(plus, "score")[0]
        rows.append(
            {
                "setting": label,
                "half_life_days": setting,
                "match_micro_log_loss": float(pred["log_loss"].mean()),
                "tournament_macro_log_loss": float(per_event.mean()),
                "mean_monthly_top100_retention": float(
                    transitions["top100_retention"].mean()
                ),
                "median_monthly_rank_movement": float(
                    transitions["median_absolute_rank_movement"].median()
                ),
                "plus_institutional_macro_auc_secondary": macro,
                "plus_like_for_like_auc_secondary": strat,
            }
        )
        if checkpoint_path is not None:
            pd.DataFrame(rows).to_parquet(checkpoint_path, index=False)
    return pd.DataFrame(rows)


def property_tests(
    data: EvaluationData,
    *,
    cutoff_ts: float | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Small real-data tests for schedule strength and residual volume effects."""

    if cutoff_ts is None:
        cutoff_ts = float(data.resolved["ts"].max()) + 1.0
    frame = data.resolved[data.resolved["ts"] < cutoff_ts]
    rankings = loopr_rankings(data, cutoff_ts)
    rating = score_map(rankings, "score")
    wins: Counter[int] = Counter()
    losses: Counter[int] = Counter()
    opponents: dict[int, list[int]] = defaultdict(list)
    for row in frame.itertuples(index=False):
        for pid in row.winners:
            pid = int(pid)
            wins[pid] += 1
            opponents[pid].extend(map(int, row.losers))
        for pid in row.losers:
            pid = int(pid)
            losses[pid] += 1
            opponents[pid].extend(map(int, row.winners))
    rows = []
    for pid in sorted(set(wins) | set(losses)):
        n = wins[pid] + losses[pid]
        opp_scores = [rating.get(opp, 0.0) for opp in opponents[pid]]
        rows.append(
            {
                "player_id": pid,
                "matches": n,
                "win_rate": wins[pid] / n if n else np.nan,
                "average_opponent_loopr": float(np.mean(opp_scores))
                if opp_scores
                else np.nan,
                "loopr_score": rating.get(pid),
            }
        )
    players = pd.DataFrame(rows).dropna()
    eligible = players[players["matches"] >= 10].copy()

    # Real-player matched schedule comparisons: similar record and volume, but
    # materially different opponent strength.
    schedule_pairs = []
    ordered = eligible.sort_values("average_opponent_loopr").reset_index(drop=True)
    for i, low in ordered.iterrows():
        candidates = ordered[
            (ordered["player_id"] != low["player_id"])
            & ((ordered["win_rate"] - low["win_rate"]).abs() <= 0.03)
            & ((np.log1p(ordered["matches"]) - np.log1p(low["matches"])).abs() <= 0.25)
            & (
                ordered["average_opponent_loopr"]
                >= low["average_opponent_loopr"] + 0.25
            )
        ]
        if candidates.empty:
            continue
        high = candidates.iloc[0]
        schedule_pairs.append(
            {
                "lower_schedule_player": int(low["player_id"]),
                "stronger_schedule_player": int(high["player_id"]),
                "win_rate_difference": float(high["win_rate"] - low["win_rate"]),
                "match_log_difference": float(
                    np.log1p(high["matches"]) - np.log1p(low["matches"])
                ),
                "opponent_score_difference": float(
                    high["average_opponent_loopr"] - low["average_opponent_loopr"]
                ),
                "loopr_score_difference": float(
                    high["loopr_score"] - low["loopr_score"]
                ),
                "intended_direction": bool(high["loopr_score"] > low["loopr_score"]),
            }
        )
    pair_frame = pd.DataFrame(schedule_pairs)

    # Residual volume coefficient after flexible controls for observed record
    # and opponent strength.  Standardization makes the coefficient readable.
    design = eligible[["win_rate", "average_opponent_loopr", "matches"]].copy()
    design["win_rate_sq"] = design["win_rate"] ** 2
    design["opponent_sq"] = design["average_opponent_loopr"] ** 2
    design["log_matches"] = np.log1p(design["matches"])
    columns = [
        "win_rate",
        "win_rate_sq",
        "average_opponent_loopr",
        "opponent_sq",
        "log_matches",
    ]
    x = design[columns].to_numpy(dtype=float)
    x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=0)
    x = np.c_[np.ones(len(x)), x]
    y = eligible["loopr_score"].to_numpy(dtype=float)
    y_std = (y - y.mean()) / y.std(ddof=0)
    coef, *_ = np.linalg.lstsq(x, y_std, rcond=None)
    residual_full = y_std - x @ coef
    x_reduced = x[:, :-1]
    coef_reduced, *_ = np.linalg.lstsq(x_reduced, y_std, rcond=None)
    residual_reduced = y_std - x_reduced @ coef_reduced
    partial_r2 = 1.0 - float(np.sum(residual_full**2) / np.sum(residual_reduced**2))

    summary = {
        "strength_of_schedule": {
            "matched_pairs": int(len(pair_frame)),
            "share_rewarded_in_intended_direction": float(
                pair_frame["intended_direction"].mean()
            )
            if len(pair_frame)
            else None,
            "median_score_advantage": float(
                pair_frame["loopr_score_difference"].median()
            )
            if len(pair_frame)
            else None,
            "matching_rule": "|win-rate difference| <= .03, |log-match difference| <= .25, opponent-score gap >= .25",
        },
        "volume_resistance": {
            "players": int(len(eligible)),
            "standardized_log_match_coefficient": float(coef[-1]),
            "partial_r_squared_log_matches": partial_r2,
            "controls": "win rate (linear and squared), mean opponent LOOPR (linear and squared)",
            "interpretation_limit": "observational residual association, not proof of causal or perfect volume cancellation",
        },
    }
    return summary, pair_frame


def smoothing_perturbation_test(
    data: EvaluationData,
    *,
    cutoff_ts: float | None = None,
    players_per_band: int = 6,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove one recent match for players across exposure bands."""

    if cutoff_ts is None:
        cutoff_ts = float(data.resolved["ts"].max()) + 1.0
    baselines = {
        True: loopr_rankings(data, cutoff_ts, smoothing=True),
        False: loopr_rankings(data, cutoff_ts, smoothing=False),
    }
    smooth_base = baselines[True].copy()
    match_counts: Counter[int] = Counter()
    player_matches: dict[int, list[tuple[float, int]]] = defaultdict(list)
    history = data.resolved[data.resolved["ts"] < cutoff_ts]
    for row in history.itertuples(index=False):
        for pid in tuple(row.winners) + tuple(row.losers):
            pid = int(pid)
            match_counts[pid] += 1
            player_matches[pid].append((float(row.ts), int(row.match_id)))
    smooth_base["matches"] = smooth_base["player_id"].map(match_counts).fillna(0)
    eligible = smooth_base[smooth_base["matches"] >= 2].copy()
    eligible["exposure_band"] = pd.qcut(
        eligible["exposure"],
        q=4,
        labels=["low", "medium-low", "medium-high", "high"],
        duplicates="drop",
    )
    rng = np.random.default_rng(seed)
    selected: list[tuple[int, str]] = []
    for band, group in eligible.groupby("exposure_band", observed=True):
        indices = rng.choice(
            group.index.to_numpy(),
            size=min(players_per_band, len(group)),
            replace=False,
        )
        selected.extend(
            (int(eligible.loc[index, "player_id"]), str(band)) for index in indices
        )

    baseline_maps = {
        smoothing: score_map(frame, "score") for smoothing, frame in baselines.items()
    }
    baseline_ranks = {
        smoothing: dict(
            zip(
                frame.sort_values(["score", "player_id"], ascending=[False, True])[
                    "player_id"
                ].astype(int),
                np.arange(1, len(frame) + 1),
            )
        )
        for smoothing, frame in baselines.items()
    }
    rows: list[dict[str, Any]] = []
    for index, (pid, band) in enumerate(selected):
        recent_match = max(player_matches[pid])[1]
        override = data.matches.filter(pl.col("match_id") != recent_match)
        for smoothing in [True, False]:
            perturbed = loopr_rankings(
                data,
                cutoff_ts,
                smoothing=smoothing,
                matches_override=override,
            )
            perturbed_map = score_map(perturbed, "score")
            ordered = perturbed.sort_values(
                ["score", "player_id"], ascending=[False, True]
            )
            perturbed_rank = dict(
                zip(ordered["player_id"].astype(int), np.arange(1, len(ordered) + 1))
            )
            score_before = baseline_maps[smoothing].get(pid, 0.0)
            score_after = perturbed_map.get(pid, 0.0)
            rank_before = baseline_ranks[smoothing].get(
                pid, len(baselines[smoothing]) + 1
            )
            rank_after = perturbed_rank.get(pid, len(perturbed) + 1)
            rows.append(
                {
                    "player_id": pid,
                    "exposure_band": band,
                    "full_exposure": float(
                        smooth_base.loc[
                            smooth_base["player_id"] == pid, "exposure"
                        ].iloc[0]
                    ),
                    "full_matches": match_counts[pid],
                    "removed_match_id": recent_match,
                    "smoothing": smoothing,
                    "score_before": score_before,
                    "score_after": score_after,
                    "absolute_score_change": abs(score_after - score_before),
                    "rank_before": rank_before,
                    "rank_after": rank_after,
                    "absolute_rank_change": abs(rank_after - rank_before),
                    "player_disappeared": pid not in perturbed_map,
                }
            )
        print(
            f"smoothing perturbation {index + 1}/{len(selected)}: player {pid}",
            flush=True,
        )
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["exposure_band", "smoothing"], as_index=False).agg(
        players=("player_id", "nunique"),
        median_absolute_score_change=("absolute_score_change", "median"),
        p90_absolute_score_change=("absolute_score_change", lambda x: x.quantile(0.9)),
        median_absolute_rank_change=("absolute_rank_change", "median"),
        p90_absolute_rank_change=("absolute_rank_change", lambda x: x.quantile(0.9)),
        disappeared=("player_disappeared", "sum"),
    )
    return detail, summary


def _rankings_and_influence(
    data: EvaluationData,
    cutoff_ts: float,
    *,
    tournament_beta: float,
    matches_override: pl.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[int, float]]:
    matches, players, appearances = slice_for_cutoff(data, cutoff_ts, matches_override)
    engine = ExposureLogOddsEngine(
        production_config(tournament_beta=tournament_beta), now_ts=cutoff_ts
    )
    result = engine.rank_players(
        matches,
        players,
        appearances=appearances if not appearances.is_empty() else None,
    )
    frame = result.select(
        [pl.col("id").alias("player_id"), "score", "exposure"]
    ).to_pandas()
    influence = {
        int(tid): float(value)
        for tid, value in (getattr(engine, "tournament_influence", {}) or {}).items()
    }
    return frame, influence


def tournament_weighting_property_test(
    data: EvaluationData,
    *,
    cutoff_ts: float | None = None,
    events_per_group: int = 8,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare event-removal contributions with and without strength weights."""

    if cutoff_ts is None:
        cutoff_ts = float(data.resolved["ts"].max()) + 1.0
    weighted_base, influence = _rankings_and_influence(
        data, cutoff_ts, tournament_beta=1.0
    )
    unweighted_base, _ = _rankings_and_influence(data, cutoff_ts, tournament_beta=0.0)
    event_counts = (
        data.resolved[data.resolved["ts"] < cutoff_ts].groupby("tournament_id").size()
    )
    strength = pd.DataFrame(
        [
            {
                "tournament_id": tid,
                "influence": value,
                "matches": int(event_counts.get(tid, 0)),
            }
            for tid, value in influence.items()
            if event_counts.get(tid, 0) >= 10
        ]
    ).sort_values(["influence", "tournament_id"])
    selected = pd.concat(
        [
            strength.head(events_per_group).assign(strength_group="weak"),
            strength.tail(events_per_group).assign(strength_group="strong"),
        ],
        ignore_index=True,
    )
    base_maps = {
        "weighted": score_map(weighted_base, "score"),
        "unweighted": score_map(unweighted_base, "score"),
    }
    rows: list[dict[str, Any]] = []
    for index, event in selected.iterrows():
        tid = int(event["tournament_id"])
        event_matches = data.resolved[data.resolved["tournament_id"] == tid]
        winners = {int(pid) for lineup in event_matches["winners"] for pid in lineup}
        override = data.matches.filter(pl.col("tournament_id") != tid)
        for label, beta in [("weighted", 1.0), ("unweighted", 0.0)]:
            reduced = loopr_rankings(
                data,
                cutoff_ts,
                tournament_beta=beta,
                matches_override=override,
            )
            reduced_map = score_map(reduced, "score")
            contributions = np.array(
                [
                    base_maps[label].get(pid, 0.0) - reduced_map.get(pid, 0.0)
                    for pid in winners
                ],
                dtype=float,
            )
            rows.append(
                {
                    "tournament_id": tid,
                    "strength_group": event["strength_group"],
                    "influence": float(event["influence"]),
                    "matches": int(event["matches"]),
                    "weighting": label,
                    "winner_players": len(winners),
                    "median_winner_score_contribution": float(np.median(contributions)),
                    "mean_winner_score_contribution": float(np.mean(contributions)),
                    "median_contribution_per_match": float(
                        np.median(contributions) / max(1, int(event["matches"]))
                    ),
                }
            )
        print(f"tournament weighting {index + 1}/{len(selected)}: {tid}", flush=True)
    detail = pd.DataFrame(rows)
    grouped = (
        detail.groupby(["weighting", "strength_group"])["median_contribution_per_match"]
        .median()
        .unstack()
    )
    summary = {
        "events_per_strength_group": events_per_group,
        "weak_influence_range": [
            float(selected[selected["strength_group"] == "weak"]["influence"].min()),
            float(selected[selected["strength_group"] == "weak"]["influence"].max()),
        ],
        "strong_influence_range": [
            float(selected[selected["strength_group"] == "strong"]["influence"].min()),
            float(selected[selected["strength_group"] == "strong"]["influence"].max()),
        ],
        "median_contribution_per_match": {
            weighting: {
                group: float(grouped.loc[weighting, group])
                for group in ["weak", "strong"]
            }
            for weighting in grouped.index
        },
        "strong_minus_weak_gap": {
            weighting: float(
                grouped.loc[weighting, "strong"] - grouped.loc[weighting, "weak"]
            )
            for weighting in grouped.index
        },
        "limitation": "observed event-removal sensitivity; event categories are defined by the model's fitted influence and are not randomized",
    }
    return detail, summary
