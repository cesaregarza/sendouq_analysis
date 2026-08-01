"""Publication-only analyses for the Part I essay figures.

The frozen audit point estimates remain authoritative.  This module adds only
the intervals, sensitivity summaries, and new exhibits requested for the essay
figure package.  Every stochastic procedure derives from ``DEFAULT_SEED`` and
uses tournaments as the resampling cluster where match dependence matters.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from .core import (
    DEFAULT_SEED,
    EvaluationData,
    active_roster_player_ids,
    loopr_rankings,
    safe_auc,
    score_map,
    utc_date,
)
from .intended_use.evaluation import graph_metrics_for_players


PREDICTION_BOOTSTRAP_REPETITIONS = 5_000
PLUS_BOOTSTRAP_REPETITIONS = 2_000
DEFAULT_RANK_BOOTSTRAP_REPETITIONS = 200


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _cluster_bootstrap_mean(
    frame: pd.DataFrame,
    value_column: str,
    *,
    repetitions: int,
    seed: int,
    cluster_column: str = "tournament_id",
) -> tuple[float, float, float]:
    """Match-weighted mean and tournament-cluster percentile interval."""

    scoped = frame[[cluster_column, value_column]].dropna().copy()
    grouped = scoped.groupby(cluster_column, sort=True)[value_column].agg(
        ["sum", "count"]
    )
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    observed = float(sums.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    draw_index = rng.integers(0, len(grouped), size=(repetitions, len(grouped)))
    draws = sums[draw_index].sum(axis=1) / counts[draw_index].sum(axis=1)
    return (
        observed,
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def _cluster_bootstrap_difference(
    frame: pd.DataFrame,
    value_column: str,
    *,
    repetitions: int,
    seed: int,
    cluster_column: str = "tournament_id",
) -> tuple[float, float, float]:
    return _cluster_bootstrap_mean(
        frame,
        value_column,
        repetitions=repetitions,
        seed=seed,
        cluster_column=cluster_column,
    )


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """Two-sided 95% Wilson score interval without a package-version dependency."""

    if trials <= 0:
        return np.nan, np.nan
    z = 1.959963984540054
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * np.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return float(center - half_width), float(center + half_width)


def add_prediction_intervals(output: Path) -> pd.DataFrame:
    """Add individual tournament-cluster intervals without changing point estimates."""

    summary_path = output / "tables" / "prediction_summary.csv"
    raw_path = output / "tables" / "walk_forward_predictions.parquet"
    summary = pd.read_csv(summary_path)
    raw = pd.read_parquet(raw_path)
    raw = raw[raw["is_report_window"].astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    for method_index, row in enumerate(summary.itertuples(index=False)):
        group = raw[raw["method"] == row.method]
        estimate, low, high = _cluster_bootstrap_mean(
            group,
            "log_loss",
            repetitions=PREDICTION_BOOTSTRAP_REPETITIONS,
            seed=DEFAULT_SEED + 101_000 + method_index,
        )
        if not np.isclose(estimate, float(row.match_micro_log_loss), atol=1e-12):
            raise AssertionError(f"Prediction point estimate changed for {row.method}")
        rows.append(
            {
                **row._asdict(),
                "match_micro_log_loss_ci_low": low,
                "match_micro_log_loss_ci_high": high,
                "individual_ci_cluster": "tournament",
                "individual_ci_repetitions": PREDICTION_BOOTSTRAP_REPETITIONS,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(summary_path, index=False)
    return result


def add_calibration_intervals(output: Path) -> pd.DataFrame:
    source = pd.read_csv(output / "tables" / "loopr_calibration.csv")
    lows: list[float] = []
    highs: list[float] = []
    successes: list[int] = []
    for row in source.itertuples(index=False):
        won = int(round(float(row.observed_win_rate) * int(row.n_matches)))
        low, high = _wilson_interval(won, int(row.n_matches))
        successes.append(won)
        lows.append(low)
        highs.append(high)
    source["observed_wins"] = successes
    source["wilson_ci_low"] = lows
    source["wilson_ci_high"] = highs
    source["interval"] = "95% Wilson score"
    source.to_csv(output / "tables" / "loopr_calibration_publication.csv", index=False)
    return source


def add_familiarity_intervals(output: Path) -> pd.DataFrame:
    summary = pd.read_csv(output / "tables" / "prediction_by_familiarity.csv")
    raw = pd.read_parquet(output / "tables" / "walk_forward_predictions.parquet")
    raw = raw[raw["is_report_window"].astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(summary.itertuples(index=False)):
        group = raw[
            (raw["method"] == row.method)
            & (raw["familiarity_group"] == row.familiarity_group)
        ]
        estimate, low, high = _cluster_bootstrap_mean(
            group,
            "log_loss",
            repetitions=PREDICTION_BOOTSTRAP_REPETITIONS,
            seed=DEFAULT_SEED + 111_000 + row_index,
        )
        if not np.isclose(estimate, float(row.log_loss), atol=1e-12):
            raise AssertionError(
                f"Familiarity point estimate changed for {row.method}/{row.familiarity_group}"
            )
        rows.append(
            {
                **row._asdict(),
                "log_loss_ci_low": low,
                "log_loss_ci_high": high,
                "ci_cluster": "tournament",
                "ci_repetitions": PREDICTION_BOOTSTRAP_REPETITIONS,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(
        output / "tables" / "prediction_by_familiarity_publication.csv",
        index=False,
    )
    return result


def audit_stability_dip(data: EvaluationData, output: Path) -> dict[str, Any]:
    snapshots = pd.read_parquet(output / "tables" / "monthly_top100.parquet")
    month_a = "2025-10-31"
    month_b = "2025-11-30"
    prior = set(
        snapshots.loc[snapshots["snapshot_date"] == month_a, "player_id"].astype(int)
    )
    following = set(
        snapshots.loc[snapshots["snapshot_date"] == month_b, "player_id"].astype(int)
    )
    # The November 30 snapshot is evaluated just after that UTC month end.
    cutoff_ts = float(pd.Timestamp("2025-12-01", tz="UTC").timestamp())
    active = active_roster_player_ids(data, cutoff_ts, active_days=90.0)
    departures = prior - following
    result = {
        "transition": f"{month_a} to {month_b}",
        "top100_departures": len(departures),
        "departures_no_longer_active_under_90_day_rule": len(departures - active),
        "departures_still_active_but_displaced": len(departures & active),
        "retained": len(prior & following),
        "interpretation": (
            "The 60% retention dip is an activity-eligibility cliff: 39 of 40 "
            "departures aged beyond the 90-day roster-activity window, while only "
            "one still-active member of the prior top 100 was displaced."
        ),
    }
    _write_json(output / "results" / "monthly_stability_dip_audit.json", result)
    pd.DataFrame([result]).to_csv(
        output / "tables" / "monthly_stability_dip_audit.csv", index=False
    )
    return result


def _bootstrap_rank_intervals(
    data: EvaluationData,
    output: Path,
    *,
    repetitions: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Tournament-cluster bootstrap of the final production LOOPR ordering."""

    cutoff_ts = float(data.resolved["ts"].max()) + 1.0
    active = active_roster_player_ids(data, cutoff_ts, active_days=90.0)
    full = loopr_rankings(data, cutoff_ts)
    full = full[full["player_id"].isin(active)].copy()
    full = full.sort_values(
        ["score", "player_id"], ascending=[False, True]
    ).reset_index(drop=True)
    full["published_rank"] = np.arange(1, len(full) + 1)
    target = full.head(50)[["player_id", "published_rank"]].copy()
    target["player_token"] = target["published_rank"].map(
        lambda rank: f"Player {rank:02d}"
    )

    tournament_ids = np.asarray(
        sorted(int(value) for value in data.matches["tournament_id"].unique()),
        dtype=np.int64,
    )
    rng = np.random.default_rng(DEFAULT_SEED)
    sampled = rng.choice(
        tournament_ids,
        size=(repetitions, len(tournament_ids)),
        replace=True,
    )
    count_rows: list[dict[str, int]] = []
    for repetition, sample in enumerate(sampled):
        counts = Counter(map(int, sample))
        count_rows.extend(
            {
                "repetition": repetition,
                "tournament_id": tournament_id,
                "sample_count": count,
            }
            for tournament_id, count in sorted(counts.items())
        )
    counts_frame = pd.DataFrame(count_rows)
    counts_path = output / "cache" / "rank_bootstrap_tournament_counts.parquet"
    counts_path.parent.mkdir(parents=True, exist_ok=True)
    counts_frame.to_parquet(counts_path, index=False)

    draw_path = output / "cache" / "rank_bootstrap_draws.parquet"
    draw_rows: list[dict[str, Any]] = []
    completed: set[int] = set()
    if draw_path.exists():
        old = pd.read_parquet(draw_path)
        expected_tokens = set(target["player_token"])
        if (
            set(old.get("player_token", pd.Series(dtype=str))) <= expected_tokens
            and int(old["repetition"].max()) < repetitions
        ):
            draw_rows.extend(old.to_dict("records"))
            completed = set(old["repetition"].astype(int))

    event_frames = {
        tournament_id: data.matches.filter(pl.col("tournament_id") == tournament_id)
        for tournament_id in tournament_ids
    }
    target_lookup = target.set_index("player_id")
    active_sorted = np.asarray(sorted(active), dtype=np.int64)
    for repetition in range(repetitions):
        if repetition in completed:
            continue
        counts = Counter(map(int, sampled[repetition]))
        frames: list[pl.DataFrame] = []
        for tournament_id, count in sorted(counts.items()):
            frames.extend([event_frames[tournament_id]] * count)
        resampled_matches = pl.concat(frames, rechunk=True)
        ranking = loopr_rankings(
            data,
            cutoff_ts,
            matches_override=resampled_matches,
        )
        mapping = score_map(ranking, "score")
        boot = pd.DataFrame(
            {
                "player_id": active_sorted,
                "score": [
                    mapping.get(int(player_id), 0.0) for player_id in active_sorted
                ],
            }
        ).sort_values(["score", "player_id"], ascending=[False, True])
        boot["bootstrap_rank"] = np.arange(1, len(boot) + 1)
        ranks = boot.set_index("player_id")["bootstrap_rank"]
        for player_id, target_row in target_lookup.iterrows():
            draw_rows.append(
                {
                    "repetition": repetition,
                    "player_id": int(player_id),
                    "player_token": target_row.player_token,
                    "published_rank": int(target_row.published_rank),
                    "bootstrap_rank": int(ranks.loc[int(player_id)]),
                }
            )
        if repetition % 5 == 0 or repetition == repetitions - 1:
            pd.DataFrame(draw_rows).sort_values(
                ["repetition", "published_rank"]
            ).to_parquet(draw_path, index=False)
        if repetition % 10 == 0 or repetition == repetitions - 1:
            print(f"rank bootstrap {repetition + 1}/{repetitions}", flush=True)

    draws = (
        pd.DataFrame(draw_rows)
        .sort_values(["repetition", "published_rank"])
        .reset_index(drop=True)
    )
    draws.to_parquet(draw_path, index=False)
    intervals = (
        draws.groupby(["player_id", "player_token", "published_rank"], as_index=False)
        .agg(
            bootstrap_median_rank=("bootstrap_rank", "median"),
            bootstrap_rank_q025=(
                "bootstrap_rank",
                lambda values: np.quantile(values, 0.025),
            ),
            bootstrap_rank_q25=(
                "bootstrap_rank",
                lambda values: np.quantile(values, 0.25),
            ),
            bootstrap_rank_q75=(
                "bootstrap_rank",
                lambda values: np.quantile(values, 0.75),
            ),
            bootstrap_rank_q975=(
                "bootstrap_rank",
                lambda values: np.quantile(values, 0.975),
            ),
            probability_in_top50=(
                "bootstrap_rank",
                lambda values: np.mean(np.asarray(values) <= 50),
            ),
        )
        .sort_values("published_rank")
    )
    intervals["bootstrap_repetitions"] = repetitions
    intervals.to_csv(output / "tables" / "rank_bootstrap_intervals.csv", index=False)
    summary = {
        "repetitions": repetitions,
        "seed": DEFAULT_SEED,
        "resampling_unit": "tournament",
        "resampling_procedure": (
            "sample all eligible tournaments with replacement; preserve every sampled "
            "event's teams, timestamps, lineups, and outcome directions; recompute the "
            "complete production LOOPR pipeline"
        ),
        "ranking_cutoff": utc_date(cutoff_ts - 1.0),
        "active_universe": len(active),
        "displayed_players": 50,
        "interval": "2.5th to 97.5th percentile of bootstrap rank",
        "missing_in_resample_score": 0.0,
        "cache": str(draw_path.relative_to(output)),
        "sample_count_cache": str(counts_path.relative_to(output)),
    }
    _write_json(output / "results" / "rank_bootstrap.json", summary)
    return intervals, summary


def _plus_vote_field_audit(repository_root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    candidate_fields = {
        "vote_percentage",
        "vote_percent",
        "approval_percentage",
        "approval_percent",
        "yes_votes",
        "no_votes",
        "votes_for",
        "votes_against",
    }
    found: set[str] = set()
    for path in sorted(repository_root.glob("sendou_plus_voting_results*.csv")):
        columns = list(pd.read_csv(path, nrows=0).columns)
        found.update(candidate_fields & set(columns))
        files.append({"file": path.name, "columns": columns})
    return {
        "files_checked": files,
        "candidate_level_vote_fields_found": sorted(found),
        "vote_percentage_available": bool(found),
        "conclusion": (
            "No candidate-level vote percentage or vote-count field exists in the "
            "repository's historical Plus exports. Pass/fail is the only preserved vote measure."
            if not found
            else "At least one candidate-level vote field is available."
        ),
    }


def _assign_exposure_bands(
    frame: pd.DataFrame,
) -> tuple[pd.Series, dict[str, int]]:
    values = frame["prior_matches"].astype(int).to_numpy()
    q1 = int(np.quantile(values, 1.0 / 3.0, method="nearest"))
    q2 = int(np.quantile(values, 2.0 / 3.0, method="nearest"))
    if q1 >= q2:
        raise RuntimeError("Exposure tertile cut points are not distinct")
    labels = pd.Series(index=frame.index, dtype="object")
    labels.loc[frame["prior_matches"] <= q1] = "Low exposure"
    labels.loc[(frame["prior_matches"] > q1) & (frame["prior_matches"] <= q2)] = (
        "Middle exposure"
    )
    labels.loc[frame["prior_matches"] > q2] = "High exposure"
    return labels, {"low_max_prior_matches": q1, "middle_max_prior_matches": q2}


def plus_incumbent_autopsy(
    data: EvaluationData,
    output: Path,
    repository_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    common = pd.read_parquet(output / "tables" / "plus_common_candidates.parquet")
    cutoff_ts = float(pd.Timestamp("2025-11-16", tz="UTC").timestamp())
    train = data.resolved[data.resolved["ts"] < cutoff_ts]
    support = graph_metrics_for_players(
        train,
        common["player_id"].astype(int),
        compute_path_metrics=False,
    )[["player_id", "prior_matches", "core_number", "unique_opponents"]]
    frame = common.merge(support, on="player_id", how="left")
    for column in ["prior_matches", "core_number", "unique_opponents"]:
        frame[column] = (
            pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)
        )
    frame["candidate_status"] = np.where(frame["suggested"], "Suggested", "Incumbent")
    frame["decision"] = np.where(frame["passed"].astype(bool), "Pass", "Fail")
    frame["history_label"] = np.select(
        [
            (~frame["suggested"]) & (frame["passed"] == 0),
            (~frame["suggested"]) & (frame["passed"] == 1),
            frame["suggested"] & (frame["passed"] == 0),
            frame["suggested"] & (frame["passed"] == 1),
        ],
        [
            "incumbent—voted out",
            "incumbent—retained",
            "suggested—not admitted",
            "suggested—admitted",
        ],
        default="unknown",
    )
    frame["exposure_band"], cut_points = _assign_exposure_bands(frame)
    frame["vote_percentage"] = np.nan
    # Deterministic vertical jitter for the binary threshold display.
    frame["decision_y"] = frame["passed"].astype(float)
    frame["decision_y_jittered"] = frame.apply(
        lambda row: (
            float(row.decision_y)
            + (
                ((int(row.player_id) * 2_654_435_761 + DEFAULT_SEED) % 10_001)
                / 10_000.0
                - 0.5
            )
            * 0.22
        ),
        axis=1,
    )
    keep = [
        "player_id",
        "tier",
        "suggested",
        "candidate_status",
        "passed",
        "decision",
        "history_label",
        "score",
        "prior_matches",
        "matches_90d_activity",
        "tournaments_90d",
        "core_number",
        "unique_opponents",
        "exposure_band",
        "vote_percentage",
        "decision_y",
        "decision_y_jittered",
    ]
    result = frame[keep].sort_values(["candidate_status", "tier", "player_id"])
    result.to_parquet(output / "tables" / "plus_incumbent_autopsy.parquet", index=False)
    field_audit = _plus_vote_field_audit(repository_root)
    summary = {
        "preregistered_before_computation_commit": "544984b",
        "vote_field_audit": field_audit,
        "display_used": (
            "pre-registered fallback: binary threshold outcome versus LOOPR score, "
            "faceted by candidate status and sized by outcome-independent exposure tertile"
        ),
        "exposure_definition": "tertiles of pre-vote prior match appearances, computed without outcomes or LOOPR scores",
        "exposure_cut_points": cut_points,
        "never_stood_for_vote_label_available": False,
        "voted_out_definition": "incumbent (suggested=False) with fail outcome",
        "candidates": len(result),
    }
    _write_json(output / "results" / "plus_incumbent_autopsy.json", summary)
    return result, summary


def _macro_tier_auc(frame: pd.DataFrame) -> float:
    values = [
        safe_auc(group["passed"], group["score"])
        for _, group in frame.groupby("tier", sort=True)
    ]
    valid = [float(value) for value in values if value is not None]
    return float(np.mean(valid)) if valid else np.nan


def plus_auc_by_exposure(
    autopsy: pd.DataFrame,
    output: Path,
) -> pd.DataFrame:
    order = ["Low exposure", "Middle exposure", "High exposure"]
    rows: list[dict[str, Any]] = []
    for band_index, band in enumerate(order):
        group = autopsy[autopsy["exposure_band"] == band].copy()
        estimate = _macro_tier_auc(group)
        strata = [
            stratum.index.to_numpy()
            for _, stratum in group.groupby(
                ["tier", "suggested", "passed"], sort=True, dropna=False
            )
        ]
        rng = np.random.default_rng(DEFAULT_SEED + 121_000 + band_index)
        draws: list[float] = []
        for _ in range(PLUS_BOOTSTRAP_REPETITIONS):
            sampled_indices = np.concatenate(
                [rng.choice(index, size=len(index), replace=True) for index in strata]
            )
            value = _macro_tier_auc(group.loc[sampled_indices])
            if np.isfinite(value):
                draws.append(value)
        rows.append(
            {
                "exposure_band": band,
                "candidates": len(group),
                "pass": int(group["passed"].sum()),
                "fail": int((1 - group["passed"]).sum()),
                "minimum_prior_matches": int(group["prior_matches"].min()),
                "maximum_prior_matches": int(group["prior_matches"].max()),
                "institutional_macro_auc": estimate,
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
                "bootstrap_repetitions": PLUS_BOOTSTRAP_REPETITIONS,
                "bootstrap_strata": "tier x candidate status x outcome",
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output / "tables" / "plus_auc_by_exposure_band.csv", index=False)
    return result


def core0_diagnostic(output: Path, intended_output: Path) -> pd.DataFrame:
    support = pd.read_parquet(
        intended_output / "tables" / "graph_support_match.parquet"
    )
    raw = pd.read_parquet(output / "tables" / "walk_forward_predictions.parquet")
    raw = raw[raw["is_report_window"].astype(bool)].merge(
        support[
            [
                "tournament_id",
                "match_id",
                "minimum_core_number",
                "is_report_window",
            ]
        ].drop(columns="is_report_window"),
        on=["tournament_id", "match_id"],
        how="inner",
        validate="many_to_one",
    )
    methods = ["LOOPR", "OpenSkill μ", "OpenSkill ordinal"]
    scoped = raw[raw["method"].isin(methods) & (raw["minimum_core_number"] == 0)].copy()
    rows: list[dict[str, Any]] = []
    for method_index, method in enumerate(methods):
        group = scoped[scoped["method"] == method]
        estimate, low, high = _cluster_bootstrap_mean(
            group,
            "log_loss",
            repetitions=PREDICTION_BOOTSTRAP_REPETITIONS,
            seed=DEFAULT_SEED + 131_000 + method_index,
        )
        rows.append(
            {
                "method": method,
                "matches": len(group),
                "tournaments": group["tournament_id"].nunique(),
                "match_micro_log_loss": estimate,
                "ci_low": low,
                "ci_high": high,
                "ci_cluster": "tournament",
                "ci_repetitions": PREDICTION_BOOTSTRAP_REPETITIONS,
            }
        )
    wide = scoped.pivot(
        index=["tournament_id", "match_id"], columns="method", values="log_loss"
    ).reset_index()
    wide["ordinal_minus_mu_log_loss"] = wide["OpenSkill ordinal"] - wide["OpenSkill μ"]
    difference, difference_low, difference_high = _cluster_bootstrap_difference(
        wide,
        "ordinal_minus_mu_log_loss",
        repetitions=PREDICTION_BOOTSTRAP_REPETITIONS,
        seed=DEFAULT_SEED + 131_500,
    )
    result = pd.DataFrame(rows)
    result["ordinal_minus_mu_log_loss"] = difference
    result["ordinal_minus_mu_ci_low"] = difference_low
    result["ordinal_minus_mu_ci_high"] = difference_high
    result["diagnostic_interpretation"] = (
        "OpenSkill ordinal and mu use the same match history; ordinal differs by "
        "subtracting 3 sigma. Its core-0 advantage therefore reflects the conservative "
        "uncertainty penalty, not additional match information."
    )
    result.to_csv(output / "tables" / "core0_diagnostic.csv", index=False)
    return result


def cross_system_event_deletion(output: Path, intended_output: Path) -> pd.DataFrame:
    source = pd.read_csv(
        intended_output / "tables" / "comparative_tournament_removal.csv"
    )
    source = source[source["support_threshold"] == 0].copy()
    rows: list[dict[str, Any]] = []
    for method, group in source.groupby("method", sort=False):
        rows.append(
            {
                "method": method,
                "events_removed": len(group),
                "selection": "25 events with most active 8-core player-match appearances",
                "median_of_event_median_top100_rank_shift": float(
                    group["median_abs_rank_change_original_top100"].median()
                ),
                "median_of_event_largest_top100_rank_shift": float(
                    group["largest_abs_rank_change_original_top100"].median()
                ),
                "maximum_observed_top100_rank_shift": float(
                    group["largest_abs_rank_change_original_top100"].max()
                ),
                "median_top100_retention": float(group["top100_retention"].median()),
                "worst_top100_retention": float(group["top100_retention"].min()),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(
        output / "tables" / "cross_system_event_deletion_summary.csv",
        index=False,
    )
    return result


def ablation_walk(
    repository_root: Path, output: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_path = repository_root / "docs" / "loopr_ablation_table.csv"
    if not source_path.exists():
        raise FileNotFoundError(
            "N6 requires docs/loopr_ablation_table.csv from the audited project materials"
        )
    source = pd.read_csv(source_path)
    stages = [
        ("Winrate 90d", "Winrate 90d", "Baseline", "Observed record"),
        ("Forward PR only", "+ forward PageRank", "R1", "Strength of schedule"),
        ("LOOPR (no decay)", "+ log-ratio", "R2", "Volume resistance"),
        ("LOOPR + time decay", "+ time decay", "R3", "Recency"),
        (
            "LOOPR + smoothing",
            "+ teleport + smoothing",
            "R4",
            "Limited evidence",
        ),
        (
            "LOOPR + tick-tock",
            "+ tick-tock",
            "Goodhart",
            "Event farming resistance",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for position, (variant, label, requirement, purpose) in enumerate(stages):
        match = source[source["variant"] == variant]
        if len(match) != 1:
            raise RuntimeError(f"Ablation source has {len(match)} rows for {variant!r}")
        row = match.iloc[0].to_dict()
        rows.append(
            {
                "stage": position,
                "variant": variant,
                "display_label": label,
                "requirement": requirement,
                "purpose": purpose,
                **row,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output / "tables" / "ablation_walk.csv", index=False)
    provenance = {
        "source": str(source_path.relative_to(repository_root)),
        "source_sha256": sha256(source_path.read_bytes()).hexdigest(),
        "protocol_note": (
            "Historical ablation protocol retained exactly; it is not the unified "
            "walk-forward protocol and is displayed as a component walk, not a new model comparison."
        ),
        "displayed_stages": [row[1] for row in stages],
    }
    _write_json(output / "results" / "ablation_walk.json", provenance)
    return result, provenance


def _update_manifest(
    output: Path,
    *,
    rank_summary: dict[str, Any],
    autopsy_summary: dict[str, Any],
    ablation_provenance: dict[str, Any],
) -> dict[str, Any]:
    path = output / "evaluation_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if "incumbent_autopsy_preregistration" not in manifest.get("plus_vote", {}):
        raise AssertionError("N2 pre-registration is missing from the manifest")
    manifest["publication_figures"] = {
        "random_seed": DEFAULT_SEED,
        "surfaces": ["dark", "light"],
        "png_width_pixels": 1520,
        "validation_artifact": "results/figure_validation.json",
        "color_accessibility_check": (
            "Machado severity-100 deuteranopia transform over adjacent palette entries; "
            "all systems retain redundant marker or dashed-line encodings"
        ),
        "entity_colors": {
            "LOOPR": "#ec5fa8",
            "OpenSkill ordinal": "#4d9de0",
            "OpenSkill mu": "#8fc4ec",
            "Bradley-Terry (L2)": "#e0a03d",
            "90-day win rate": "#9aa3b2",
            "Chance (0.5)": "#6b7280",
        },
        "prediction_individual_intervals": {
            "unit": "tournament cluster",
            "repetitions": PREDICTION_BOOTSTRAP_REPETITIONS,
        },
        "calibration_intervals": "95% Wilson score intervals",
        "familiarity_intervals": {
            "unit": "tournament cluster",
            "repetitions": PREDICTION_BOOTSTRAP_REPETITIONS,
        },
        "N1_rank_bootstrap": rank_summary,
        "N2_incumbent_autopsy": autopsy_summary,
        "N3_exposure_auc": {
            "bands": "tertiles of pre-vote prior match appearances; outcomes and scores not used to define cut points",
            "estimand": "unweighted macro average of valid within-tier passer-versus-failer AUCs",
            "uncertainty": "candidate bootstrap stratified by tier x status x outcome",
            "repetitions": PLUS_BOOTSTRAP_REPETITIONS,
        },
        "N4_core0": {
            "stratum": "held-out matches whose minimum participant pre-event opponent-graph core number is exactly zero",
            "uncertainty": "tournament-cluster bootstrap",
            "repetitions": PREDICTION_BOOTSTRAP_REPETITIONS,
        },
        "N5_event_deletion": {
            "events": 25,
            "selection": "most active 8-core player-match appearances; outcome direction and scores not used",
            "active_universe": "held fixed at full-data definition",
            "interpretation": "deterministic sensitivity analysis, not a confidence interval",
        },
        "N6_ablation_walk": ablation_provenance,
    }
    _write_json(path, manifest)
    return manifest


def run_publication_analyses(
    data: EvaluationData,
    *,
    repository_root: Path,
    output: Path,
    intended_output: Path,
    rank_bootstrap_repetitions: int = DEFAULT_RANK_BOOTSTRAP_REPETITIONS,
) -> dict[str, Any]:
    """Build every publication-only table/cache and update the manifest."""

    output.mkdir(parents=True, exist_ok=True)
    for child in ["tables", "cache", "results"]:
        (output / child).mkdir(parents=True, exist_ok=True)
    prediction = add_prediction_intervals(output)
    calibration = add_calibration_intervals(output)
    familiarity = add_familiarity_intervals(output)
    stability_dip = audit_stability_dip(data, output)
    rank_intervals, rank_summary = _bootstrap_rank_intervals(
        data,
        output,
        repetitions=rank_bootstrap_repetitions,
    )
    autopsy, autopsy_summary = plus_incumbent_autopsy(data, output, repository_root)
    exposure_auc = plus_auc_by_exposure(autopsy, output)
    core0 = core0_diagnostic(output, intended_output)
    deletion = cross_system_event_deletion(output, intended_output)
    ablation, ablation_provenance = ablation_walk(repository_root, output)
    _update_manifest(
        output,
        rank_summary=rank_summary,
        autopsy_summary=autopsy_summary,
        ablation_provenance=ablation_provenance,
    )
    summary = {
        "seed": DEFAULT_SEED,
        "prediction_methods": len(prediction),
        "calibration_bins": len(calibration),
        "familiarity_rows": len(familiarity),
        "stability_dip": stability_dip,
        "rank_bootstrap": rank_summary,
        "N2_vote_percentage_available": autopsy_summary["vote_field_audit"][
            "vote_percentage_available"
        ],
        "N3_exposure_bands": exposure_auc.to_dict("records"),
        "N4_core0": core0.to_dict("records"),
        "N5_event_deletion": deletion.to_dict("records"),
        "N6_ablation_stages": ablation["display_label"].tolist(),
        "rank_interval_rows": len(rank_intervals),
    }
    _write_json(output / "results" / "publication.json", summary)
    return summary
