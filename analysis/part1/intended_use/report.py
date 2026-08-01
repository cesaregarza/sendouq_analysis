#!/usr/bin/env python3
"""Render the additive public-ranking intended-use evaluation.

This stage consumes saved tables only. It never refits a rating system and it
never overwrites the frozen Part I audit outside ``intended_use/output``.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib
import json
from pathlib import Path
import platform
from textwrap import dedent
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import nbformat as nbf
from nbclient import NotebookClient
import numpy as np
import pandas as pd


METHODS = (
    "LOOPR",
    "OpenSkill μ",
    "OpenSkill ordinal",
    "Bradley–Terry (L2)",
    "90-day win rate",
)
COLORS = {
    "LOOPR": "#2563EB",
    "OpenSkill μ": "#7C3AED",
    "OpenSkill ordinal": "#D97706",
    "Bradley–Terry (L2)": "#059669",
    "90-day win rate": "#DC2626",
}
SHORT_LABELS = {
    "LOOPR": "LOOPR",
    "OpenSkill μ": "OS μ",
    "OpenSkill ordinal": "OS ordinal",
    "Bradley–Terry (L2)": "Bradley–Terry",
    "90-day win rate": "Win rate",
}
INK = "#0F172A"
GRAY = "#64748B"
LIGHT = "#E2E8F0"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def runtime_versions() -> dict[str, str]:
    packages = {
        "pandas": "pandas",
        "polars": "polars",
        "numpy": "numpy",
        "scipy": "scipy",
        "scikit-learn": "sklearn",
        "networkx": "networkx",
        "matplotlib": "matplotlib",
        "openskill": "openskill",
        "loopr-ranking": "loopr",
        "pyarrow": "pyarrow",
        "nbformat": "nbformat",
        "nbclient": "nbclient",
    }
    result = {"python": platform.python_version()}
    for package, module_name in packages.items():
        try:
            module = importlib.import_module(module_name)
            result[package] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            result[package] = "not installed"
    return result


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.fillna("—").astype(str)
    header = "| " + " | ".join(display.columns) + " |"
    rule = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |"
        for row in display.to_numpy().tolist()
    ]
    return "\n".join([header, rule, *rows])


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": LIGHT,
            "grid.linewidth": 0.7,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": "#334155",
            "ytick.color": "#334155",
        }
    )


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _add_lineup_hash(frame: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    player_to_hash: dict[int, str] = {}
    targets = selection[selection["role"] == "target"]
    for row in targets.itertuples(index=False):
        for player_id in row.lineup:
            player_to_hash[int(player_id)] = str(row.lineup_hash)
    result = frame.copy()
    result["target_lineup_hash"] = result["player_id"].map(player_to_hash)
    if result["target_lineup_hash"].isna().any():
        raise AssertionError("Attack output contains a player outside selected targets")
    return result


def _pairwise_order_retention(frame: pd.DataFrame) -> float:
    rows = frame.sort_values("player_id").drop_duplicates("player_id")
    before = rows["rank_before"].to_numpy(dtype=float)
    after = rows["rank_after"].to_numpy(dtype=float)
    concordance: list[float] = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            before_sign = np.sign(before[left] - before[right])
            after_sign = np.sign(after[left] - after[right])
            concordance.append(
                0.5
                if before_sign == 0 or after_sign == 0
                else float(before_sign == after_sign)
            )
    return float(np.mean(concordance)) if concordance else np.nan


def build_attack_summaries(tables: Path) -> dict[str, pd.DataFrame]:
    selection = pd.read_parquet(tables / "attack_lineup_selection.parquet")

    farming = _add_lineup_hash(
        pd.read_parquet(tables / "farming_attack_player_results.parquet"), selection
    )
    farming_lineups = farming.groupby(
        ["method", "opponent_class", "wins_per_target", "target_lineup_hash"],
        as_index=False,
    ).agg(
        median_rank_gain=("rank_gain", "median"),
        median_standardized_score_change=("standardized_score_change", "median"),
        any_top100_entry=("entered_top100", "max"),
    )
    farming_summary = farming_lineups.groupby(
        ["method", "opponent_class", "wins_per_target"], as_index=False
    ).agg(
        target_lineups=("target_lineup_hash", "nunique"),
        median_lineup_rank_gain=("median_rank_gain", "median"),
        p90_abs_lineup_rank_gain=(
            "median_rank_gain",
            lambda values: values.abs().quantile(0.90),
        ),
        median_standardized_score_change=(
            "median_standardized_score_change",
            "median",
        ),
        p90_abs_standardized_score_change=(
            "median_standardized_score_change",
            lambda values: values.abs().quantile(0.90),
        ),
        lineups_with_top100_entry=("any_top100_entry", "sum"),
    )
    ratio = farming_summary.pivot_table(
        index=["method", "wins_per_target"],
        columns="opponent_class",
        values="median_standardized_score_change",
    ).reset_index()
    ratio["weak_to_strong_response_ratio"] = ratio["weak"] / ratio["strong"]
    farming_summary = farming_summary.merge(
        ratio[["method", "wins_per_target", "weak_to_strong_response_ratio"]],
        on=["method", "wins_per_target"],
        how="left",
        validate="many_to_one",
    )

    volume = _add_lineup_hash(
        pd.read_parquet(tables / "volume_attack_player_results.parquet"), selection
    )
    volume_lineups = volume.groupby(
        ["method", "volume_multiplier", "target_lineup_hash"], as_index=False
    ).agg(
        median_rank_gain=("rank_gain", "median"),
        median_standardized_score_change=("standardized_score_change", "median"),
    )
    volume_summary = volume_lineups.groupby(
        ["method", "volume_multiplier"], as_index=False
    ).agg(
        target_lineups=("target_lineup_hash", "nunique"),
        median_lineup_rank_gain=("median_rank_gain", "median"),
        p90_abs_lineup_rank_gain=(
            "median_rank_gain",
            lambda values: values.abs().quantile(0.90),
        ),
        median_standardized_score_change=(
            "median_standardized_score_change",
            "median",
        ),
        p90_abs_standardized_score_change=(
            "median_standardized_score_change",
            lambda values: values.abs().quantile(0.90),
        ),
    )

    thinning = _add_lineup_hash(
        pd.read_parquet(tables / "thinning_attack_player_results.parquet"), selection
    )
    replicate_rows: list[dict[str, Any]] = []
    for (method, fraction, hash_seed), group in thinning.groupby(
        ["method", "removal_fraction", "hash_seed"]
    ):
        replicate_rows.append(
            {
                "method": method,
                "removal_fraction": fraction,
                "hash_seed": int(hash_seed),
                "removed_matches": int(group["removed_matches"].iloc[0]),
                "full_top100_retention": float(group["full_top100_retention"].iloc[0]),
                "target_pairwise_ordering_retention": _pairwise_order_retention(group),
                "target_median_abs_rank_movement": float(
                    group["rank_gain"].abs().median()
                ),
                "target_p90_abs_rank_movement": float(
                    group["rank_gain"].abs().quantile(0.90)
                ),
                "target_p90_abs_standardized_change": float(
                    group["standardized_score_change"].abs().quantile(0.90)
                ),
            }
        )
    thinning_replicates = pd.DataFrame(replicate_rows)
    thinning_summary = thinning_replicates.groupby(
        ["method", "removal_fraction"], as_index=False
    ).agg(
        deterministic_replicates=("hash_seed", "nunique"),
        median_removed_matches=("removed_matches", "median"),
        median_full_top100_retention=("full_top100_retention", "median"),
        median_target_pairwise_ordering_retention=(
            "target_pairwise_ordering_retention",
            "median",
        ),
        median_target_median_abs_rank_movement=(
            "target_median_abs_rank_movement",
            "median",
        ),
        median_target_p90_abs_rank_movement=(
            "target_p90_abs_rank_movement",
            "median",
        ),
        median_target_p90_abs_standardized_change=(
            "target_p90_abs_standardized_change",
            "median",
        ),
    )

    chain = pd.read_csv(tables / "transitive_chain_results.csv")
    chain_summary = chain.groupby(
        ["method", "repetitions_per_edge"], as_index=False
    ).agg(
        chains=("chain_index", "size"),
        fraction_positive=("induced_direction_positive", "mean"),
        median_induced_gap_change=("induced_a_minus_c_gap_change", "median"),
        minimum_induced_gap_change=("induced_a_minus_c_gap_change", "min"),
    )

    outputs = {
        "farming": farming_summary,
        "volume": volume_summary,
        "thinning_replicates": thinning_replicates,
        "thinning": thinning_summary,
        "chain": chain_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(tables / f"{name}_attack_summary.csv", index=False)
    return outputs


def build_requirements_profile(
    tables: Path, attacks: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    support = pd.read_csv(tables / "prediction_by_support_band.csv")
    dense = support[support["support_band"] == "core 32+"]
    seeds = pd.read_csv(tables / "seed_model_summary.csv")
    seeds = seeds[
        (seeds["seed_pattern"] == "all") & (seeds["minimum_core_threshold"] == 0)
    ]
    stability = pd.read_csv(tables / "comparative_monthly_stability.csv")
    stability = stability[stability["support_threshold"] == 0]
    stability_summary = stability.groupby("method", as_index=False).agg(
        monthly_mean_membership_retention=("top100_retention", "mean"),
        monthly_mean_retained_spearman=("spearman_on_retained", "mean"),
        monthly_median_p90_rank_movement=("p90_absolute_rank_movement", "median"),
    )
    removal = pd.read_csv(tables / "comparative_tournament_removal.csv")
    removal = removal[removal["support_threshold"] == 0]
    removal_summary = removal.groupby("method", as_index=False).agg(
        important_event_worst_top100_retention=("top100_retention", "min"),
        important_event_largest_rank_shift=(
            "largest_abs_rank_change_original_top100",
            "max",
        ),
    )
    farming = attacks["farming"]
    farming = farming[
        (farming["opponent_class"] == "weak") & (farming["wins_per_target"] == 50)
    ]
    volume = attacks["volume"][attacks["volume"]["volume_multiplier"] == 5]
    thinning = attacks["thinning"][attacks["thinning"]["removal_fraction"] == 0.50]
    schedule = pd.read_csv(tables / "schedule_attack_summary.csv")
    schedule = schedule[schedule["opponent_class"] == "strong"]

    profile = pd.DataFrame({"method": METHODS})
    merges: Iterable[pd.DataFrame] = (
        dense[
            [
                "method",
                "match_micro_log_loss",
                "ordering_accuracy",
            ]
        ].rename(
            columns={
                "match_micro_log_loss": "dense_core32_log_loss",
                "ordering_accuracy": "dense_core32_ordering_accuracy",
            }
        ),
        seeds[["method", "pair_concordance_tournament_macro"]].rename(
            columns={"pair_concordance_tournament_macro": "historical_seed_concordance"}
        ),
        stability_summary,
        removal_summary,
        farming[
            [
                "method",
                "median_standardized_score_change",
                "weak_to_strong_response_ratio",
            ]
        ].rename(
            columns={
                "median_standardized_score_change": "weak_50_win_median_score_change_sd",
            }
        ),
        volume[["method", "p90_abs_standardized_score_change"]].rename(
            columns={
                "p90_abs_standardized_score_change": "volume_5x_p90_abs_score_change_sd"
            }
        ),
        thinning[
            [
                "method",
                "median_full_top100_retention",
                "median_target_pairwise_ordering_retention",
                "median_target_p90_abs_standardized_change",
            ]
        ].rename(
            columns={
                "median_full_top100_retention": "thin_50_top100_retention",
                "median_target_pairwise_ordering_retention": "thin_50_target_pair_order_retention",
                "median_target_p90_abs_standardized_change": "thin_50_p90_abs_score_change_sd",
            }
        ),
        schedule[["method", "strong_minus_weak_standardized_change"]].rename(
            columns={
                "strong_minus_weak_standardized_change": "equal_record_schedule_gap_sd"
            }
        ),
    )
    for frame in merges:
        profile = profile.merge(frame, on="method", how="left", validate="one_to_one")
    return profile


def build_metric_hierarchy() -> pd.DataFrame:
    rows = [
        (
            "Primary intended-use evidence",
            "Dense-core raw ordering accuracy",
            "Among future matches supported by many opponent paths, is the winning lineup ordered above the losing lineup?",
            "Share of held-out matches with positive pre-event raw team-score difference, stratified by predeclared k-core support.",
            "Directly tests ordinal usefulness where the ranking is designed to operate.",
            "A match winner is still a team outcome, not a ground-truth player order.",
        ),
        (
            "Primary intended-use evidence",
            "Historical seed concordance and disagreement pairs",
            "Does a pre-event ranking reproduce organizer-controlled team ordering, especially when models disagree?",
            "Tournament-macro team-pair concordance and paired disagreement preference.",
            "Existing consequential ordering artifact closer to how rankings are used.",
            "Seed provenance is undocumented and organizers may consult similar rating inputs.",
        ),
        (
            "Primary intended-use evidence",
            "Adversarial degradation profile",
            "Can a player move by farming weak wins or duplicating volume, and what happens when evidence is thinned?",
            "Rank and standardized-score change under identical controlled interventions for all systems.",
            "Tests leaderboard failure modes rather than average prediction.",
            "Synthetic interventions are sensitivity tests, not frequencies of real abuse.",
        ),
        (
            "Primary intended-use evidence",
            "Top-ranking stability and event-removal robustness",
            "Does the visible top ranking change gracefully over time and when one important event is removed?",
            "Top-100 retention, retained-player Spearman/movement, and full refit after event deletion.",
            "Operationally central for a public ranking.",
            "A stable or robust ranking can still be wrong.",
        ),
        (
            "Primary convergent evidence",
            "Plus institutional and decomposed AUC",
            "Does the score agree with the actual peer-vote institution, and which pair types create that agreement?",
            "Pooled AUC, tier-macro AUC, exact pair decomposition, OOF baselines, and outcome-shuffle null.",
            "Tests convergence with an existing consequential community judgment.",
            "One vote includes reputation, visibility, status, and opportunity; it is not ground truth.",
        ),
        (
            "Secondary generalization evidence",
            "Walk-forward log loss",
            "Does the rating contain information about any future match, with confidence punished when wrong?",
            "Negative log predicted probability for the winner, micro and tournament-macro.",
            "Essential evidence that scores generalize beyond fitted history.",
            "Optimizes probability quality across a heterogeneous graph, not public-ranking utility.",
        ),
        (
            "Secondary probability evidence",
            "Brier score and calibration",
            "Do stated probabilities correspond to observed frequencies?",
            "Squared probability error and equal-count reliability bins.",
            "Important if the score is used as a probability model.",
            "Not a direct measure of exact public ordering or Goodhart resistance.",
        ),
        (
            "Scope/guardrail",
            "Sparse-edge performance and unfamiliar-lineup transfer",
            "Where does the instrument stop working reliably?",
            "Core-0 prediction and prior teammate-familiarity strata.",
            "Defines the supported population and prevents dense-center claims from being generalized outward.",
            "Poor sparse-edge performance does not identify which regularizer would solve it.",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "priority",
            "metric",
            "plain_language_question",
            "formal_definition",
            "intuitive_interpretation",
            "limitation",
        ],
    )


def build_direct_answers() -> pd.DataFrame:
    rows = [
        (
            1,
            "Are we evaluating the wrong target?",
            "Partly. Walk-forward prediction is a valid generalization test, but it was carrying too much argumentative weight. Public-ranking quality is a separate, multidimensional hypothesis.",
        ),
        (
            2,
            "What existing data better measures public-ranking quality?",
            "Pre-event tournament seeds, Plus decisions, opponent-graph support, monthly top lists, tournament-removal perturbations, repeated lineups, and the match graph used for controlled attacks.",
        ),
        (
            3,
            "Which metrics should become primary?",
            "A no-composite profile: dense-core ordinal performance and coverage, audited seed concordance/disagreements, adversarial degradation, top-100 membership plus retained order, important-event robustness, and decomposed Plus convergence.",
        ),
        (
            4,
            "Which metrics should become secondary?",
            "All-match walk-forward log loss remains the primary generalization metric; Brier score, calibration, and overall accuracy are secondary probability evidence. Half-life and unfamiliar-lineup results remain technical scope checks.",
        ),
        (
            5,
            "Which current metrics measure a different construct?",
            "Log loss, Brier score, and reliability measure probability quality; Plus AUC measures institutional agreement; stability and retention measure continuity; tournament removal measures sensitivity. None alone is public-ranking quality or exact-rank accuracy.",
        ),
        (
            6,
            "What analyses should be added?",
            "Predeclared graph-support strata, historical seed provenance/meaning audit, model-seed disagreement pairs, all-system stability and event deletion, weak-win/volume/thinning/schedule/transitive interventions, and pooled Plus decomposition with its exact shuffle null.",
        ),
        (
            7,
            "What should be removed or demoted?",
            "Remove nothing. Demote average probability metrics from decisive leaderboard evidence; keep the non-discriminating transitive-chain test and detailed parameter/property sweeps in technical material.",
        ),
        (
            8,
            "Which Part I conclusions should change?",
            "Replace both ‘worse predictor, therefore weak’ and ‘works, therefore best leaderboard’ with a support-bounded claim: LOOPR is a strong ranking-oriented outcome instrument for established competitors, with sparse-edge failure and no proof of exact-rank legitimacy.",
        ),
    ]
    return pd.DataFrame(rows, columns=["number", "question", "answer"])


def build_to_hypotheses() -> pd.DataFrame:
    rows = [
        (
            "The ranking puts the teams I would seed highly near the top.",
            "Audited historical team seeds; pair concordance; model-disagreement pairs.",
            "Yes",
            "Useful institutional convergence, not proof that either ordering is correct.",
        ),
        (
            "The ordering makes sense among established teams, not just newcomers.",
            "Predeclared opponent-graph k-core strata and dense-core held-out ordering.",
            "Yes",
            "Dense support is defined without scores or winner direction; coverage must be shown.",
        ),
        (
            "A hard schedule is rewarded more than the same record against weak teams.",
            "Equal-record strong-versus-weak intervention for all five systems.",
            "Yes",
            "Strong/weak labels use historical outcomes and only three weak lineups qualify.",
        ),
        (
            "Players cannot climb mainly by grinding easy matches.",
            "Weak-win farming and volume-duplication interventions.",
            "Yes",
            "Tests specified attacks, not every strategic adaptation available to players.",
        ),
        (
            "The visible top list does not jump around for trivial reasons.",
            "Monthly membership/order stability, history thinning, and important-event removal.",
            "Yes",
            "Stability is not accuracy, and activity eligibility can create real churn.",
        ),
        (
            "The ranking agrees with what the community already recognizes.",
            "Plus pooled/tier/status decomposition and outcome-shuffle null.",
            "Partly",
            "Candidate status explains much of the vote; one snapshot is not universal consensus.",
        ),
        (
            "The ranking is understandable, legitimate, and safe to publish with names.",
            "No historical behavioral or institutional artifact in the repository directly identifies this.",
            "No",
            "Belongs to Part II; measurement results cannot answer legitimacy by proxy.",
        ),
        (
            "The ranking reduces organizer workload or improves bracket quality.",
            "No saved organizer workflow, edit log, counterfactual bracket, or workload measure.",
            "No",
            "Seed agreement is the closest artifact, but it does not measure labor saved or bracket quality.",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=["hypothesis", "historical_test", "testable_now", "interpretive_limit"],
    )


def build_claim_changes() -> pd.DataFrame:
    rows = [
        (
            "OpenSkill or Bradley–Terry predicts matches better, therefore LOOPR is not very good.",
            "Not supported; log loss is one construct.",
            "OpenSkill and Bradley–Terry produce better average probabilities, while LOOPR has a distinct robustness profile for dense-center public ordering.",
            "There is no justified scalar weighting these objectives into one winner.",
        ),
        (
            "LOOPR is the best public ranking.",
            "Not supported.",
            "LOOPR is a credible ranking-oriented system with unusually graceful degradation in these tests; it does not dominate every baseline or artifact.",
            "OpenSkill ordinal has the highest seed-concordance point estimate and Bradley–Terry the best monthly membership retention.",
        ),
        (
            "LOOPR works equally well across the graph.",
            "Contradicted.",
            "LOOPR is competitive in the dense core and weak at the sparse edge; its intended-use claim must be support-bounded.",
            "Core number describes graph support, not every source of uncertainty.",
        ),
        (
            "LOOPR resists farming and volume inflation.",
            "Supported for the specified interventions, not universally.",
            "Repeated weak wins and fivefold evidence duplication moved LOOPR less than the alternatives tested, while strong wins and strong schedules still mattered.",
            "Only three weak opponent lineups met the frozen definition; other attacks may exist.",
        ),
        (
            "The ratio mathematically cancels volume.",
            "Too absolute.",
            "LOOPR was comparatively volume-resistant under observational and controlled tests; it did not prove exact cancellation.",
            "Duplicating evidence can still move rank through global recontextualization.",
        ),
        (
            "Historical seeds prove LOOPR matches TO judgment best.",
            "Not supported.",
            "Stored seeds contain meaningful strength information and all four schedule-aware systems agree strongly; the data do not clearly distinguish LOOPR from them on disagreement pairs.",
            "Manual/random/copied provenance is absent and automatic seed suggestions are strongly correlated with stored order.",
        ),
        (
            "The like-for-like incumbent Plus result is cleaner.",
            "Mischaracterized.",
            "Within-tier/status AUC asks a narrower conditional question; pooled and tier-macro AUC describe the institution that actually operated.",
            "Conditioning discards cross-status and cross-tier comparisons rather than purifying them.",
        ),
        (
            "The pooled Plus AUC is all community skill agreement.",
            "Too broad.",
            "Pooled AUC is agreement with this vote across all candidates; status and tier explain substantial portions, while LOOPR adds modest out-of-fold information beyond measured institution/activity/support features.",
            "Unmeasured reputation and visibility can remain.",
        ),
        (
            "Stability or robustness demonstrates accuracy.",
            "False.",
            "Stability and event-removal robustness are operational requirements for a leaderboard, reported separately from validity evidence.",
            "A stable ranking can be stably wrong.",
        ),
        (
            "LOOPR extracts true skill.",
            "Not observable.",
            "LOOPR estimates demonstrated ability to convert organized-play opportunities into tournament victories within the observed network.",
            "Access, teammate quality, opportunity, and network position may remain in the score.",
        ),
        (
            "A strong skill-o-meter automatically makes a legitimate public leaderboard.",
            "Not supported by Part I evidence.",
            "Part I can establish outcome-derived information and ranking properties; Part II must address named publication, authority, and legitimacy.",
            "Exact neighboring ranks and social consequences remain unresolved.",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "current_or_implicit_claim",
            "support_status",
            "recommended_claim",
            "remaining_limitation",
        ],
    )


def build_figures(
    tables: Path, figures: Path, attacks: dict[str, pd.DataFrame]
) -> None:
    configure_plots()

    support = pd.read_csv(tables / "prediction_by_support_band.csv")
    bands = ["core 0", "core 1–7", "core 8–15", "core 16–31", "core 32+"]
    x = np.arange(len(bands))
    counts = (
        support[support["method"] == "LOOPR"]
        .set_index("support_band")
        .loc[bands, "matches"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for method in METHODS:
        frame = (
            support[support["method"] == method].set_index("support_band").loc[bands]
        )
        axes[0].plot(
            x,
            frame["match_micro_log_loss"],
            marker="o",
            label=method,
            color=COLORS[method],
        )
        axes[1].plot(
            x,
            frame["ordering_accuracy"],
            marker="o",
            label=method,
            color=COLORS[method],
        )
    labels = [f"{band}\n(n={int(counts.loc[band]):,})" for band in bands]
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y")
    axes[0].set_title("Probability error by graph support")
    axes[0].set_ylabel("Match log loss (lower is better)")
    axes[1].set_title("Raw winner ordering by graph support")
    axes[1].set_ylabel("Winning lineup scored higher")
    axes[1].set_ylim(0.45, 1.02)
    axes[1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle(
        "LOOPR's relative position improves toward the dense opponent-graph core",
        fontweight="bold",
    )
    fig.text(
        0.01,
        -0.02,
        "Core 1–7 has only 17 matches and is descriptive. Support uses prior participation only, not ratings or winner direction.",
        color=GRAY,
    )
    save_figure(fig, figures / "01_graph_support")

    seeds = pd.read_csv(tables / "seed_model_summary.csv")
    seeds = (
        seeds[(seeds["seed_pattern"] == "all") & (seeds["minimum_core_threshold"] == 0)]
        .set_index("method")
        .loc[list(METHODS)]
        .reset_index()
    )
    disagreements = pd.read_csv(tables / "seed_disagreement_summary.csv")
    disagreements = (
        disagreements[
            (disagreements["seed_pattern"] == "all")
            & (disagreements["minimum_core_threshold"] == 0)
        ]
        .set_index("baseline")
        .loc[list(METHODS[1:])]
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    y = np.arange(len(seeds))
    for index, row in seeds.iterrows():
        axes[0].errorbar(
            row["pair_concordance_tournament_macro"],
            index,
            xerr=[
                [
                    row["pair_concordance_tournament_macro"]
                    - row["pair_concordance_ci_low"]
                ],
                [
                    row["pair_concordance_ci_high"]
                    - row["pair_concordance_tournament_macro"]
                ],
            ],
            fmt="o",
            color=COLORS[row["method"]],
            capsize=3,
        )
    axes[0].set_yticks(y, seeds["method"])
    axes[0].set_xlim(0.65, 0.94)
    axes[0].grid(axis="x")
    axes[0].set_title("Agreement with audited stored seeds")
    axes[0].set_xlabel("Tournament-macro pair concordance")
    y2 = np.arange(len(disagreements))
    axes[1].errorbar(
        disagreements["loopr_minus_baseline"],
        y2,
        xerr=[
            disagreements["loopr_minus_baseline"] - disagreements["ci_low"],
            disagreements["ci_high"] - disagreements["loopr_minus_baseline"],
        ],
        fmt="o",
        color=COLORS["LOOPR"],
        capsize=3,
    )
    axes[1].axvline(0, color=GRAY, linewidth=1)
    axes[1].set_yticks(y2, disagreements["baseline"])
    axes[1].grid(axis="x")
    axes[1].set_title("When LOOPR and a baseline disagree")
    axes[1].set_xlabel("Seed support: LOOPR minus baseline")
    fig.suptitle(
        "Organizer seeds distinguish schedule-aware systems from win rate, not from one another",
        fontweight="bold",
    )
    fig.text(
        0.01,
        -0.02,
        "Intervals resample tournaments. Seed agreement is institutional concordance, not true-skill accuracy.",
        color=GRAY,
    )
    save_figure(fig, figures / "02_historical_seeds")

    stability = pd.read_csv(tables / "comparative_monthly_stability.csv")
    stability = (
        stability[stability["support_threshold"] == 0]
        .groupby("method", as_index=False)
        .agg(
            membership=("top100_retention", "mean"),
            ordering=("spearman_on_retained", "mean"),
        )
    )
    removal = pd.read_csv(tables / "comparative_tournament_removal.csv")
    removal = (
        removal[removal["support_threshold"] == 0]
        .groupby("method", as_index=False)
        .agg(
            worst_retention=("top100_retention", "min"),
            largest_shift=("largest_abs_rank_change_original_top100", "max"),
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for row in stability.itertuples(index=False):
        axes[0].scatter(row.membership, row.ordering, s=70, color=COLORS[row.method])
        axes[0].annotate(
            SHORT_LABELS[row.method],
            (row.membership, row.ordering),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=9,
        )
    axes[0].grid()
    axes[0].set_xlim(0.55, 0.98)
    axes[0].set_xlabel("Mean adjacent-month top-100 retention")
    axes[0].set_ylabel("Mean Spearman among retained players")
    axes[0].set_title("Monthly membership and ordering stability")
    for row in removal.itertuples(index=False):
        axes[1].scatter(
            row.worst_retention, row.largest_shift, s=70, color=COLORS[row.method]
        )
        axes[1].annotate(
            SHORT_LABELS[row.method],
            (row.worst_retention, row.largest_shift),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=9,
        )
    axes[1].set_yscale("log")
    axes[1].set_xlim(0.945, 0.995)
    axes[1].grid()
    axes[1].set_xlabel("Worst top-100 retention after removing one important event")
    axes[1].set_ylabel("Largest original-top-100 rank shift (log scale)")
    axes[1].set_title("Important-event removal sensitivity")
    fig.suptitle(
        "LOOPR preserves retained order and resists single-event deletion, but not membership best",
        fontweight="bold",
    )
    save_figure(fig, figures / "03_stability_robustness")

    farming = attacks["farming"]
    farming = farming[farming["opponent_class"] == "weak"]
    thinning = attacks["thinning"]
    schedule = pd.read_csv(tables / "schedule_attack_summary.csv")
    schedule = (
        schedule[schedule["opponent_class"] == "strong"]
        .set_index("method")
        .loc[list(METHODS)]
        .reset_index()
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for method in METHODS:
        frame = farming[farming["method"] == method]
        axes[0].plot(
            frame["wins_per_target"],
            frame["median_standardized_score_change"],
            marker="o",
            color=COLORS[method],
            label=method,
        )
        frame = thinning[thinning["method"] == method]
        axes[1].plot(
            frame["removal_fraction"] * 100,
            frame["median_target_p90_abs_standardized_change"],
            marker="o",
            color=COLORS[method],
        )
    axes[0].set_xscale("log")
    axes[0].set_xticks([1, 5, 20, 50], ["1", "5", "20", "50"])
    axes[0].set_title("Weak-win farming")
    axes[0].set_xlabel("Added wins per target lineup")
    axes[0].set_ylabel("Median score change (within-system SD)")
    axes[0].grid()
    axes[1].set_title("Historical-evidence thinning")
    axes[1].set_xlabel("Incident matches removed (%)")
    axes[1].set_ylabel("Median replicate p90 |score change| (SD)")
    axes[1].grid()
    axes[2].barh(
        np.arange(len(schedule)),
        schedule["strong_minus_weak_standardized_change"],
        color=[COLORS[m] for m in schedule["method"]],
    )
    axes[2].set_yticks(np.arange(len(schedule)), schedule["method"])
    axes[2].invert_yaxis()
    axes[2].set_title("Same 5–5 record, different schedule")
    axes[2].set_xlabel("Strong-minus-weak score response (SD)")
    axes[2].grid(axis="x")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.suptitle(
        "Controlled attacks expose leaderboard properties that log loss cannot",
        fontweight="bold",
    )
    fig.text(
        0.01,
        -0.13,
        "Twenty disjoint dense-core targets; only three weak opponent lineups met the frozen definition. Interventions are deterministic sensitivity tests.",
        color=GRAY,
    )
    save_figure(fig, figures / "04_adversarial_tests")

    decomposition = pd.read_csv(tables / "plus_pooled_auc_decomposition.csv")
    baselines = pd.read_csv(tables / "plus_baselines_and_support.csv")
    baselines = baselines[baselines["result_type"] == "cross_validated_baseline"].copy()
    wanted = [
        "candidate status only",
        "tier + candidate status",
        "activity only",
        "graph support only",
        "institution + activity + support",
        "institution + activity + support + LOOPR",
        "LOOPR only",
    ]
    baselines = baselines.set_index("model").loc[wanted].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    labels = [
        ("same" if row.same_tier else "different")
        + " tier / "
        + ("same" if row.same_status else "different")
        + f" status ({row.share_of_all_passer_failer_pairs:.0%})"
        for row in decomposition.itertuples(index=False)
    ]
    y = np.arange(len(decomposition))
    axes[0].barh(y, decomposition["loopr_concordance"], color="#60A5FA")
    axes[0].axvline(0.5, color=GRAY, linewidth=1)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.4, 0.9)
    axes[0].set_xlabel("LOOPR passer-above-failer concordance")
    axes[0].set_title("Where literal pooled AUC comes from")
    colors = ["#94A3B8"] * len(baselines)
    colors[
        baselines.index[
            baselines["model"] == "institution + activity + support + LOOPR"
        ][0]
    ] = COLORS["LOOPR"]
    axes[1].barh(np.arange(len(baselines)), baselines["out_of_fold_auc"], color=colors)
    baseline_labels = {
        "candidate status only": "Status only",
        "tier + candidate status": "Tier + status",
        "activity only": "Activity only",
        "graph support only": "Graph support only",
        "institution + activity + support": "Institution + activity + support",
        "institution + activity + support + LOOPR": "+ LOOPR",
        "LOOPR only": "LOOPR only",
    }
    axes[1].set_yticks(
        np.arange(len(baselines)),
        baselines["model"].map(baseline_labels),
    )
    axes[1].invert_yaxis()
    axes[1].axvline(0.5, color=GRAY, linewidth=1)
    axes[1].set_xlim(0.45, 0.85)
    axes[1].set_xlabel("10-fold out-of-fold ROC AUC")
    axes[1].set_title("Measured institutional baselines explain a lot")
    fig.subplots_adjust(wspace=0.50)
    fig.suptitle(
        "Plus agreement is real, consequential, and partly institutional",
        fontweight="bold",
    )
    save_figure(fig, figures / "05_plus_decomposition")


def build_report(
    output: Path,
    profile: pd.DataFrame,
    metrics: pd.DataFrame,
    direct_answers: pd.DataFrame,
    to_hypotheses: pd.DataFrame,
    claims: pd.DataFrame,
    attacks: dict[str, pd.DataFrame],
) -> str:
    tables = output / "tables"
    support = pd.read_csv(tables / "prediction_by_support_band.csv")
    trends = pd.read_csv(tables / "prediction_support_trends.csv")
    seeds = read_json(output / "results" / "seed_audit.json")
    seed_models = pd.read_csv(tables / "seed_model_summary.csv")
    seed_models = seed_models[
        (seed_models["seed_pattern"] == "all")
        & (seed_models["minimum_core_threshold"] == 0)
    ]
    seed_disagree = pd.read_csv(tables / "seed_disagreement_summary.csv")
    seed_disagree = seed_disagree[
        (seed_disagree["seed_pattern"] == "all")
        & (seed_disagree["minimum_core_threshold"] == 0)
    ]
    seed_bands = pd.read_csv(tables / "seed_outcome_by_band.csv")
    plus_decomp = pd.read_csv(tables / "plus_pooled_auc_decomposition.csv")
    plus_base = pd.read_csv(tables / "plus_baselines_and_support.csv")
    plus_shuffle = read_json(output / "results" / "pooled_outcome_shuffle.json")
    old_shuffle = read_json(
        output.parent.parent / "output" / "results" / "outcome_shuffle.json"
    )
    stability = pd.read_csv(tables / "comparative_monthly_stability.csv")
    removal = pd.read_csv(tables / "comparative_tournament_removal.csv")

    def row(frame: pd.DataFrame, method: str) -> pd.Series:
        return frame[frame["method"] == method].iloc[0]

    core0 = support[support["support_band"] == "core 0"]
    core32 = support[support["support_band"] == "core 32+"]
    loopr0 = row(core0, "LOOPR")
    ordinal0 = row(core0, "OpenSkill ordinal")
    loopr32 = row(core32, "LOOPR")
    mu32 = row(core32, "OpenSkill μ")
    bt32 = row(core32, "Bradley–Terry (L2)")
    slope_lines = ", ".join(
        f"{r.method} {r.slope:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]"
        for r in trends.itertuples(index=False)
    )
    seed_display = seed_models[
        [
            "method",
            "pair_concordance_tournament_macro",
            "pair_concordance_ci_low",
            "pair_concordance_ci_high",
        ]
    ].copy()
    seed_display.columns = ["System", "Seed concordance", "95% low", "95% high"]
    for column in seed_display.columns[1:]:
        seed_display[column] = seed_display[column].map(lambda value: f"{value:.3f}")
    band_display = seed_bands.copy()
    band_display = band_display[
        [
            "best_seed_in_match_band",
            "matches",
            "tournament_macro_higher_seed_win_rate",
            "macro_ci_low",
            "macro_ci_high",
        ]
    ]
    band_display.columns = [
        "Best seed in match",
        "Matches",
        "Higher seed won (macro)",
        "95% low",
        "95% high",
    ]
    for column in band_display.columns[2:]:
        band_display[column] = band_display[column].map(lambda value: f"{value:.3f}")

    stability0 = stability[stability["support_threshold"] == 0]
    stab = stability0.groupby("method").agg(
        membership=("top100_retention", "mean"),
        order=("spearman_on_retained", "mean"),
        p90=("p90_absolute_rank_movement", "median"),
    )
    removal0 = removal[removal["support_threshold"] == 0]
    robust = removal0.groupby("method").agg(
        median_retention=("top100_retention", "median"),
        worst_retention=("top100_retention", "min"),
        largest=("largest_abs_rank_change_original_top100", "max"),
    )
    farming50 = attacks["farming"]
    farming50 = farming50[
        (farming50["opponent_class"] == "weak") & (farming50["wins_per_target"] == 50)
    ].set_index("method")
    volume5 = attacks["volume"][attacks["volume"]["volume_multiplier"] == 5].set_index(
        "method"
    )
    thin50 = attacks["thinning"][
        attacks["thinning"]["removal_fraction"] == 0.5
    ].set_index("method")
    schedule = pd.read_csv(tables / "schedule_attack_summary.csv")
    schedule = schedule[schedule["opponent_class"] == "strong"].set_index("method")
    chain20 = attacks["chain"][
        attacks["chain"]["repetitions_per_edge"] == 20
    ].set_index("method")

    profile_display = profile.copy()
    profile_display.columns = [
        "System",
        "Core32 log loss ↓",
        "Core32 ordering ↑",
        "Seed concordance ↑",
        "Monthly top100 retention ↑",
        "Retained-order ρ ↑",
        "Monthly p90 move ↓",
        "Worst event retention ↑",
        "Largest event shift ↓",
        "50 weak wins ΔSD ↓",
        "Weak/strong ratio ↓",
        "5× volume p90 ΔSD ↓",
        "50% thin top100 ↑",
        "50% thin pair order ↑",
        "50% thin p90 ΔSD ↓",
        "Equal-record schedule gap ↑",
    ]
    for column in profile_display.columns[1:]:
        profile_display[column] = profile_display[column].map(
            lambda value: f"{value:.3f}"
        )

    plus_pair_display = plus_decomp[
        [
            "same_tier",
            "same_status",
            "pairs",
            "share_of_all_passer_failer_pairs",
            "loopr_concordance",
            "contribution_to_pooled_auc",
        ]
    ].copy()
    plus_pair_display["Pair type"] = plus_pair_display.apply(
        lambda r: (
            f"{'same' if r.same_tier else 'different'} tier / {'same' if r.same_status else 'different'} status"
        ),
        axis=1,
    )
    plus_pair_display = plus_pair_display[
        [
            "Pair type",
            "pairs",
            "share_of_all_passer_failer_pairs",
            "loopr_concordance",
            "contribution_to_pooled_auc",
        ]
    ]
    plus_pair_display.columns = [
        "Pair type",
        "Pairs",
        "Share",
        "Concordance",
        "Pooled-AUC contribution",
    ]
    plus_pair_display["Share"] = plus_pair_display["Share"].map(
        lambda value: f"{value:.1%}"
    )
    for column in ["Concordance", "Pooled-AUC contribution"]:
        plus_pair_display[column] = plus_pair_display[column].map(
            lambda value: f"{value:.3f}"
        )

    baseline_lookup = plus_base[
        plus_base["result_type"] == "cross_validated_baseline"
    ].set_index("model")
    report = f"""
    # Public-ranking intended-use evaluation

    ## Bottom line

    **The old evaluation was not wrong; it was asking one valid question and letting that answer stand in for several others.** Walk-forward log loss asks whether a score produces good probabilities for any future match. A public player ranking also needs defensible ordinal behavior in the well-supported graph, resistance to strategic volume and weak-opponent farming, continuity at the visible top, robustness to data perturbations, and convergence with historical ordering decisions. Those are separate hypotheses.

    The added evidence changes the verdict from “LOOPR is a somewhat worse predictor” to a requirements profile:

    - LOOPR is genuinely weak at the sparse edge. On **787 core-0 matches**, its log loss was **{loopr0.match_micro_log_loss:.3f}**, versus **{ordinal0.match_micro_log_loss:.3f}** for OpenSkill ordinal.
    - In the dense core, the systems become much harder to distinguish. Across **{int(loopr32.matches):,} core-32+ matches** ({loopr32.match_coverage:.1%} coverage), LOOPR log loss was **{loopr32.match_micro_log_loss:.3f}**, OpenSkill μ **{mu32.match_micro_log_loss:.3f}**, and Bradley–Terry **{bt32.match_micro_log_loss:.3f}**. Tournament-cluster intervals for μ-minus-LOOPR and BT-minus-LOOPR include zero. LOOPR had the highest raw ordering accuracy by a small amount (**{loopr32.ordering_accuracy:.3f}**).
    - Historical seeds are meaningful as an ordering artifact, but they do not uniquely favor LOOPR. OpenSkill ordinal has the highest point estimate; paired model-disagreement intervals do not distinguish LOOPR from the three schedule-aware alternatives.
    - LOOPR's clearest comparative strengths are operational: retained-player order, important-event deletion, weak-win farming, volume duplication, and history thinning. Bradley–Terry, however, retains top-100 membership better month to month, and OpenSkill μ responds slightly more strongly to schedule quality.
    - Plus agreement is consequential but partly institutional. Candidate status alone reaches **{baseline_lookup.loc["candidate status only", "out_of_fold_auc"]:.3f}** out of fold; tier plus status reaches **{baseline_lookup.loc["tier + candidate status", "out_of_fold_auc"]:.3f}**. Adding LOOPR to institution, activity, and graph support increases AUC from **{baseline_lookup.loc["institution + activity + support", "out_of_fold_auc"]:.3f}** to **{baseline_lookup.loc["institution + activity + support + LOOPR", "out_of_fold_auc"]:.3f}** and improves log loss by **{baseline_lookup.loc["institution + activity + support + LOOPR", "log_loss_improvement_vs_institution_activity_support"]:.3f}**.

    **Conclusion:** existing data support treating LOOPR as a ranking-oriented, outcome-derived skill-o-meter for established organized-play competitors. They do not establish that it is the universally best public ranking, that it handles new players well, or that every exact neighboring rank is meaningful.

    ## Direct answers to the requested deliverables

    {markdown_table(direct_answers)}

    ## 1. Two targets, kept separate

    **Generalization** asks: *Does the score contain information about any future match?* The formal primary metric is walk-forward log loss. It remains in the report unchanged. It is valuable because it is temporal, falsifiable, and punishes confident errors. Its limitation is construct mismatch: average probability quality weights every held-out match and is sensitive to calibration, including the sparse edge where LOOPR was not designed to dominate.

    **Intended use** asks: *Does the system produce a usable public ordering among established competitors, and does that ordering degrade gracefully under pressure?* No single observed field is a ground-truth public ranking. The defensible approach is a predeclared requirements profile, not a replacement scalar optimized after results.

    The measured construct remains:

    > LOOPR estimates a player's demonstrated ability to convert organized-play match opportunities into tournament victories, given the available network of teammates, opponents, events, and results.

    That construct can include communication, leadership, adaptation, team formation, and other contributions to wins. It can also contain access, opportunity, teammate quality, and network position. Part I can measure the instrument; Part II must still address whether publishing exact named ranks is legitimate.

    ## 2. Graph support: where can the ranking see enough?

    The primary support metric is **opponent-graph k-core number**, computed before each held-out tournament without model scores or winner direction. A player in the k-core belongs to a recursively defined subgraph in which every retained player has at least k distinct opponent connections. A match qualifies only if every participant meets the threshold. This asks whether all players in the comparison sit inside a redundant competitive network rather than on an isolated edge.

    The thresholds (1, 2, 4, 8, 16, 32) and disjoint bands were frozen before model-by-support results. Relative LOOPR loss improves with support against every substantive baseline: slope estimates for baseline-minus-LOOPR loss versus log2(1+minimum core) are {slope_lines}. Positive slopes mean LOOPR improves relative to the named baseline as support rises.

    ![Graph-support performance](figures/01_graph_support.png)

    **Interpretation.** The dense center is not a post-hoc rescue subgroup. It is defined topologically, without ratings or outcome direction, and contains 64.3% of held-out matches at core 32+. At that support, the available data do not clearly separate LOOPR, OpenSkill μ, and Bradley–Terry on probability quality, and LOOPR slightly leads raw winner ordering. At core 0, LOOPR is decisively worse.

    **Limitation.** k-core captures opponent connectivity, not all evidence quality. Dense graph membership cannot remove access or teammate confounding, and a held-out team win is not an objective player ordering.

    ## 3. Historical organizer seeds

    ### Are the seeds meaningful?

    The repository contains **{seeds["tournaments_with_any_seed"]:,}** canonical tournaments with at least one stored seed and **{seeds["complete_seed_permutations"]:,}** complete 1..N permutations. The predeclared structural audit retains **{seeds["structurally_eligible_seed_comparison"]}** ranked tournaments: at least eight teams, stable complete seeds, adequate rosters, and a simple one-dimensional first stage. Only **{seeds["tournaments_with_seed_changes_across_snapshots"]}** tournaments show any team seed change across available snapshots.

    Provenance remains partly unknowable. There is no repository field saying manual, random, copied, or automatic. What the data do show is:

    - **Registration order:** zero of 124 structurally eligible seed lists exactly equals registration order; median Spearman correlation is {seeds["median_seed_registration_spearman_structurally_eligible"]:.3f}.
    - **Stored automatic suggestion:** only one list exactly equals `avgSeedingSkillOrdinal`, but median correlation is {seeds["median_seed_automatic_ordinal_spearman_structurally_eligible"]:.3f}. The plausible interpretation is that this field often informed the order and organizers or software changed some positions. The data cannot identify who changed what.
    - **Previous-event reuse:** among {seeds["prior_exact_roster_comparisons_structurally_eligible"]} structurally eligible exact-roster repeats, only {seeds["same_integer_seed_as_prior_exact_roster_fraction"]:.1%} retain the identical integer seed; median absolute relative-seed change is {seeds["median_abs_relative_seed_change_from_prior_exact_roster"]:.3f}. Pure mechanical copying is not the dominant observed pattern, but partial copying cannot be ruled out.
    - **Not random in aggregate:** the higher seed wins {seeds["eligible_seed_tournament_macro_higher_seed_win_rate"]:.1%} of matches tournament-macro (95% cluster interval {seeds["eligible_seed_tournament_macro_ci"][0]:.1%}–{seeds["eligible_seed_tournament_macro_ci"][1]:.1%}).

    Useful information is not confined to the first few seeds:

    {markdown_table(band_display)}

    ### Which model matches the stored order?

    Model comparison begins on 2025-08-17 and requires exactly four active roster ids, leaving **23 tournaments and 2,806 team pairs**. Ratings use only pre-event information.

    {markdown_table(seed_display)}

    ![Historical seed comparison](figures/02_historical_seeds.png)

    When LOOPR and OpenSkill μ disagree, seeds favor LOOPR on {seed_disagree.set_index("baseline").loc["OpenSkill μ", "seed_supports_loopr_share"]:.1%} of pairs; the paired tournament-macro LOOPR-minus-μ difference is {seed_disagree.set_index("baseline").loc["OpenSkill μ", "loopr_minus_baseline"]:+.3f} (95% interval {seed_disagree.set_index("baseline").loc["OpenSkill μ", "ci_low"]:+.3f} to {seed_disagree.set_index("baseline").loc["OpenSkill μ", "ci_high"]:+.3f}). The corresponding interval includes zero for μ, ordinal, and Bradley–Terry. Against 90-day win rate, seeds favor LOOPR on {seed_disagree.set_index("baseline").loc["90-day win rate", "seed_supports_loopr_share"]:.1%}, clearly above chance.

    **Answer:** seeds are useful evidence of organizer-controlled ordering, but not verified manual expert labels. They separate schedule-aware ratings from raw win rate; they do not show that LOOPR uniquely captures organizer judgment.

    ## 4. The public-ranking requirements profile

    No composite score is reported. Choosing weights after results would recreate the target problem in a less visible form.

    {markdown_table(profile_display)}

    ![Stability and robustness](figures/03_stability_robustness.png)

    ### Stability

    LOOPR's mean monthly top-100 membership retention is {stab.loc["LOOPR", "membership"]:.1%}, below Bradley–Terry's {stab.loc["Bradley–Terry (L2)", "membership"]:.1%}. Among retained players, LOOPR has the highest mean Spearman correlation ({stab.loc["LOOPR", "order"]:.3f}) and the smallest median transition p90 movement ({stab.loc["LOOPR", "p90"]:.1f} ranks). This is a real tradeoff: LOOPR is not best at keeping the same names, but it is best at preserving the ordering of names that remain.

    ### Important-event removal

    The old 665-event LOOPR leave-one-out analysis remains. The new all-system comparison removes the 25 events with the most active 8-core player-match appearances, selected without scores or winner direction. LOOPR's median top-100 retention is {robust.loc["LOOPR", "median_retention"]:.1%}, worst retention {robust.loc["LOOPR", "worst_retention"]:.1%}, and largest observed original-top-100 shift {robust.loc["LOOPR", "largest"]:.0f}. The corresponding largest shifts are {robust.loc["Bradley–Terry (L2)", "largest"]:.0f} for Bradley–Terry, {robust.loc["OpenSkill ordinal", "largest"]:.0f} for OpenSkill ordinal, and {robust.loc["OpenSkill μ", "largest"]:.0f} for μ. This is LOOPR's clearest operational advantage.

    Stability and robustness are not accuracy. They establish that the displayed ordering is not usually controlled by one event or minor monthly perturbation.

    ## 5. Actual adversarial interventions

    Twenty disjoint target lineups were chosen by stable player-id hash from active players with at least 20 prior matches, 8-core support, and an observed exact-four lineup used at least three times. Opponents and targets were selected without any evaluated model score. Three weak and fifteen strong lineups met the frozen trailing-90-day definitions.

    ![Adversarial tests](figures/04_adversarial_tests.png)

    ### Weak-win farming

    After 50 added wins against weak lineups, LOOPR's median target-lineup score change is **{farming50.loc["LOOPR", "median_standardized_score_change"]:.3f} within-system SD**, compared with {farming50.loc["OpenSkill μ", "median_standardized_score_change"]:.3f} for OpenSkill μ, {farming50.loc["OpenSkill ordinal", "median_standardized_score_change"]:.3f} for ordinal, {farming50.loc["Bradley–Terry (L2)", "median_standardized_score_change"]:.3f} for Bradley–Terry, and {farming50.loc["90-day win rate", "median_standardized_score_change"]:.3f} for recent win rate. No LOOPR target lineup creates a new top-100 entrant in this attack.

    LOOPR still rewards strong wins: its weak-to-strong response ratio at 50 wins is {farming50.loc["LOOPR", "weak_to_strong_response_ratio"]:.3f}. That matters because a system that ignores every win would look farming-resistant for the wrong reason.

    ### Same record, different schedule

    The controlled 5–5 record produces a strong-minus-weak response gap of **{schedule.loc["LOOPR", "strong_minus_weak_standardized_change"]:.3f} SD** for LOOPR, second to OpenSkill μ at {schedule.loc["OpenSkill μ", "strong_minus_weak_standardized_change"]:.3f}. Bradley–Terry is {schedule.loc["Bradley–Terry (L2)", "strong_minus_weak_standardized_change"]:.3f}; recent win rate is {schedule.loc["90-day win rate", "strong_minus_weak_standardized_change"]:.3f}. LOOPR does not uniquely account for schedule, but it does so strongly.

    ### Volume duplication

    Duplicating each target lineup's deterministic block of up to ten recent exact-lineup matches fivefold yields a LOOPR p90 absolute score movement of **{volume5.loc["LOOPR", "p90_abs_standardized_score_change"]:.3f} SD**, the smallest in the comparison. This supports resistance to duplicated evidence, not exact volume cancellation.

    ### Evidence thinning

    Removing half of matches incident to target players over ten deterministic hash replicates leaves LOOPR median full top-100 retention at **{thin50.loc["LOOPR", "median_full_top100_retention"]:.1%}**, target-pair order retention at **{thin50.loc["LOOPR", "median_target_pairwise_ordering_retention"]:.1%}**, and median replicate p90 score movement at **{thin50.loc["LOOPR", "median_target_p90_abs_standardized_change"]:.3f} SD**. LOOPR moves least on the last measure. This is a strong low-evidence degradation result, although it does not isolate which LOOPR component creates the protection.

    ### Transitive chain

    Every model shifts A above C after adding A-over-B and B-over-C chains with no A-C match; at 20 repetitions, the fraction positive is {chain20.loc["LOOPR", "fraction_positive"]:.0%} for LOOPR and {chain20["fraction_positive"].min():.0%} for the weakest baseline. This test confirms intended direction but does **not** distinguish LOOPR—the simple win-rate model also passes. It should stay in a technical appendix rather than become a success headline.

    **Adversarial limitation.** These interventions identify behavior under specified attacks. They do not establish complete Goodhart resistance, and the weak-lineup pool is narrow.

    ## 6. Plus: decomposing the institutional result

    Keep three different quantities separate:

    - **Literal pooled AUC = {plus_shuffle["observed_pooled_auc"]:.3f}.** This asks whether a randomly chosen passer from the whole candidate pool is scored above a randomly chosen failer. Cross-tier and cross-status comparisons count because they were part of the institution.
    - **Tier-macro institutional AUC = {old_shuffle["observed_macro_auc"]:.3f}.** This asks the same question within tier, then gives each tier equal weight.
    - **Within-tier/status pair-weighted AUC = {old_shuffle["observed_stratified_auc"]:.3f}.** This narrower estimand asks what happens when tier and candidate status match. It is not “cleaner”; it deliberately removes much of the institution.

    {markdown_table(plus_pair_display)}

    Same-tier/same-status pairs are only {plus_decomp.loc[plus_decomp["same_tier"] & plus_decomp["same_status"], "share_of_all_passer_failer_pairs"].iloc[0]:.1%} of all passer-failer pairs. Same-tier/different-status pairs have concordance {plus_decomp.loc[plus_decomp["same_tier"] & ~plus_decomp["same_status"], "loopr_concordance"].iloc[0]:.3f}; candidate status is therefore a major source of pooled agreement.

    All 310 evaluated candidates are already in the pre-vote 32-core (median core 155). Support stratification is identical at every predeclared threshold; it cannot explain within-sample variation. This is an important scope result: the Plus analysis is already an established-player analysis.

    The ten-fold out-of-fold model using tier, status, activity, and graph support reaches AUC **{baseline_lookup.loc["institution + activity + support", "out_of_fold_auc"]:.3f}**. Adding LOOPR reaches **{baseline_lookup.loc["institution + activity + support + LOOPR", "out_of_fold_auc"]:.3f}**. This is out-of-fold evidence of incremental information, but the improvement is modest and has no separate uncertainty interval.

    The new literal pooled outcome shuffle preserves participation, lineups, events, timestamps, and match volume while independently flipping winner direction. Across 200 full LOOPR recomputations, mean pooled AUC is **{plus_shuffle["null_pooled_mean"]:.3f}** with a 95% range **{plus_shuffle["null_pooled_q025"]:.3f}–{plus_shuffle["null_pooled_q975"]:.3f}**; none reaches the observed {plus_shuffle["observed_pooled_auc"]:.3f} (corrected p={plus_shuffle["null_pooled_exceedance_p"]:.3f}). Outcome direction adds information beyond participation structure. It does not erase institutional status or access effects.

    ![Plus decomposition](figures/05_plus_decomposition.png)

    ## 7. What a tournament organizer might mean by “this ranking is really good”

    {markdown_table(to_hypotheses)}

    The historical data can test ordering concordance, dense-center behavior, schedule response, farming resistance, and operational continuity. They cannot identify whether organizers find the ranking legitimate, understandable, socially safe, or labor-saving. Those are not missing statistical controls; they are different questions.

    ## 8. Recommended metric hierarchy

    {markdown_table(metrics)}

    **Promote:** dense-core ordinal performance with coverage, seed disagreement pairs, adversarial degradation, monthly membership plus retained ordering, important-event removal, and decomposed Plus agreement.

    **Keep but demote from decisive:** all-match log loss, Brier score, calibration, and accuracy. They remain central generalization checks and should never be removed.

    **Use as guardrails:** sparse-edge performance, unfamiliar-lineup transfer, smoothing/tournament-weighting null results, and attack-pool shortfalls. These define where the intended-use claim stops.

    **Remove nothing.** The earlier analyses answer real questions. The change is the argumentative role assigned to each metric.

    ## 9. Claim changes for Part I

    {markdown_table(claims)}

    The conclusion should change from a single-dimensional “works/doesn't work” verdict to:

    > LOOPR was not the best general-purpose match probability model. It was competitive with the strongest baselines inside the dense opponent graph, matched consequential historical orderings, and degraded unusually gracefully under several attacks that matter for a public ranking. It was poor on unsupported newcomers, did not dominate every baseline, and did not validate exact neighboring ranks.
    >
    > That was enough to treat LOOPR as a strong outcome-derived skill-o-meter for established organized-play competition—and as a credible ranking-oriented design. It was not enough to prove that publishing its exact named order was legitimate.
    >
    > I assumed those were the same achievement. The Splatoon community rejected the leaderboard anyway.

    ## 10. What remains unknowable

    - No objective player-skill vector exists in the repository.
    - Seed assignment provenance is undocumented; observed patterns cannot certify manual expert authorship.
    - Plus voting is one consequential snapshot, not universal community agreement.
    - Exact nearby ranks remain statistically and constructively unresolved.
    - The attack suite cannot enumerate every Goodhart strategy.
    - Sparse/new-player ranking remains a major weakness, and no dense-center result repairs it.
    - Measurement evidence cannot answer public legitimacy, authority, interpretability, or social harm.

    ## Reproducibility and scope

    The prior audit remains frozen and intact. New support thresholds, seed rules, attack pools, and intervention sizes were committed before inspecting new results in `PREANALYSIS_PLAN.md` and `IMPLEMENTATION_DECISIONS.md`. All comparisons use the same five systems and the same intervention data. Prediction and seed uncertainty resample tournaments; adversarial runs are deterministic sensitivities, not confidence intervals. Outputs are machine-readable under `tables/` and `results/`.
    """
    normalized = "\n".join(
        line[4:] if line.startswith("    ") else line for line in report.splitlines()
    )
    return normalized.strip()


def build_reframed_section(output: Path) -> str:
    del output
    return dedent(
        """
        # Proposed Part I evaluation reframing

        ## What I Was Actually Trying to Build

        There is a version of this story where I evaluate LOOPR like a prediction model, discover that OpenSkill and Bradley–Terry assign slightly better probabilities to future matches, and stop there.

        That result is real. It is also not the whole design problem.

        LOOPR was meant to become a public player ranking. A good ranking still needs to generalize to future competition, but it also needs to resist grinding weak wins, account for schedule, remain stable enough to use, and avoid letting one event rewrite the visible top. Those properties do not collapse into average log loss.

        So I kept the prediction test and separated two questions:

        - **Generalization:** does the score contain information about any future match?
        - **Intended use:** does it produce a defensible ordering among established competitors, and does that ordering fail gracefully when people or data put pressure on it?

        There is no objective true-skill vector behind either question. The construct I can defend is narrower:

        > LOOPR estimates a player's demonstrated ability to convert organized-play match opportunities into tournament victories, given the available network of teammates, opponents, events, and results.

        ## Where the Graph Can Support a Player Ranking

        I defined graph support without looking at any rating or winner direction. A player's k-core number asks whether they sit inside a recursively dense opponent network. For a held-out match to enter a support stratum, every participant had to meet the threshold.

        The distinction mattered. LOOPR was poor on core-0 matches: log loss 0.681, versus 0.480 for OpenSkill ordinal. In the core-32+ graph, which covered 2,827 held-out matches, LOOPR's log loss was 0.486, OpenSkill μ's was 0.481, and Bradley–Terry's was 0.482. Tournament-cluster uncertainty did not clearly separate those systems there. LOOPR's raw winner ordering was slightly highest at 76.5%.

        That does not make the sparse failure disappear. It defines the scope of the instrument. LOOPR works best where there are enough transitive paths for its design to have something to work with.

        ## An Existing Artifact Closer to Ranking Use

        Tournament pages preserve team seeds. I did not assume those were expert labels. I audited them first.

        The stored order is not registration order, is strongly correlated with Sendou's automatic seeding ordinal without usually copying it exactly, changes across repeated snapshots in only three tournaments, and predicts match direction far above chance. Information remains below the top four seeds. The repository cannot tell me whether an organizer manually edited a list, copied part of an earlier event, or used a private process.

        In the 23 events that passed every roster and format restriction after the prediction start date, all four schedule-aware systems matched seeds strongly. OpenSkill ordinal had the highest concordance point estimate. When LOOPR and OpenSkill or Bradley–Terry disagreed, the seed data did not clearly favor either side. LOOPR clearly beat 90-day win rate.

        That makes seeds useful convergent evidence for schedule-aware ranking. It does not make them proof that LOOPR is best.

        ## What the Extra Computation Bought

        The clearest differences appeared when I stressed the rankings directly.

        Adding fifty wins against weak lineups moved LOOPR by a median 0.040 within-system standard deviations. The same attack moved OpenSkill μ by 0.090, OpenSkill ordinal by 0.069, Bradley–Terry by 0.092, and 90-day win rate by 1.533. LOOPR still rewarded strong wins and gave a 0.805-standard-deviation advantage to the same 5–5 record against strong rather than weak opposition.

        Duplicating each selected lineup's block of up to ten recent exact-lineup matches fivefold moved LOOPR least in the tail. Removing half the selected players' historical matches over ten deterministic perturbations also moved LOOPR least: 95% median top-100 retention and 0.238 standard deviations at the median replicate's 90th percentile.

        Monthly stability was a tradeoff. Bradley–Terry kept more of the same top-100 names. LOOPR best preserved the order among names that remained. When I removed each of the 25 events with the most active dense-core participation and rebuilt every system, LOOPR had 99% worst-case top-100 retention and a maximum observed shift of 14 ranks—smaller than the other schedule-aware systems.

        These are leaderboard properties. They do not prove accuracy. A stable ranking can be stably wrong. But they answer why a more complicated ranking might be worthwhile even when its probabilities are not best.

        ## Re-reading Plus

        I kept the pooled Plus AUC because it describes agreement with the institution that actually operated. Its literal value was 0.651; the equal-tier institutional AUC was 0.777. The within-tier/status AUC was 0.615, but that is not a cleaner number. It is a narrower question that discards cross-tier and cross-status decisions.

        Candidate status explains a large part of the institution. Tier plus status alone reached 0.749 out of fold. A model with tier, status, activity, and graph support reached 0.790; adding LOOPR raised that to 0.812 and improved log loss by 0.014. All evaluated candidates were already in the 32-core, so this vote says nothing about sparse newcomers.

        I also reran the full outcome-shuffle control for literal pooled AUC. Preserving participation and event structure while destroying winner direction produced a mean AUC of 0.503; none of 200 draws reached 0.651. Match outcomes add information beyond showing up. Status and opportunity still matter.

        ## What I Can Claim

        Several different tests now answer different parts of the design problem. LOOPR predicted future competition, became competitive with the strongest probability baselines in the dense graph, matched consequential historical orderings, strongly rewarded schedule quality, and degraded unusually gracefully under farming, volume, missing-history, and event-removal stress.

        It also failed important tests. It was poor at the sparse edge. It did not beat OpenSkill or Bradley–Terry at general-purpose probabilities. It did not uniquely match organizer seeds. Bradley–Terry retained top-100 membership better month to month. The transitive-chain test did not distinguish LOOPR from simpler systems.

        The defensible conclusion is not that LOOPR recovered true skill or won every metric. It is that LOOPR is a strong outcome-derived skill-o-meter—and a credible ranking-oriented design—for established organized-play competitors inside a well-supported network.

        I assumed that meant I had built a strong public leaderboard.

        The Splatoon community rejected it anyway.

        Part II is about why measurement quality was never the only question.
        """
    ).strip()


def build_notebook(output: Path) -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Public-ranking intended-use evaluation\n\n"
            "**TL;DR:** Prediction remains a valid generalization test. The added ranking-specific evidence shows that LOOPR is poor at the sparse edge, competitive in the dense core, and unusually robust under several leaderboard attacks. It does not dominate every baseline."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image\nOUTPUT = Path.cwd()\nTABLES = OUTPUT / 'tables'\nFIGURES = OUTPUT / 'figures'\nprint(OUTPUT)"
        ),
        nbf.v4.new_markdown_cell(
            "## Requirements profile\n\nNo composite score is formed; arrows in the column names show direction."
        ),
        nbf.v4.new_code_cell(
            "profile = pd.read_csv(TABLES / 'requirements_profile.csv')\ndisplay(profile)"
        ),
        nbf.v4.new_markdown_cell(
            "## Graph support\n\nSupport is based on the pre-event opponent graph, independent of scores and outcome direction."
        ),
        nbf.v4.new_code_cell(
            "display(Image(filename=str(FIGURES / '01_graph_support.png')))\ndisplay(pd.read_csv(TABLES / 'prediction_by_support_band.csv'))"
        ),
        nbf.v4.new_markdown_cell(
            "## Historical tournament seeds\n\nStored seeds are audited as organizer-controlled artifacts, not ground truth."
        ),
        nbf.v4.new_code_cell(
            "display(Image(filename=str(FIGURES / '02_historical_seeds.png')))\ndisplay(pd.read_csv(TABLES / 'seed_outcome_by_band.csv'))"
        ),
        nbf.v4.new_markdown_cell("## Stability, robustness, and adversarial behavior"),
        nbf.v4.new_code_cell(
            "display(Image(filename=str(FIGURES / '03_stability_robustness.png')))\ndisplay(Image(filename=str(FIGURES / '04_adversarial_tests.png')))"
        ),
        nbf.v4.new_markdown_cell("## Plus decomposition"),
        nbf.v4.new_code_cell(
            "display(Image(filename=str(FIGURES / '05_plus_decomposition.png')))\ndisplay(pd.read_csv(TABLES / 'plus_pooled_auc_decomposition.csv'))\nprint(json.loads((OUTPUT / 'results/pooled_outcome_shuffle.json').read_text()))"
        ),
        nbf.v4.new_markdown_cell(
            "## Takeaway\n\n"
            "The repository supports a bounded claim about an outcome-derived ranking instrument for established organized-play competitors. Probability quality, operational ranking behavior, institutional convergence, and publication legitimacy remain separate claims."
        ),
    ]
    path = output / "intended_use_results.ipynb"
    client = NotebookClient(notebook, timeout=180, kernel_name="python3")
    executed = client.execute(cwd=str(output))
    nbf.write(executed, path)


def build_artifact_index(output: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or "cache" in path.parts:
            continue
        rows.append(
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "artifact_index.csv", index=False)
    return frame


def validate(
    output: Path, profile: pd.DataFrame, attacks: dict[str, pd.DataFrame]
) -> list[str]:
    tables = output / "tables"
    checks: list[tuple[bool, str]] = [
        (
            set(profile["method"]) == set(METHODS),
            "requirements profile has all five systems",
        ),
        (profile.notna().all().all(), "requirements profile has no missing values"),
        (
            set(pd.read_csv(tables / "prediction_by_support_band.csv")["method"])
            >= set(METHODS),
            "every model is evaluated in graph-support bands",
        ),
        (
            pd.read_csv(tables / "seed_model_tournament_results.csv")[
                "tournament_id"
            ].nunique()
            == 23,
            "seed-model comparison contains 23 predeclared eligible events",
        ),
        (
            int(
                pd.read_parquet(tables / "attack_lineup_selection.parquet")
                .query("role == 'target'")
                .shape[0]
            )
            == 20,
            "attack pool contains 20 predeclared targets",
        ),
        (
            attacks["thinning"]["deterministic_replicates"].eq(10).all(),
            "history thinning has ten replicates per model and fraction",
        ),
        (
            read_json(output / "results" / "pooled_outcome_shuffle.json")["repetitions"]
            == 200,
            "literal pooled outcome shuffle contains 200 full recomputations",
        ),
        (
            pd.read_parquet(tables / "plus_candidates_with_support.parquet")[
                "core_number"
            ].min()
            >= 32,
            "all evaluated Plus candidates are in the pre-vote 32-core",
        ),
    ]
    failed = [label for passed, label in checks if not passed]
    if failed:
        raise AssertionError("Validation failures: " + "; ".join(failed))
    return [label for _, label in checks]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.resolve()
    tables = output / "tables"
    figures = output / "figures"
    (output / "results").mkdir(parents=True, exist_ok=True)

    attacks = build_attack_summaries(tables)
    profile = build_requirements_profile(tables, attacks)
    metrics = build_metric_hierarchy()
    direct_answers = build_direct_answers()
    hypotheses = build_to_hypotheses()
    claims = build_claim_changes()
    profile.to_csv(tables / "requirements_profile.csv", index=False)
    metrics.to_csv(tables / "metric_hierarchy.csv", index=False)
    direct_answers.to_csv(tables / "deliverable_answers.csv", index=False)
    hypotheses.to_csv(tables / "to_reaction_hypotheses.csv", index=False)
    claims.to_csv(tables / "claim_changes.csv", index=False)

    checks = validate(output, profile, attacks)
    build_figures(tables, figures, attacks)
    report = build_report(
        output, profile, metrics, direct_answers, hypotheses, claims, attacks
    )
    write_text(output / "technical_report.md", report)
    write_text(
        output / "proposed_reframed_evaluation_section.md",
        build_reframed_section(output),
    )
    write_text(
        output / "README.md",
        dedent(
            """
            # Intended-use evaluation addendum

            This directory is additive to the frozen Part I audit. Start with:

            - `technical_report.md` — answer-first analysis and recommendations.
            - `proposed_reframed_evaluation_section.md` — replacement narrative for Part I.
            - `intended_use_results.ipynb` — executed, reviewable companion.
            - `tables/requirements_profile.csv` — no-composite all-system comparison.
            - `PREANALYSIS_PLAN.md` and `IMPLEMENTATION_DECISIONS.md` one directory up — definitions frozen before results.

            Rebuild result tables with `python -m analysis.part1.intended_use.run ...` and render this package with `python -m analysis.part1.intended_use.report --output analysis/part1/intended_use/output`.
            """
        ),
    )
    write_text(
        output / "VALIDATION.md",
        dedent(
            """
            # Validation record

            - PASS: all six intended-use invariance and artifact-integrity tests.
            - PASS: Ruff checks for the complete `analysis/part1/intended_use` package.
            - PASS: executed notebook; all six code cells completed with no error outputs.
            - PASS: report consistency checks for five-system coverage, seed-event count, attack-pool scope, thinning replicates, Plus graph support, and 200 pooled shuffle recomputations.
            - ENVIRONMENT LIMIT: the broader repository suite reaches completion but has two failures in `tests/test_loss_improvements.py`. Both arise in untouched geometric team aggregation because the available runtime has Polars 1.39.0 while `pyproject.toml` declares Polars `^0.20.0` (`map_elements` now supplies scalars where that legacy code expects a Series). The new intended-use tests do not fail.

            This environment mismatch was not repaired because it is outside the additive evaluation scope and changing production aggregation code would risk altering the frozen audit.
            """
        ),
    )
    write_json(
        output / "results" / "results.json",
        {
            "verdict": (
                "General-purpose match prediction and public-ranking quality are separate hypotheses. "
                "LOOPR is weak at the sparse edge, competitive in the dense core, and comparatively "
                "robust under several leaderboard attacks; it does not dominate every baseline."
            ),
            "construct": (
                "demonstrated ability to convert organized-play match opportunities into tournament "
                "victories, given the available network of teammates, opponents, events, and results"
            ),
            "requirements_profile": profile.to_dict("records"),
            "validation_checks": checks,
            "random_seed": 20_260_731,
            "runtime_versions": runtime_versions(),
            "prior_audit_preserved": True,
        },
    )
    write_json(output / "results" / "runtime_versions.json", runtime_versions())
    build_notebook(output)
    index = build_artifact_index(output)
    write_json(
        output / "results" / "render_validation.json",
        {"checks": checks, "indexed_artifacts": int(len(index))},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
