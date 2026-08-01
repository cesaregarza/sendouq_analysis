#!/usr/bin/env python3
"""Freeze the Part I audit into reader-facing and machine-readable artifacts.

This module does not refit ratings.  It consumes the saved, auditable outputs
from :mod:`analysis.part1.run`, performs consistency checks, and renders the
small main-path package plus technical appendices.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from textwrap import dedent
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import nbformat as nbf
import numpy as np
import pandas as pd


SEED = 20_260_731
BLUE = "#2563EB"
GREEN = "#059669"
ORANGE = "#D97706"
RED = "#DC2626"
PURPLE = "#7C3AED"
GRAY = "#64748B"
LIGHT = "#E2E8F0"
INK = "#0F172A"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small dataframe without requiring tabulate."""

    display = frame.fillna("—").astype(str)
    header = "| " + " | ".join(display.columns) + " |"
    rule = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(row.replace("\n", " ") for row in values) + " |"
        for values in display.to_numpy().tolist()
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
            "grid.color": "#E2E8F0",
            "grid.linewidth": 0.7,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": "#334155",
            "ytick.color": "#334155",
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def normalize_loto(output: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Upgrade the first corrected LOTO labels without refitting 665 models."""

    table_path = output / "tables" / "leave_one_tournament_out.csv"
    result_path = output / "results" / "leave_one_tournament_out.json"
    frame = pd.read_csv(table_path)
    frame = frame.rename(
        columns={
            "original_top100_became_unrankable": "original_top100_no_longer_rank_eligible",
            "highest_ranked_disappeared_player_id": "highest_ranked_newly_ineligible_player_id",
            "highest_ranked_disappeared_full_rank": "highest_ranked_newly_ineligible_full_rank",
            "highest_ranked_disappeared_full_exposure": "highest_ranked_newly_ineligible_full_exposure",
            "highest_ranked_disappeared_full_matches": "highest_ranked_newly_ineligible_full_matches",
        }
    )
    frame.to_csv(table_path, index=False)

    summary = read_json(result_path)
    key_map = {
        "events_making_at_least_one_original_top100_player_unrankable": "events_changing_rank_eligibility_for_at_least_one_original_top100_player",
        "maximum_original_top100_players_made_unrankable": "maximum_original_top100_players_made_rank_ineligible",
        "unrankable_is_not_rank_movement": "rank_eligibility_is_not_rank_movement",
    }
    for old, new in key_map.items():
        if old in summary:
            summary[new] = summary.pop(old)
    summary["rank_eligibility_is_not_rank_movement"] = (
        "Players absent from the reduced active/ranked population are counted "
        "separately, not assigned a fictitious bottom rank; absence can reflect "
        "the 90-day activity rule rather than disappearance from the rating graph."
    )
    write_json(result_path, summary)
    return frame, summary


def merge_half_lives(output: Path) -> pd.DataFrame:
    labels = ["90", "120", "180", "270", "365", "no_decay"]
    paths = [output / "tables" / f"half_life_{label}.csv" for label in labels]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing half-life rows: {missing}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    frame.to_csv(output / "tables" / "half_life_tradeoff.csv", index=False)
    write_json(output / "results" / "half_life.json", frame.to_dict("records"))
    return frame


def consistency_checks(
    manifest: dict[str, Any],
    prediction: pd.DataFrame,
    plus_methods: pd.DataFrame,
    half_life: pd.DataFrame,
    loto: pd.DataFrame,
) -> list[str]:
    checks: list[tuple[bool, str]] = [
        (
            manifest["dataset"]["tournaments"] == 665,
            "manifest has 665 eligible tournaments",
        ),
        (manifest["dataset"]["matches"] == 28_647, "manifest has 28,647 matches"),
        (
            prediction["matches"].nunique() == 1,
            "prediction methods use identical match counts",
        ),
        (
            prediction["tournaments"].nunique() == 1,
            "prediction methods use identical tournament counts",
        ),
        (
            plus_methods["n"].nunique() == 1,
            "Plus methods use identical candidate counts",
        ),
        (
            len(loto) == manifest["dataset"]["tournaments"],
            "LOTO removed every eligible tournament",
        ),
    ]
    loopr = prediction.loc[prediction["method"] == "LOOPR"].iloc[0]
    sweep_180 = half_life.loc[half_life["setting"] == "180d"].iloc[0]
    checks.append(
        (
            abs(loopr["match_micro_log_loss"] - sweep_180["match_micro_log_loss"])
            < 1e-12,
            "180-day sweep reproduces primary LOOPR micro log loss",
        )
    )
    checks.append(
        (
            abs(
                loopr["tournament_macro_log_loss"]
                - sweep_180["tournament_macro_log_loss"]
            )
            < 1e-12,
            "180-day sweep reproduces primary LOOPR macro log loss",
        )
    )
    failed = [label for passed, label in checks if not passed]
    if failed:
        raise AssertionError("Consistency failures: " + "; ".join(failed))
    return [label for _, label in checks]


def build_main_tables(output: Path, data: dict[str, Any]) -> dict[str, pd.DataFrame]:
    prediction = data["prediction"]
    plus = data["plus_methods"]
    stability = data["stability"]
    loto = data["loto"]

    can_see = pd.DataFrame(
        [
            [
                "Largest weak component",
                "5,518 / 5,518 players (100%)",
                "The organized-play graph is connected enough for results to propagate.",
            ],
            [
                "Top 100 in that component",
                "100 / 100",
                "Every displayed top player is connected to the same result network.",
            ],
            [
                "Plus candidates in graph",
                "459 / 468 (98.1%)",
                "Nine historical candidates never appear in a resolved eligible match.",
            ],
            [
                "Unique teammates",
                "median 6; IQR 3–14",
                "The typical player changes teammates, but mixing varies greatly.",
            ],
            [
                "Unique opponents",
                "median 48; IQR 20–151",
                "Most players connect to many opponents.",
            ],
            [
                "Distinct lineups",
                "median 3; IQR 1–8",
                "Many players supply more than one lineup context.",
            ],
            [
                "Most-common-lineup share",
                "median 53.8%; IQR 28.3–100%",
                "A quarter of players never leave their most common recorded lineup.",
            ],
            [
                "Persistent lineup",
                "841 players (15.2%)",
                "At least 80% of matches use one lineup (flag requires ≥5 matches).",
            ],
            [
                "Near-inseparable teammate",
                "1,746 players (31.6%)",
                "One teammate appears in at least 90% of matches (flag requires ≥5 matches).",
            ],
        ],
        columns=["Measure", "Result", "What it means"],
    )

    pred_main = prediction[
        [
            "method",
            "match_micro_log_loss",
            "tournament_macro_log_loss",
            "brier_score",
            "accuracy",
            "paired_log_loss_difference_vs_loopr",
            "paired_difference_ci_low",
            "paired_difference_ci_high",
            "tournaments_outperforming_loopr",
            "median_per_tournament_difference",
        ]
    ].copy()

    plus_main = plus[
        plus["method"].isin(
            [
                "LOOPR",
                "90-day win rate",
                "Matches in prior 90d",
                "Tournaments in prior 90d",
            ]
        )
    ][
        [
            "method",
            "n",
            "institutional_macro_auc",
            "institutional_ci_low",
            "institutional_ci_high",
            "like_for_like_stratified_auc",
            "stratified_ci_low",
            "stratified_ci_high",
        ]
    ].copy()

    stability_main = stability[
        [
            "month_a",
            "month_b",
            "top100_retained",
            "spearman_on_retained",
            "median_absolute_rank_movement",
            "p90_absolute_rank_movement",
        ]
    ].copy()

    loto_main = pd.DataFrame(
        [
            ["Median top-100 retention", f"{loto['median_top100_retention']:.0%}"],
            [
                "Worst top-100 retention",
                f"{loto['worst_top100_retention']:.0%} ({loto['worst_retention_tournament_name']})",
            ],
            [
                "Median of per-event median movements",
                f"{loto['median_of_median_abs_rank_changes']:.0f} places",
            ],
            [
                "90th percentile of per-event 90th-percentile movement",
                f"{loto['p90_of_all_tournament_p90_rank_changes']:.0f} place",
            ],
            [
                "Largest shared-player shift",
                f"{loto['largest_observed_shared_player_rank_shift']:.0f} places ({loto['largest_shared_shift_tournament_name']})",
            ],
            [
                "Events changing top-100 rank eligibility",
                f"{loto['events_changing_rank_eligibility_for_at_least_one_original_top100_player']} / {loto['eligible_tournaments_removed']}",
            ],
        ],
        columns=["Measure", "Result"],
    )

    tables = {
        "what_the_dataset_can_see": can_see,
        "future_prediction": pred_main,
        "plus_agreement": plus_main,
        "monthly_stability": stability_main,
        "loto_robustness": loto_main,
    }
    for name, frame in tables.items():
        frame.to_csv(output / "tables" / f"main_{name}.csv", index=False)
        write_text(output / "tables" / f"main_{name}.md", markdown_table(frame))
    return tables


def plot_prediction(output: Path, prediction: pd.DataFrame) -> None:
    frame = prediction.sort_values("match_micro_log_loss", ascending=True).copy()
    colors = [
        BLUE
        if method == "LOOPR"
        else GREEN
        if method in {"OpenSkill ordinal", "OpenSkill μ", "Bradley–Terry (L2)"}
        else GRAY
        for method in frame["method"]
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    y = np.arange(len(frame))
    ax.hlines(y, 0.47, frame["match_micro_log_loss"], color=LIGHT, linewidth=3)
    ax.scatter(frame["match_micro_log_loss"], y, color=colors, s=95, zorder=3)
    ax.set_yticks(y, frame["method"])
    ax.set_xlim(0.47, 0.72)
    ax.set_xlabel("Held-out match log loss (lower is better)")
    ax.set_title(
        "Future tournament prediction: LOOPR is informative, but not the best predictor"
    )
    ax.grid(axis="x")
    for yi, value in zip(y, frame["match_micro_log_loss"]):
        ax.text(value + 0.004, yi, f"{value:.3f}", va="center", fontsize=9)
    fig.text(
        0.5,
        0.015,
        "4,397 matches in 110 tournaments; ratings rebuilt before each event; prior-only calibration",
        ha="center",
        color=GRAY,
        fontsize=8.5,
    )
    fig.subplots_adjust(bottom=0.16)
    save_figure(fig, output / "figures" / "01_future_prediction")


def plot_calibration(output: Path, calibration: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color=GRAY, label="Perfect calibration")
    ax.plot(
        calibration["mean_predicted_probability"],
        calibration["observed_win_rate"],
        marker="o",
        color=BLUE,
        linewidth=2,
        markersize=7,
        label="LOOPR",
    )
    for row in calibration.itertuples(index=False):
        ax.annotate(
            f"n={int(row.n_matches)}",
            (row.mean_predicted_probability, row.observed_win_rate),
            xytext=(5, -13),
            textcoords="offset points",
            fontsize=8,
            color=GRAY,
        )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean predicted chance that team 1 wins",
        ylabel="Observed team 1 win rate",
    )
    ax.set_title("LOOPR reliability across five equal-count bins")
    ax.grid()
    ax.legend(frameon=False, loc="upper left")
    save_figure(fig, output / "figures" / "02_loopr_calibration")


def plot_plus(
    output: Path,
    methods: pd.DataFrame,
    shuffle: pd.DataFrame,
    shuffle_summary: dict[str, Any],
) -> None:
    selected = methods[
        methods["method"].isin(
            ["LOOPR", "Matches in prior 90d", "Tournaments in prior 90d"]
        )
    ].copy()
    order = ["LOOPR", "Matches in prior 90d", "Tournaments in prior 90d"]
    selected["order"] = selected["method"].map(
        {name: i for i, name in enumerate(order)}
    )
    selected = selected.sort_values("order")
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.7), gridspec_kw={"width_ratios": [1.05, 1.25]}
    )
    ax = axes[0]
    x = np.arange(len(selected))
    width = 0.34
    ax.bar(
        x - width / 2,
        selected["institutional_macro_auc"],
        width,
        color=BLUE,
        label="Within tier",
    )
    ax.bar(
        x + width / 2,
        selected["like_for_like_stratified_auc"],
        width,
        color=ORANGE,
        label="Within tier + status",
    )
    ax.axhline(0.5, linestyle="--", color=GRAY, linewidth=1)
    ax.set_xticks(x, ["LOOPR", "Recent\nmatches", "Recent\nevents"])
    ax.set_ylim(0.35, 0.86)
    ax.set_ylabel("ROC AUC")
    ax.set_title("Agreement with the November 2025 Plus vote")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)

    ax = axes[1]
    bins = np.linspace(0.34, 0.66, 22)
    ax.hist(
        shuffle["macro_auc"],
        bins=bins,
        alpha=0.62,
        color=BLUE,
        label="Shuffled: within tier",
    )
    ax.hist(
        shuffle["stratified_auc"],
        bins=bins,
        alpha=0.55,
        color=ORANGE,
        label="Shuffled: tier + status",
    )
    ax.axvline(
        shuffle_summary["observed_macro_auc"],
        color=BLUE,
        linewidth=2.3,
        label=f"Observed within tier: {shuffle_summary['observed_macro_auc']:.3f}",
    )
    ax.axvline(
        shuffle_summary["observed_stratified_auc"],
        color=ORANGE,
        linewidth=2.3,
        label=f"Observed tier + status: {shuffle_summary['observed_stratified_auc']:.3f}",
    )
    ax.set_xlabel("AUC after recomputing LOOPR")
    ax.set_xlim(0.34, 0.80)
    ax.set_ylabel("Shuffle repetitions")
    ax.set_title("Real outcome direction exceeds the participation-only null")
    ax.grid(axis="y")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle(
        "Plus convergence is real but weaker under like-for-like comparison",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, output / "figures" / "03_plus_and_shuffle")


def plot_stability(output: Path, stability: pd.DataFrame) -> None:
    labels = [
        f"{a[5:7]}/{a[2:4]}→{b[5:7]}/{b[2:4]}"
        for a, b in zip(stability["month_a"], stability["month_b"])
    ]
    x = np.arange(len(stability))
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 6.4), sharex=True)
    axes[0].bar(x, stability["top100_retention"], color=BLUE)
    axes[0].axhline(
        stability["top100_retention"].mean(),
        color=ORANGE,
        linestyle="--",
        label=f"Mean {stability['top100_retention'].mean():.1%}",
    )
    axes[0].set_ylim(0.5, 1.02)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_ylabel("Top-100 retained")
    axes[0].set_title(
        "Monthly stability: retained ordering is high, but membership can churn"
    )
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y")
    axes[1].plot(
        x,
        stability["median_absolute_rank_movement"],
        marker="o",
        color=GREEN,
        label="Median movement",
    )
    axes[1].plot(
        x,
        stability["p90_absolute_rank_movement"],
        marker="o",
        color=PURPLE,
        label="90th percentile",
    )
    axes[1].set_ylabel("Absolute rank movement")
    axes[1].set_xticks(x, labels, rotation=45, ha="right")
    axes[1].grid(axis="y")
    axes[1].legend(frameon=False, ncol=2)
    fig.tight_layout()
    save_figure(fig, output / "figures" / "04_monthly_stability")


def plot_loto(output: Path, frame: pd.DataFrame, summary: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    retention_counts = frame["top100_retained"].value_counts().sort_index()
    axes[0].bar(retention_counts.index, retention_counts.values, color=BLUE)
    axes[0].set_xlabel("Original top-100 players retained")
    axes[0].set_ylabel("Tournament removals")
    axes[0].set_title("Most single-event removals leave all 100 names")
    axes[0].grid(axis="y")

    sorted_frame = frame.sort_values(
        "p90_abs_rank_change_original_top100", ascending=False
    ).reset_index(drop=True)
    axes[1].plot(
        np.arange(len(sorted_frame)),
        sorted_frame["p90_abs_rank_change_original_top100"],
        color=PURPLE,
    )
    axes[1].axhline(
        1, color=ORANGE, linestyle="--", label="90% of events at or below 1 place"
    )
    axes[1].set_xlabel("Tournament removals, sorted by disruption")
    axes[1].set_ylabel("90th-percentile shared-player movement")
    axes[1].set_title("Movement is usually tiny, with a short disruption tail")
    axes[1].grid(axis="y")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"Leave-one-tournament-out: median retention {summary['median_top100_retention']:.0%}; worst {summary['worst_top100_retention']:.0%}",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, output / "figures" / "05_leave_one_tournament_out")


def plot_half_life(output: Path, frame: pd.DataFrame) -> None:
    labels = frame["setting"].tolist()
    x = np.arange(len(frame))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(
        x, frame["match_micro_log_loss"], marker="o", color=BLUE, label="Match-micro"
    )
    axes[0].plot(
        x,
        frame["tournament_macro_log_loss"],
        marker="o",
        color=PURPLE,
        label="Tournament-macro",
    )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Held-out log loss (lower is better)")
    axes[0].set_title("Prediction changes gradually across decay settings")
    axes[0].grid()
    axes[0].legend(frameon=False)
    axes[1].plot(x, frame["mean_monthly_top100_retention"], marker="o", color=GREEN)
    axes[1].set_xticks(x, labels)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].set_ylim(0.80, 0.86)
    axes[1].set_ylabel("Mean monthly top-100 retention")
    axes[1].set_title("No setting uniquely dominates stability")
    axes[1].grid()
    fig.suptitle(
        "Half-life tradeoff (Plus AUC treated as secondary, not a tuning target)",
        fontsize=13,
        y=1.03,
    )
    fig.tight_layout()
    save_figure(fig, output / "figures" / "A1_half_life_tradeoff")


def plot_lineup_transfer(output: Path, familiarity: pd.DataFrame) -> None:
    wanted = [
        "LOOPR",
        "OpenSkill ordinal",
        "Bradley–Terry (L2)",
        "90-day win rate",
        "Chance (0.5)",
    ]
    groups = ["familiar", "mixed", "mostly novel"]
    frame = familiarity[familiarity["method"].isin(wanted)].copy()
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    width = 0.15
    x = np.arange(len(groups))
    palette = [BLUE, GREEN, PURPLE, ORANGE, GRAY]
    for i, (method, color) in enumerate(zip(wanted, palette)):
        values = [
            float(
                frame[
                    (frame["method"] == method) & (frame["familiarity_group"] == group)
                ]["log_loss"].iloc[0]
            )
            for group in groups
        ]
        ax.bar(x + (i - 2) * width, values, width, label=method, color=color)
    ax.set_xticks(
        x, ["Familiar\n(n=1,618)", "Mixed\n(n=2,465)", "Mostly novel\n(n=314)"]
    )
    ax.set_ylabel("Held-out log loss (lower is better)")
    ax.set_title("LOOPR loses predictive strength on mostly novel lineups")
    ax.grid(axis="y")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    save_figure(fig, output / "figures" / "A2_unfamiliar_lineups")


def claim_ledger(output: Path) -> pd.DataFrame:
    columns = [
        "Current claim in the draft",
        "Evidence offered",
        "Whether the evidence supports it",
        "Recommended revised wording",
        "Remaining limitation",
        "Suggested main-path or collapsible placement",
    ]
    rows = [
        [
            "“Bayesian systems … are stateless.”",
            "Conceptual comparison only.",
            "No. Elo/OpenSkill ratings are stateful; the intended distinction is lack of global retroactive recomputation in the specified implementation.",
            "“Sequential systems usually preserve past updates; LOOPR deliberately recomputes the whole result network so later evidence can recontextualize earlier wins.”",
            "Some Bayesian models can be smoothed or refit globally.",
            "Main-path mechanism explanation.",
        ],
        [
            "“They’re gameable … [and] can’t tell the difference between a major and a local.”",
            "No direct gaming experiment in the draft.",
            "Not established. OpenSkill variants actually predict held-out matches better here.",
            "“A match-by-match system without event features may not encode the distinction we wanted; LOOPR attempts to do so through event influence.”",
            "The present tournament-weighting property test is inconclusive.",
            "Collapsible; remove the categorical main-path claim.",
        ],
        [
            "“PageRank captures something Elo can’t: transitive relationships.”",
            "Graph mechanism and examples.",
            "Too absolute. Both families can transmit information; LOOPR’s distinctive feature is global recontextualization over the observed graph.",
            "“LOOPR makes those transitive paths explicit and recomputes them globally.”",
            "Mechanism alone does not prove better measurement.",
            "Main-path mechanism explanation.",
        ],
        [
            "“The ratio is the same” / “Volume-invariant” / “It doesn’t matter how much you play.”",
            "Illustration, prior simulations, and a real-data residual test.",
            "Supports resistance, not cancellation. Standardized log-match coefficient −0.009; partial R² 0.00025 after limited controls.",
            "“In these data LOOPR was largely volume-resistant after accounting for record and opponent strength; it did not prove perfect volume cancellation.”",
            "Observational controls are incomplete; opportunity and selection can still matter.",
            "One sentence main path; diagnostics in a collapsible.",
        ],
        [
            "“Score differences map to predicted win probabilities via Bradley–Terry.”",
            "Log-odds form.",
            "Only after defining team aggregation and fitting a prior-only temperature; the raw scale is not automatically calibrated.",
            "“A single calibration parameter maps pre-event team-score differences to win probabilities.”",
            "Reliability still departs from the diagonal in some bins.",
            "Main path beside calibration plot.",
        ],
        [
            "Tournament-strength weighting prevents farming weaker events.",
            "Intended tick-tock mechanism; observed event-removal check.",
            "Not demonstrated. The signed contribution pattern does not cleanly show weak-event damping.",
            "“Tournament influence is designed to reduce weak-event leverage; this audit did not establish complete or even clean monotonic resistance.”",
            "Event strength is model-derived and event categories are not randomized.",
            "Technical collapsible; no main-path success claim.",
        ],
        [
            "Exposure teleportation prevents sparse players from swinging wildly.",
            "24 one-match perturbations with and without smoothing.",
            "Not demonstrated by this test; smoothed and unsmoothed movement were nearly identical.",
            "“Exposure teleportation is intended as regularization, but this small perturbation test did not show a material stability gain.”",
            "Only six sampled players per exposure band; rank movement is noisy.",
            "Technical collapsible.",
        ],
        [
            "The 180-day half-life is the appropriate setting.",
            "Six-setting walk-forward/stability sweep.",
            "Supports a reasonable tradeoff, not a unique optimum. Ninety days predicts slightly better (0.511 vs 0.513 log loss).",
            "“Several settings performed similarly. We retained 180 days as an explainable recency–continuity choice, not an empirically unique optimum.”",
            "No formal utility function combines responsiveness and continuity.",
            "Short main-path note; full sweep collapsible.",
        ],
        [
            "“Strong concordance (~0.65–0.70) and well-calibrated probabilities.”",
            "Old cross-validation output.",
            "Partly. Revised accuracy is 0.748; calibration is monotone but imperfect. Log loss should be primary.",
            "“LOOPR’s held-out log loss was 0.513, versus 0.578 for recent win rate and 0.693 for chance; its reliability curve was directionally sensible but not perfect.”",
            "Accuracy is threshold-dependent and does not measure probability quality.",
            "Main prediction table and reliability figure.",
        ],
        [
            "LOOPR and OpenSkill are “essentially the same” / “by all our metrics.”",
            "Rank-agreement plots and older ablations.",
            "No. All three stronger identity-only baselines have clearly lower paired held-out log loss than LOOPR.",
            "“The systems share signal but differ materially in predictive performance and in some subgroup behavior.”",
            "No baseline is a ground truth for latent skill.",
            "Main prediction table.",
        ],
        [
            "OpenSkill is an “industry standard.”",
            "No evidence needed for the actual comparison; the label is rhetorical.",
            "Avoid. It neither strengthens the design nor makes OpenSkill a reference truth.",
            "“We included OpenSkill μ and ordinal as existing Bayesian baselines.”",
            "Implementation and score choice matter.",
            "Methods collapsible.",
        ],
        [
            "“The result: 0.86 AUC.”",
            "Figure and draft assertion; no recoverable protocol or saved row.",
            "No. The value is unreproduced. Revised institutional macro AUC is 0.777 (95% CI 0.719–0.831).",
            "“Averaged equally across the three tier-specific comparisons, the passer-above-failer probability was 77.7%.”",
            "Only 310 of 468 candidates met the predeclared 90-day activity definition.",
            "Main Plus table.",
        ],
        [
            "LOOPR is “excellent at identifying what players classify as skill.”",
            "Pooled/tier vote agreement.",
            "Too broad. Like-for-like tier/status AUC is 0.615, and its advantage over activity alone is unresolved.",
            "“LOOPR was concordant with this consequential vote, especially at the institutional tier level; agreement weakened under like-for-like comparison.”",
            "One vote snapshot may encode reputation, access, visibility, and current rules.",
            "Main Plus result and limitation.",
        ],
        [
            "Agreement is “mutual validation” proving both capture “something real about player skill.”",
            "Institutional AUC plus independence rhetoric.",
            "Convergent evidence, not proof. The shuffle control shows outcome direction adds information; subjectivity is not a defect.",
            "“The vote supplies convergent evidence from a differently aggregated community judgment.”",
            "The sources are not fully independent because both arise from the same competitive community.",
            "Main interpretation paragraph.",
        ],
        [
            "Disagreeing after validation is a “tough position.”",
            "Aggregate Plus agreement.",
            "No. Aggregate evidence does not adjudicate every exact pair or make criticism illegitimate.",
            "Remove. Add: “A high AUC supports broad ordering, not every neighboring rank.”",
            "Individual reversals can be both socially meaningful and statistically unresolved.",
            "Main limitation; sets up Part II.",
        ],
        [
            "LOOPR “extracts skill” / contains a “real skill signal.”",
            "Prediction, Plus, robustness, property tests.",
            "Only with a bounded construct. No objective player-skill vector is observed.",
            "“LOOPR contains substantial outcome-derived information about demonstrated organized-play success.”",
            "It can also reflect access, teammate quality, opportunity, and network position.",
            "Construct definition at the start and conclusion.",
        ],
        [
            "“But it works” / claims of “accuracy.”",
            "Multiple heterogeneous metrics.",
            "Too vague. Some tests support LOOPR; stronger baselines predict better, smoothing/weighting are unresolved, and novel-lineup transfer is weak.",
            "Name the supported property each time: predictive information, vote concordance, stability, or robustness.",
            "No single metric validates exact rank accuracy.",
            "Throughout; delete the blanket conclusion.",
        ],
        [
            "Predictive and vote evidence establish leaderboard legitimacy.",
            "Implicit bridge in the validation and skeptic sections.",
            "No. Measurement quality and legitimacy are different levels of claim.",
            "“Part I supports an outcome-derived skill-o-meter. Whether named exact ranks should govern public status is Part II’s question.”",
            "Publication creates social and institutional effects not measured here.",
            "Final Part I transition.",
        ],
    ]
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(output / "claim_ledger.csv", index=False)
    write_text(output / "claim_ledger.md", "# Claim ledger\n\n" + markdown_table(frame))
    return frame


def revised_evaluation_section(output: Path, data: dict[str, Any]) -> str:
    p = data["prediction"].set_index("method")
    plus = data["plus_methods"].set_index("method")
    shuffle = data["shuffle"]
    stability = data["stability_summary"]
    loto = data["loto"]
    text = f"""
### Validation Without a Ground Truth

Building a complicated ranking system is easy. Showing that it measures anything useful is the hard part.

There is no objective player-skill number hiding in the data for LOOPR to recover. Skill is latent, and team results are produced by more than mechanics: communication, leadership, adaptation, teammate selection, access to events, and opportunity all matter. So I stopped asking whether LOOPR had found *the* true ranking. I asked a narrower set of questions that the available evidence could actually answer.

Throughout this section, the construct is:

> **LOOPR estimates a player’s demonstrated ability to convert organized-play match opportunities into tournament victories, given the available network of teammates, opponents, events, and results.**

That is deliberately narrower than “true skill,” and broader than an individual-stat line. A player who makes teams win through coordination or leadership belongs inside the construct. Network access and teammate opportunity may also leak into it. Part II deals with what happens when that conditional, outcome-derived estimate is published as an exact social ordering.

#### What the Dataset Can See

The rating assumes that players mix enough for individual effects to emerge from team results. The graph passes the first test: all 5,518 players observed in valid matches belong to one weakly connected component, as do all of the final top 100. The typical player has six unique teammates, 48 unique opponents, and three distinct lineups.

But connection is not the same as individual identifiability. Fifteen percent of players with sufficient matches use one lineup at least 80% of the time, and 32% have one teammate beside them in at least 90% of matches. For those players, the data may support a claim about the pair or lineup more strongly than a claim about the exact ordering within it.

[Insert “What the Dataset Can See” table.]

#### Can It Predict What Happens Next?

The cleanest test is temporal. Before each of 110 held-out tournaments, I rebuilt every rating using only earlier eligible matches. I calibrated probabilities using earlier held-out events only, predicted the next tournament, then advanced the cutoff. Ratings were frozen within an event and recomputed before the next one.

The primary metric was **log loss**. In plain language: did the model assign high probability to the team that actually won, without becoming recklessly confident? Lower is better.

LOOPR’s match-weighted log loss was **{p.loc["LOOPR", "match_micro_log_loss"]:.3f}** across 4,397 matches. That was substantially better than a 90-day win-rate model (**{p.loc["90-day win rate", "match_micro_log_loss"]:.3f}**) and a 50/50 prediction (**{p.loc["Chance (0.5)", "match_micro_log_loss"]:.3f}**). The paired LOOPR advantage over recent win rate was {p.loc["90-day win rate", "paired_log_loss_difference_vs_loopr"]:.3f} log-loss points, with a tournament-cluster 95% interval from {p.loc["90-day win rate", "paired_difference_ci_low"]:.3f} to {p.loc["90-day win rate", "paired_difference_ci_high"]:.3f}. Tournament-weighted results told the same basic story, so one giant event did not create the finding.

That supports a real but bounded claim: LOOPR contains substantial information about future tournament outcomes beyond recent win rate alone.

It was not the best predictor in the comparison. OpenSkill ordinal reached **{p.loc["OpenSkill ordinal", "match_micro_log_loss"]:.3f}**, OpenSkill μ **{p.loc["OpenSkill μ", "match_micro_log_loss"]:.3f}**, and a regularized team Bradley–Terry model **{p.loc["Bradley–Terry (L2)", "match_micro_log_loss"]:.3f}**. Their paired intervals versus LOOPR all excluded zero in their favor. These systems are baselines, not ground truth, but the result rules out saying that LOOPR extracts the available outcome signal better than the alternatives.

[Insert future-prediction table and calibration figure.]

The calibration curve was directionally sensible: higher predicted probabilities corresponded to higher observed win rates. It was not perfect. LOOPR was somewhat underconfident in the lower and middle bins and slightly overconfident in the highest bin.

There was also an important transfer limit. LOOPR log loss was 0.452 on familiar lineups, 0.530 on mixed lineups, and 0.688—nearly chance—on the 314 mostly novel-lineup matches. The model is strongest inside the teammate/opponent network it has already seen. That is evidence for conditional organized-play measurement, not a context-free player essence.

#### Does It Converge With Plus Voting?

Plus voting is not ground truth. It is a different, fallible, socially consequential measurement process. That makes agreement useful as **convergent evidence**.

I used the November 16, 2025 peer-vote snapshot and the historical 60% pass rule. Automatic SendouQ admissions were not present in these vote-result rows. Of 468 candidates, 310 met the predeclared 90-day roster-activity rule; every method was evaluated on the same 310.

First I asked the institutional question: within each tier, did the score tend to put passers above failers? LOOPR’s macro AUC was **{plus.loc["LOOPR", "institutional_macro_auc"]:.3f}** (95% CI {plus.loc["LOOPR", "institutional_ci_low"]:.3f}–{plus.loc["LOOPR", "institutional_ci_high"]:.3f}). This is the equal-weight mean of the three tier-specific passer-above-failer probabilities: 77.7% here. It does not mean the ranking is 77.7% accurate.

Then I made the comparison stricter. Incumbents and newly suggested candidates face different visibility, activity, and pass-rate conditions, so I compared passers and failers only within the same tier *and* candidate status, then combined the six cells by their available passer–failer pairs. That like-for-like AUC fell to **{plus.loc["LOOPR", "like_for_like_stratified_auc"]:.3f}** (95% CI {plus.loc["LOOPR", "stratified_ci_low"]:.3f}–{plus.loc["LOOPR", "stratified_ci_high"]:.3f}). The evidence still points in the intended direction, but much less dramatically.

Activity alone did not reproduce the institutional result: recent-match AUC was {plus.loc["Matches in prior 90d", "institutional_macro_auc"]:.3f}, and recent-tournament AUC was {plus.loc["Tournaments in prior 90d", "institutional_macro_auc"]:.3f}. Under the like-for-like comparison, however, their AUCs were {plus.loc["Matches in prior 90d", "like_for_like_stratified_auc"]:.3f} and {plus.loc["Tournaments in prior 90d", "like_for_like_stratified_auc"]:.3f}. LOOPR’s estimated advantage was small relative to uncertainty. So activity does not explain the headline institutional agreement, but this sample does not clearly show that LOOPR beats activity within every comparable decision context.

#### Does the Outcome Direction Matter, or Just Who Participated?

To separate results from access and visibility, I preserved every player, lineup, tournament, timestamp, and match count, replaced each winner with a fair coin flip, and recomputed the full LOOPR pipeline 200 times—including tournament-strength iteration.

The shuffled institutional AUC averaged **{shuffle["null_macro_mean"]:.3f}**, with a 95% range of {shuffle["null_macro_q025"]:.3f}–{shuffle["null_macro_q975"]:.3f}. None of the 200 draws reached the observed {shuffle["observed_macro_auc"]:.3f} (corrected tail probability {shuffle["null_macro_exceedance_p"]:.3f}). The shuffled like-for-like AUC averaged {shuffle["null_stratified_mean"]:.3f}; the observed {shuffle["observed_stratified_auc"]:.3f} also exceeded that null (corrected tail probability {shuffle["null_stratified_exceedance_p"]:.3f}).

This is strong evidence that win/loss direction adds information beyond participation structure. It does not erase the role of access or visibility; it says showing up alone did not produce the observed agreement.

[Insert Plus-and-shuffle figure.]

#### Is It Stable Without Being Frozen?

Stability is an operational requirement, not proof of accuracy. An incorrect ranking can be perfectly stable.

Across 13 adjacent monthly transitions, average top-100 retention was **{stability["mean_retention"]:.1%}**. Among players retained in both lists, average Spearman ordering correlation was **{stability["mean_spearman_retained"]:.3f}**, and the median transition’s typical retained player moved four positions. But retention ranged from 60% to 98%. Between October 31 and November 30, only 60 names remained; among those retained, the median player moved 11 places.

The old Spearman-only summary hid that churn. The accurate conclusion is: ordering among retained players is usually stable, while membership can change substantially when activity and new results alter who belongs in the top 100.

[Insert monthly-stability figure.]

#### Does One Tournament Control the Ranking?

For each of 665 eligible tournaments, I removed the event and recomputed the entire pipeline. Median top-100 retention was **{loto["median_top100_retention"]:.0%}**; the worst case was **{loto["worst_top100_retention"]:.0%}** after removing {loto["worst_retention_tournament_name"]}. The median event produced a median shared-player movement of zero places, and 90% of event removals had a 90th-percentile shared-player movement of no more than one place.

There is a tail. Removing {loto["largest_shared_shift_tournament_name"]} moved one still-eligible top-100 player {loto["largest_observed_shared_player_rank_shift"]:.0f} places. That player had {loto["largest_shared_shift_player_full_matches"]} historical matches, so the largest shared shift was not limited to a low-match newcomer. Eighteen removals also changed current rank eligibility for at least one original top-100 player; those cases are counted separately rather than pretending the player fell to a numeric bottom rank.

The ranking is broadly supported by the network at this cutoff, but no individual should be told that one event can never matter sharply.

[Insert leave-one-tournament-out figure.]

#### Did the Components Behave as Intended?

The targeted tests were mixed.

- Among 2,793 real-player comparisons matched approximately on win rate and volume, the player with the stronger measured opponent schedule had the higher LOOPR score 72.8% of the time.
- After controlling for win rate and opponent strength with simple flexible terms, the residual standardized coefficient on log match volume was −0.009, with partial R² 0.00025. That supports volume resistance, not perfect cancellation.
- In 24 one-match perturbations, smoothing and no-smoothing movement were nearly identical. This test did not demonstrate the intended low-evidence protection.
- The observed tournament-weighting test did not cleanly show that weighting suppresses leverage from weak events. Complete Goodhart resistance remains unestablished.

The half-life sweep was also a tradeoff, not an optimization victory. Ninety days predicted slightly better than 180 days (0.511 versus 0.513 match-level log loss), while monthly retention differed by about one percentage point. Several settings were similar. Choosing 180 days can be defended as a continuity preference, but not as the uniquely correct value.

### What the Evidence Supports

Several distinct tests pointed in the same direction. LOOPR predicted later tournament outcomes, beat recent win rate and chance, produced coherent probability ordering, converged with a consequential community judgment, exceeded a participation-only shuffled null, and was usually robust to removing one event. Its schedule and volume behavior also moved in the intended direction.

The same audit set boundaries. LOOPR did not beat the stronger predictive baselines. Like-for-like Plus agreement was moderate. Transfer to mostly novel lineups was weak. Monthly membership could churn. Smoothing and tournament-strength weighting did not earn strong empirical claims from these tests.

Taken together, that supports treating LOOPR as a **meaningful outcome-derived skill-o-meter** for demonstrated organized-play success. Calling it “strong” is defensible if that means substantial information supported by several different tests—not best-in-class prediction, objective skill recovery, or validation of every exact pairwise rank.

I assumed that building a meaningful measurement instrument meant I had built a meaningful public leaderboard.

The Splatoon community rejected it anyway.

That is where Part II begins.
"""
    text = dedent(text).strip()
    write_text(output / "revised_evaluation_section.md", text)
    return text


def methodology_audit(output: Path, data: dict[str, Any], checks: list[str]) -> str:
    audit = data["audit"]
    manifest = data["manifest"]
    reproduction = data["reproduction"]
    reproduction_display = reproduction.copy()
    reproduction_display["displayed"] = reproduction_display["displayed"].map(
        lambda x: f"{x:.6f}" if pd.notna(x) else "—"
    )
    reproduction_display["recomputed"] = reproduction_display["recomputed"].map(
        lambda x: f"{x:.6f}" if pd.notna(x) else "—"
    )
    text = f"""
# Data and methodology audit

## Bottom line

The historical 0.526 LOOPR prediction figure, the 0.497/0.489 ablation figures, and the 0.772/0.779 Plus figures are reproducible from saved outputs—but they are not estimates from one common protocol. The draft’s 0.86 Plus AUC has no recoverable saved protocol and should be removed. The revised analysis uses deterministic tournament canonicalization, explicit timestamp fallback, strict event-by-event temporal cutoffs, prior-only calibration, and paired tournament-cluster uncertainty.

## Available data

- {audit["canonical_raw_records"]:,} canonical tournament records selected from {audit["duplicate_source_audit"]["records_loaded"]:,} JSON records across {audit["duplicate_source_audit"]["json_files"]} files.
- {audit["eligible_tournaments"]:,} eligible ranked tournaments and {audit["eligible_decided_matches"]:,} valid decided non-bye matches from {audit["date_start"]} through {audit["date_end"]}.
- {audit["eligible_unique_players"]:,} players resolved into valid matches; {audit["eligible_unique_roster_players"]:,} unique roster players.
- Match timestamps, team IDs, winners and losers, tournament metadata, rosters, and partial appearance/substitution payloads.
- Historical Plus peer-vote snapshots with tier, suggested/incumbent status, and pass/fail outcome.
- LOOPR, OpenSkill, win-rate, evaluation scripts, saved ablations, and the two essay drafts.

The complete file-level inventory of relevant prior scripts, notebooks, saved tables/models, reports, figures, vote files, implementation files, and environment files is in [`audit/project_file_inventory.csv`](audit/project_file_inventory.csv), with byte counts and SHA-256 hashes. Raw tournament JSON is represented by its aggregate plus the canonical record-source table.

## Missing or incomplete data

- Vote percentages are absent from the available historical CSVs; continuous vote-percentage analysis is therefore omitted.
- Exact appearances are available on both sides for only {audit["appearance_both_matches"]:,} matches ({audit["appearance_both_matches"] / audit["eligible_decided_matches"]:.1%}). The other {audit["roster_fallback_matches"]:,} matches use tournament-team rosters, so unrecorded substitutions cannot be recovered.
- {audit["timestamp_imputed_matches"]:,} matches require tournament-start timestamp fallback after checking finish and creation times.
- Reliable exogenous labels for “weak,” skill-capped, local, or major events are unavailable; the tournament-weighting property test is consequently endogenous and limited.
- There is no objective player-skill vector. That is a construct boundary, not a missing field to fabricate.

## Duplicate and timestamp decisions

The raw tree contains {audit["duplicate_source_audit"]["duplicate_tournament_ids"]} duplicated tournament IDs and {audit["duplicate_source_audit"]["extra_record_copies"]} extra copies. Of those IDs, {audit["duplicate_source_audit"]["differing_duplicate_ids"]} have differing contents. The revised loader selects the most-complete record deterministically and merges supplemental appearance rows. The legacy recursive loader is retained only for reproduction.

The repository converter assigned missing match times a global/latest timestamp. That is invalid for historical cutoffs. The revised priority is exact finish time, then match creation time, then tournament start. Counts are recorded in the machine-readable manifest.

## Reproduction ledger

{markdown_table(reproduction_display[["result", "displayed", "recomputed", "reproduced", "protocol", "explanation"]])}

### Discrepancy reconciliation

- **0.526 versus about 0.499:** 0.526394 is the saved January production harness, using three sampled temporal blocks of ten tournaments with ratings frozen inside each block. The roughly 0.497 and 0.489 numbers are separate hand-built ablation rows. They are not reruns of one model under one protocol.
- **0.772 versus 0.779:** 0.772 is the exposure/smoothing variant; 0.779 is the later tick-tock variant.
- **0.86 Plus AUC:** no saved table or script output identifies a protocol that produces it. It remains unreproduced and is excluded from the revised claims.
- **Score-field semantics:** before the March 2026 implementation migration, `player_rank` held a score; afterward it holds ordinal rank and `score` holds the log-odds value. The revised harness reads `score` explicitly and pins `loopr-ranking==0.0.1`.

## Revised temporal and uncertainty protocol

- Training data for event *t* contain only matches timestamped strictly before that event’s cutoff; event *t* itself is excluded.
- Ratings are recomputed before every held-out event and frozen only within that event.
- One positive calibration temperature is fitted only from earlier out-of-sample raw differences, after at least 200 matches and five tournaments.
- The report window is {manifest["prediction"]["reported_test_start"]} through {manifest["prediction"]["test_end"]}; the warmup begins {manifest["prediction"]["calibration_warmup_start"]}.
- All prediction systems use the same 4,397 matches in 110 tournaments. Unseen and low-history players remain in the test set under declared population priors.
- Model comparisons use paired differences. The primary resampling cluster is tournament (5,000 bootstrap repetitions), because matches within an event are dependent.
- Plus candidate intervals use 2,000 bootstraps stratified by tier, status, and outcome. Outcome shuffling uses 200 full recomputations with seed {SEED}.

## Historical Plus-rule provenance

The November 16, 2025 analysis uses peer-vote results under the historical 60% rule. The [current Sendou FAQ](https://sendou.ink/faq) documents the vote and SendouQ admission paths; the version-pinned [August/November-era database migration](https://github.com/sendou-ink/sendou.ink/blob/c13ff32b406647fe66da7d971fd238951a832e54/migrations/041-plus-sixty.js) encodes the 60% decision as a mean ±1 vote score of at least 0.2. Automatic SendouQ admissions are absent from the exported `PlusVotingResult` rows. Later 2026 rule changes are not substituted into this vote.

## Leakage and dependence audit

- No random match split is used in the revised analysis.
- The held-out tournament is removed from every fitted rating, including the global LOOPR graph.
- Calibration never sees the event it predicts.
- The half-life sweep uses the same 53 warmup and 110 report events; the 180-day row exactly reproduces the primary LOOPR result.
- Plus AUC is secondary in the half-life sweep and is not used to select a winning parameter.
- Match dependence is handled by paired tournament-cluster resampling for prediction comparisons.

## Limitations that remain

1. Sparse appearance payloads make player-level lineup attribution imperfect.
2. Persistent teammate/lineup clusters weaken within-cluster identifiability even though the graph is connected.
3. Evaluation covers one observed organized-play network and ends on {audit["date_end"]}.
4. Plus results cover one historical vote and 310 active candidates; vote percentage is unavailable.
5. The unfamiliar-lineup group is small (314 matches) and especially difficult.
6. Property tests for smoothing and tournament weighting are underpowered or endogenous and do not establish their intended benefits.
7. Measurement evidence cannot decide whether publishing named exact ranks is socially legitimate.

## Automated consistency checks

""" + "\n".join(f"- Passed: {check}." for check in checks)
    text = dedent(text).strip()
    write_text(output / "data_methodology_audit.md", text)
    return text


def technical_report(
    output: Path, data: dict[str, Any], main_tables: dict[str, pd.DataFrame]
) -> str:
    p = data["prediction"].set_index("method")
    plus = data["plus_methods"].set_index("method")
    s = data["stability_summary"]
    loto = data["loto"]
    half = data["half_life"].copy()
    props = data["properties"]
    familiarity = data["familiarity"]
    plus_strata = data["plus_strata"]
    smooth = data["smoothing"]

    formatted_prediction = main_tables["future_prediction"].copy()
    for col in [
        "match_micro_log_loss",
        "tournament_macro_log_loss",
        "brier_score",
        "accuracy",
        "paired_log_loss_difference_vs_loopr",
        "paired_difference_ci_low",
        "paired_difference_ci_high",
        "median_per_tournament_difference",
    ]:
        formatted_prediction[col] = formatted_prediction[col].map(
            lambda value: f"{value:.3f}"
        )
    formatted_prediction = formatted_prediction.rename(
        columns={
            "method": "System",
            "match_micro_log_loss": "Match-micro log loss",
            "tournament_macro_log_loss": "Tournament-macro log loss",
            "brier_score": "Brier",
            "accuracy": "Accuracy",
            "paired_log_loss_difference_vs_loopr": "Paired Δ vs LOOPR",
            "paired_difference_ci_low": "Δ CI low",
            "paired_difference_ci_high": "Δ CI high",
            "tournaments_outperforming_loopr": "Events beating LOOPR",
            "median_per_tournament_difference": "Median event Δ",
        }
    )
    formatted_plus = main_tables["plus_agreement"].copy()
    for col in formatted_plus.columns:
        if col not in {"method", "n"}:
            formatted_plus[col] = formatted_plus[col].map(lambda value: f"{value:.3f}")
    formatted_plus = formatted_plus.rename(
        columns={
            "method": "System",
            "n": "Candidates",
            "institutional_macro_auc": "Institutional macro AUC",
            "institutional_ci_low": "Macro CI low",
            "institutional_ci_high": "Macro CI high",
            "like_for_like_stratified_auc": "Like-for-like AUC",
            "stratified_ci_low": "Stratified CI low",
            "stratified_ci_high": "Stratified CI high",
        }
    )
    stability_display = main_tables["monthly_stability"].copy()
    stability_display["spearman_on_retained"] = stability_display[
        "spearman_on_retained"
    ].map(lambda value: f"{value:.3f}")
    for col in ["median_absolute_rank_movement", "p90_absolute_rank_movement"]:
        stability_display[col] = stability_display[col].map(
            lambda value: f"{value:.1f}"
        )
    stability_display = stability_display.rename(
        columns={
            "month_a": "Month A",
            "month_b": "Month B",
            "top100_retained": "Retained",
            "spearman_on_retained": "Retained-player Spearman",
            "median_absolute_rank_movement": "Median move",
            "p90_absolute_rank_movement": "90th-pct move",
        }
    )
    half_display = half.copy()
    for col in half_display.columns:
        if col not in {"setting"}:
            half_display[col] = half_display[col].map(
                lambda value: "—" if pd.isna(value) else f"{value:.3f}"
            )
    strata_display = plus_strata[(plus_strata["method"] == "LOOPR")].copy()
    strata_display["suggested"] = (
        strata_display["suggested"]
        .map({True: "suggested", False: "incumbent"})
        .fillna("all")
    )
    strata_display["auc"] = strata_display["auc"].map(lambda value: f"{value:.3f}")
    for col in ["ci_low", "ci_high"]:
        strata_display[col] = strata_display[col].map(
            lambda value: "—" if pd.isna(value) else f"{value:.3f}"
        )
    familiarity_display = familiarity[familiarity["method"] == "LOOPR"][
        ["familiarity_group", "matches", "log_loss", "accuracy"]
    ].copy()
    familiarity_display["log_loss"] = familiarity_display["log_loss"].map(
        lambda x: f"{x:.3f}"
    )
    familiarity_display["accuracy"] = familiarity_display["accuracy"].map(
        lambda x: f"{x:.1%}"
    )
    smooth_display = smooth.copy()
    for col in ["median_absolute_score_change", "p90_absolute_score_change"]:
        smooth_display[col] = smooth_display[col].map(lambda value: f"{value:.3f}")
    for col in ["median_absolute_rank_change", "p90_absolute_rank_change"]:
        smooth_display[col] = smooth_display[col].map(lambda value: f"{value:.1f}")

    report = f"""
# Building Public Rankings, Part I: empirical audit

**Data cutoff:** 2026-01-28  
**Primary held-out window:** 2025-08-17 to 2026-01-28  
**Unit of temporal holdout and prediction resampling:** tournament  
**Random seed:** {SEED}

## Answer first

LOOPR is a meaningful outcome-derived measurement instrument, but this audit does not support calling it the best predictor or treating its exact public ordering as objective.

The strongest defensible construct statement is:

> **LOOPR estimates a player’s demonstrated ability to convert organized-play match opportunities into tournament victories, given the available network of teammates, opponents, events, and results.**

Why the positive part is defensible:

- LOOPR predicts later matches much better than chance (log loss {p.loc["LOOPR", "match_micro_log_loss"]:.3f} versus {p.loc["Chance (0.5)", "match_micro_log_loss"]:.3f}) and recent win rate ({p.loc["90-day win rate", "match_micro_log_loss"]:.3f}).
- Institutional Plus agreement is substantial (macro AUC {plus.loc["LOOPR", "institutional_macro_auc"]:.3f}), and both that result and the stricter tier/status AUC ({plus.loc["LOOPR", "like_for_like_stratified_auc"]:.3f}) exceed full outcome-shuffle nulls.
- The final top 100 is usually insensitive to any one event: median leave-one-event retention is {loto["median_top100_retention"]:.0%}, worst {loto["worst_top100_retention"]:.0%}.
- Residual association with match volume is tiny after basic record/schedule controls, and matched schedule comparisons usually move in the intended direction.

Why the boundary matters:

- OpenSkill ordinal ({p.loc["OpenSkill ordinal", "match_micro_log_loss"]:.3f}), OpenSkill μ ({p.loc["OpenSkill μ", "match_micro_log_loss"]:.3f}), and regularized Bradley–Terry ({p.loc["Bradley–Terry (L2)", "match_micro_log_loss"]:.3f}) all predict held-out matches more accurately by paired tournament comparison.
- Like-for-like Plus agreement is only moderate, and LOOPR’s advantage over activity-only measures is not clearly resolved there.
- Mostly novel lineups are near chance for LOOPR (log loss 0.688).
- Top-100 ordering among retained players is high (mean Spearman {s["mean_spearman_retained"]:.3f}), but membership retention averages {s["mean_retention"]:.1%} and reaches a low of 60%.
- The current tests do not establish that smoothing or tournament weighting achieves its intended protection.

“Strong skill-o-meter” is therefore defensible only if *strong* means that several distinct, falsifiable tests show substantial organized-play information. It is not defensible if it means best-in-class prediction, context-free individual skill, every exact neighboring rank, or public-leaderboard legitimacy.

## Main-path exhibit 1 — What the dataset can see

{markdown_table(main_tables["what_the_dataset_can_see"])}

**Formal metric.** Weak graph components, teammate/opponent counts, lineup diversity, and concentration.  
**Plain-language question.** Do players mix enough for team results to say something about individuals?  
**Interpretation.** Everyone connects to one graph and the typical player has several teammate contexts, but a meaningful minority is nearly inseparable from a teammate or lineup.  
**Limitation.** Connectivity permits propagation; it does not guarantee precise separation of players who nearly always appear together. Only 9.6% of matches have exact appearance lineups on both sides.

## Main-path exhibit 2 — Future prediction

![Future prediction](figures/01_future_prediction.png)

{markdown_table(formatted_prediction)}

**Formal metric.** Match-weighted log loss is primary; tournament-weighted log loss, Brier score, and 0/1 accuracy are supplementary. Paired differences are method minus LOOPR, so negative favors the comparator.  
**Plain-language question.** Before a tournament starts, does the rating put sensible probabilities on who will win?  
**Interpretation.** LOOPR contains substantial predictive information beyond chance and recent win rate, but the available data clearly favor all three stronger identity-only baselines on log loss.  
**Limitation.** Prediction validates information about future outcomes, not an objective skill vector or every exact rank. The evaluation covers 110 tournaments, not every future metagame.

The comparator advantage is not a large-event artifact: tournament-macro loss preserves the direction. OpenSkill ordinal beats LOOPR in 62 of 110 tournaments, OpenSkill μ in 70, and Bradley–Terry in 71. Their median per-event advantages are smaller than the match-micro averages, so both summaries are shown.

## Main-path exhibit 3 — Probability reliability

![LOOPR calibration](figures/02_loopr_calibration.png)

**Formal metric.** Five equal-count reliability bins compare mean predicted favorite probability with observed favorite win rate.  
**Plain-language question.** When LOOPR says “about 70%,” does that kind of favorite win about 70% of the time?  
**Interpretation.** The curve is monotone and reasonably close to the diagonal, but lower/middle bins are underconfident and the top bin is slightly overconfident.  
**Limitation.** Five bins summarize average behavior and can hide within-bin heterogeneity; calibration was learned from earlier events only.

## Main-path exhibit 4 — Plus convergence and negative control

![Plus and shuffle](figures/03_plus_and_shuffle.png)

{markdown_table(formatted_plus)}

**Formal metric.** ROC AUC asks whether a passer receives a higher score than a failer. Institutional macro AUC gives each tier equal weight when averaging its pair-ordering probability; like-for-like AUC compares within tier and candidate status, weighted by available passer–failer pairs.  
**Plain-language question.** Does LOOPR tend to order candidates the same way this consequential peer-vote process did?  
**Interpretation.** Within tier, LOOPR agreement is substantial and much higher than exposure alone. Under like-for-like comparison, agreement is moderate and the advantage over activity is uncertain. Both observed measures exceed the 200-run outcome-shuffle null, so win/loss direction adds information beyond participation structure.  
**Limitation.** AUC is not percent accuracy, does not validate adjacent ranks, and covers one historical vote among 310 active candidates. Vote percentages are unavailable.

## Main-path exhibit 5 — Monthly stability

![Monthly stability](figures/04_monthly_stability.png)

{markdown_table(stability_display)}

**Formal metric.** Top-100 retention measures membership; Spearman measures ordering only among retained players; median and 90th-percentile absolute movement measure typical and upper-tail change.  
**Plain-language question.** Does the list remain recognizable from month to month without becoming frozen?  
**Interpretation.** Retained players are usually ordered similarly, but membership churn is meaningful and occasionally large.  
**Limitation.** Stability is operational, not accuracy. The 90-day activity rule is itself a source of entry and exit.

## Main-path exhibit 6 — Leave-one-tournament-out robustness

![Leave-one-tournament-out robustness](figures/05_leave_one_tournament_out.png)

{markdown_table(main_tables["loto_robustness"])}

**Formal metric.** Each eligible event is removed and the full graph/tick-tock pipeline is recomputed; retention and rank movement compare with the full-data final ranking.  
**Plain-language question.** Is the final top 100 broadly supported by the network, or can one event rewrite it?  
**Interpretation.** Most removals have negligible effect, but a small tail creates material shared-player shifts or changes 90-day rank eligibility.  
**Limitation.** This is sensitivity analysis, not a confidence interval, and it tests one event at a time rather than coordinated data errors or missing regions.

## Technical collapsible A — Plus strata

{markdown_table(strata_display[["comparison", "tier", "suggested", "n", "pass", "fail", "auc", "ci_low", "ci_high", "pairs"]])}

The +1-suggested cell has only two passers and nine failers. Pair weighting prevents that tiny high-AUC cell from dominating the aggregate. The large +3-incumbent cell is near chance (AUC 0.544), an important reason the stricter aggregate is lower.

## Technical collapsible B — Transfer to unfamiliar lineups

![Unfamiliar lineups](figures/A2_unfamiliar_lineups.png)

{markdown_table(familiarity_display)}

**Formal metric.** Prior teammate familiarity is the fraction of the six pairings in a four-player lineup with earlier shared-match history, averaged across both teams.  
**Plain-language question.** Does the rating still help when players form new teammate combinations?  
**Interpretation.** LOOPR is strong on familiar lineups, weaker on mixed ones, and nearly chance on mostly novel lineups.  
**Limitation.** The mostly novel group has only 314 matches, and roster fallback may misclassify familiarity when substitutions are missing.

## Technical collapsible C — Half-life tradeoff

![Half-life tradeoff](figures/A1_half_life_tradeoff.png)

{markdown_table(half_display)}

**Formal metric.** The same strict walk-forward prediction and monthly stability calculations are repeated for six decay settings.  
**Plain-language question.** Is 180 days uniquely best, or one reasonable continuity/recency compromise?  
**Interpretation.** Shorter decay predicts slightly better; stability differences are small and non-monotone. Several settings are plausible.  
**Limitation.** The sweep does not define a single utility function for prediction versus continuity. Plus AUC is secondary and was not used as an independent validation target after tuning.

## Technical collapsible D — Requirement-specific property tests

### Strength of schedule

- 2,793 approximate real-player matched comparisons.
- Stronger-schedule player ranks higher in {props["strength_of_schedule"]["share_rewarded_in_intended_direction"]:.1%}.
- Median score advantage {props["strength_of_schedule"]["median_score_advantage"]:.3f}.

This supports intended directional behavior. It is descriptive, reuses players, and conditions on model-derived opponent scores.

### Volume resistance

- {props["volume_resistance"]["players"]:,} players with at least ten matches.
- Standardized log-match coefficient {props["volume_resistance"]["standardized_log_match_coefficient"]:.3f} after controls.
- Partial R² {props["volume_resistance"]["partial_r_squared_log_matches"]:.5f}.

The residual is tiny, supporting volume resistance. It does not prove exact mathematical cancellation or remove opportunity bias.

### Limited evidence and smoothing

{markdown_table(smooth_display)}

Across 24 sampled players, smoothing and no-smoothing changes are nearly identical. This test does not demonstrate that the current smoothing implementation limits low-evidence perturbations. The sample is small and rank changes are noisy.

### Tournament-strength weighting

The eight lowest- and eight highest-influence event removals do not show a clean weak-event-damping pattern. Because event groups are defined by fitted influence and the signed contribution statistic is endogenous, this is inconclusive. The article should explain the intended design without claiming complete farming or Goodhart resistance.

## Claim-level conclusion

The evidence primarily supports:

1. LOOPR contains competitive information.
2. LOOPR supports broad ordinal distinctions about demonstrated organized-play success.

It does not establish:

3. That every exact pairwise rank is meaningful.
4. That publishing the ordering as a named public leaderboard is legitimate.

Those are not statistical footnotes. They are the bridge from Part I to Part II.

## Audit trail

- [Evaluation manifest](evaluation_manifest.json)
- [Data and methodology audit](data_methodology_audit.md)
- [Claim ledger](claim_ledger.md)
- [Proposed revised evaluation section](revised_evaluation_section.md)
- [Machine-readable results](results/results.json)
- [Reproduction ledger](audit/reproduction_ledger.csv)
- [Prior-materials inventory](audit/project_file_inventory.csv)
- [Runnable analysis README](README.md)
"""
    report = dedent(report).strip()
    write_text(output / "technical_report.md", report)
    return report


def build_notebook(output: Path) -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"].update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        }
    )
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Building Public Rankings, Part I — reproducible results companion\n\n"
            "The CLI rebuilds every rating from raw data. This notebook is the review layer: it reads the frozen, auditable outputs, reproduces the main tables, and checks the central numerical claims."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image\n\n"
            "OUTPUT = Path.cwd()\n"
            "if not (OUTPUT / 'evaluation_manifest.json').exists():\n"
            "    OUTPUT = Path.cwd() / 'analysis' / 'part1' / 'output'\n"
            "manifest = json.loads((OUTPUT / 'evaluation_manifest.json').read_text())\n"
            "manifest['dataset']"
        ),
        nbf.v4.new_markdown_cell("## 1. What the dataset can see"),
        nbf.v4.new_code_cell(
            "display(pd.read_csv(OUTPUT / 'tables/main_what_the_dataset_can_see.csv'))"
        ),
        nbf.v4.new_markdown_cell("## 2. Strict walk-forward prediction"),
        nbf.v4.new_code_cell(
            "prediction = pd.read_csv(OUTPUT / 'tables/prediction_summary.csv')\n"
            "display(prediction)\n"
            "loopr = prediction.set_index('method').loc['LOOPR']\n"
            "assert loopr.match_micro_log_loss < prediction.set_index('method').loc['90-day win rate'].match_micro_log_loss\n"
            "assert loopr.match_micro_log_loss > prediction.set_index('method').loc['OpenSkill ordinal'].match_micro_log_loss\n"
            "Image(filename=str(OUTPUT / 'figures/01_future_prediction.png'))"
        ),
        nbf.v4.new_markdown_cell("## 3. Reliability"),
        nbf.v4.new_code_cell(
            "display(pd.read_csv(OUTPUT / 'tables/loopr_calibration.csv'))\n"
            "Image(filename=str(OUTPUT / 'figures/02_loopr_calibration.png'))"
        ),
        nbf.v4.new_markdown_cell("## 4. Plus agreement and outcome shuffle"),
        nbf.v4.new_code_cell(
            "plus = pd.read_csv(OUTPUT / 'tables/plus_methods.csv')\n"
            "shuffle = json.loads((OUTPUT / 'results/outcome_shuffle.json').read_text())\n"
            "display(plus)\nshuffle"
        ),
        nbf.v4.new_markdown_cell("## 5. Stability and leave-one-event robustness"),
        nbf.v4.new_code_cell(
            "display(pd.read_csv(OUTPUT / 'tables/monthly_stability.csv'))\n"
            "json.loads((OUTPUT / 'results/leave_one_tournament_out.json').read_text())"
        ),
        nbf.v4.new_markdown_cell("## 6. Parameter and property checks"),
        nbf.v4.new_code_cell(
            "half = pd.read_csv(OUTPUT / 'tables/half_life_tradeoff.csv')\n"
            "display(half)\n"
            "primary = prediction.set_index('method').loc['LOOPR', 'match_micro_log_loss']\n"
            "sweep = half.set_index('setting').loc['180d', 'match_micro_log_loss']\n"
            "assert abs(primary - sweep) < 1e-12\n"
            "json.loads((OUTPUT / 'results/property_tests.json').read_text())"
        ),
        nbf.v4.new_markdown_cell(
            "## Rebuild commands\n\nSee `README.md`. The expensive raw-data stages are intentionally CLI-driven and checkpointed; this notebook remains a compact audit surface."
        ),
    ]
    nbf.write(notebook, output / "part1_results.ipynb")


def readme(output: Path) -> str:
    text = f"""
# Part I evaluation package

This directory is the reproducible audit for *Building Public Rankings*, Part I.

## Environment

From the repository root, install the project and analysis-only packages into an isolated environment. The versions used for the frozen results are recorded in `evaluation_manifest.json`; the extra requirements are in `../requirements-analysis.txt`.

```bash
python -m pip install -e .
python -m pip install -r analysis/part1/requirements-analysis.txt
```

## Rebuild

Every stage reads raw data from `--source-root`, sets seed {SEED}, and saves intermediates under this output directory.

```bash
python -m analysis.part1.run audit --source-root /path/to/sendouq_analysis
python -m analysis.part1.run connectivity --source-root /path/to/sendouq_analysis
python -m analysis.part1.run prediction --source-root /path/to/sendouq_analysis
python -m analysis.part1.run plus --source-root /path/to/sendouq_analysis
python -m analysis.part1.run shuffle --source-root /path/to/sendouq_analysis --shuffle-repetitions 200
python -m analysis.part1.run stability --source-root /path/to/sendouq_analysis
python -m analysis.part1.run loto --source-root /path/to/sendouq_analysis
python -m analysis.part1.run properties --source-root /path/to/sendouq_analysis
```

The half-life rows can be run independently (and safely in parallel):

```bash
for setting in 90 120 180 270 365 no_decay; do
  python -m analysis.part1.run_half_life_one \\
    --source-root /path/to/sendouq_analysis \\
    --output-dir analysis/part1/output \\
    --setting "$setting"
done
```

Freeze tables, figures, reports, notebook, hashes, and the aggregate results file:

```bash
python -m analysis.part1.report --output-dir analysis/part1/output
python -m analysis.part1.run collect \\
  --source-root /path/to/sendouq_analysis \\
  --output-dir analysis/part1/output
```

The `collect` stage does not read raw data; `--source-root` remains required by the common CLI parser.

## Stage semantics

- `audit`: inventory, duplicate/timestamp audit, reproduction ledger, evaluation manifest.
- `connectivity`: graph component, teammate/opponent, lineup concentration diagnostics.
- `prediction`: strict event-by-event walk-forward ratings, prior-only calibration, paired cluster bootstrap, reliability, unfamiliar-lineup analysis.
- `plus`: institutional and tier/status-stratified AUCs with activity baselines.
- `shuffle`: full LOOPR recomputation after outcome-direction randomization.
- `stability`: adjacent monthly top-100 membership, ordering, and movement.
- `loto`: full leave-one-tournament-out sensitivity analysis.
- `properties`: schedule, residual volume, smoothing perturbation, and event-weighting checks.
- `run_half_life_one`: one predeclared decay setting under the unified event sequence.
- `report`: consistency assertions and all reader-facing deliverables.

## Primary artifacts

- `technical_report.md`
- `data_methodology_audit.md`
- `evaluation_manifest.json`
- `claim_ledger.csv` and `claim_ledger.md`
- `revised_evaluation_section.md`
- `part1_results.ipynb`
- `results/results.json`
- `artifact_index.csv`
- `audit/project_file_inventory.csv`

Large checkpoint files live in `cache/`; auditable analysis intermediates live in `intermediate/` and `tables/`.
"""
    text = dedent(text).strip()
    write_text(output / "README.md", text)
    return text


def artifact_index(output: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(output.rglob("*")):
        if (
            not path.is_file()
            or path.name == "artifact_index.csv"
            or "cache" in path.parts
        ):
            continue
        digest = sha256(path.read_bytes()).hexdigest()
        rows.append(
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "artifact_index.csv", index=False)
    return frame


def collect_results(output: Path) -> None:
    results = {}
    for path in sorted((output / "results").glob("*.json")):
        if path.name != "results.json":
            results[path.stem] = read_json(path)
    write_json(output / "results" / "results.json", results)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    configure_plots()

    half_life = merge_half_lives(output)
    loto_frame, loto_summary = normalize_loto(output)
    manifest = read_json(output / "evaluation_manifest.json")
    prediction = pd.read_csv(output / "tables" / "prediction_summary.csv")
    plus_methods = pd.read_csv(output / "tables" / "plus_methods.csv")
    checks = consistency_checks(
        manifest, prediction, plus_methods, half_life, loto_frame
    )

    data = {
        "manifest": manifest,
        "audit": read_json(output / "results" / "data_audit.json"),
        "connectivity": read_json(output / "results" / "connectivity.json"),
        "prediction": prediction,
        "calibration": pd.read_csv(output / "tables" / "loopr_calibration.csv"),
        "familiarity": pd.read_csv(output / "tables" / "prediction_by_familiarity.csv"),
        "plus": read_json(output / "results" / "plus.json"),
        "plus_methods": plus_methods,
        "plus_strata": pd.read_csv(output / "tables" / "plus_strata.csv"),
        "shuffle": read_json(output / "results" / "outcome_shuffle.json"),
        "shuffle_draws": pd.read_csv(output / "tables" / "outcome_shuffle_null.csv"),
        "stability": pd.read_csv(output / "tables" / "monthly_stability.csv"),
        "stability_summary": read_json(output / "results" / "stability.json"),
        "loto_frame": loto_frame,
        "loto": loto_summary,
        "half_life": half_life,
        "properties": read_json(output / "results" / "property_tests.json"),
        "smoothing": pd.read_csv(
            output / "tables" / "smoothing_perturbation_summary.csv"
        ),
        "reproduction": pd.read_csv(output / "audit" / "reproduction_ledger.csv"),
    }

    main_tables = build_main_tables(output, data)
    plot_prediction(output, prediction)
    plot_calibration(output, data["calibration"])
    plot_plus(output, plus_methods, data["shuffle_draws"], data["shuffle"])
    plot_stability(output, data["stability"])
    plot_loto(output, loto_frame, loto_summary)
    plot_half_life(output, half_life)
    plot_lineup_transfer(output, data["familiarity"])

    claim_ledger(output)
    revised_evaluation_section(output, data)
    methodology_audit(output, data, checks)
    technical_report(output, data, main_tables)
    build_notebook(output)
    readme(output)
    collect_results(output)
    artifact_index(output)
    print(f"Rendered Part I package to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
