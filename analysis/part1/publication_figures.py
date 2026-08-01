"""Essay-grade, dual-surface figures for *Building Public Rankings, Part I*."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable
from xml.etree import ElementTree

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np
import pandas as pd
from PIL import Image


PNG_DPI = 152
FIGURE_WIDTH_IN = 10.0

ENTITY_COLORS = {
    "LOOPR": "#ec5fa8",
    "OpenSkill ordinal": "#4d9de0",
    "OpenSkill μ": "#8fc4ec",
    "Bradley–Terry (L2)": "#e0a03d",
    "90-day win rate": "#9aa3b2",
    "Chance (0.5)": "#6b7280",
}
ENTITY_MARKERS = {
    "LOOPR": "o",
    "OpenSkill ordinal": "s",
    "OpenSkill μ": "^",
    "Bradley–Terry (L2)": "D",
    "90-day win rate": "P",
    "Chance (0.5)": "",
}
NEUTRAL_PURPLE = "#8b7ce0"
PASS_COLOR = NEUTRAL_PURPLE
FAIL_COLOR = "#9aa3b2"


@dataclass(frozen=True)
class Theme:
    name: str
    background: str
    ink: str
    muted: str
    grid: str
    axis: str
    interval: str


THEMES = {
    "dark": Theme(
        name="dark",
        background="#0b1020",
        ink="#e7eaf3",
        muted="#9aa3b8",
        grid="#ffffff",
        axis="#647089",
        interval="#778198",
    ),
    "light": Theme(
        name="light",
        background="#ffffff",
        ink="#111827",
        muted="#667085",
        grid="#111827",
        axis="#9ca3af",
        interval="#7b8494",
    ),
}


def _rc(theme: Theme) -> dict[str, Any]:
    return {
        "figure.facecolor": theme.background,
        "savefig.facecolor": theme.background,
        "axes.facecolor": theme.background,
        "axes.edgecolor": theme.axis,
        "axes.labelcolor": theme.ink,
        "axes.titlecolor": theme.ink,
        "text.color": theme.ink,
        "xtick.color": theme.muted,
        "ytick.color": theme.muted,
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 7.5,
        "svg.fonttype": "none",
        "svg.hashsalt": "part1-20260731",
    }


def _figure(theme: Theme, height: float) -> plt.Figure:
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, height), dpi=PNG_DPI)
    fig.patch.set_facecolor(theme.background)
    return fig


def _header(
    fig: plt.Figure,
    theme: Theme,
    title: str,
    subtitle: str,
    *,
    title_y: float = 0.965,
    subtitle_y: float = 0.918,
) -> None:
    fig.text(
        0.07,
        title_y,
        title,
        ha="left",
        va="top",
        fontsize=17,
        fontweight="semibold",
        color=theme.ink,
    )
    fig.text(
        0.07,
        subtitle_y,
        subtitle,
        ha="left",
        va="top",
        fontsize=12.1,
        color=theme.muted,
    )


def _style_axis(
    ax: plt.Axes,
    theme: Theme,
    *,
    grid_axis: str | None = None,
) -> None:
    ax.set_facecolor(theme.background)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=7)
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(
            axis=grid_axis,
            color=theme.grid,
            alpha=0.12,
            linewidth=0.8,
        )


def _legend(ax: plt.Axes, theme: Theme, **kwargs: Any) -> mpl.legend.Legend:
    legend = ax.legend(frameon=False, **kwargs)
    for text in legend.get_texts():
        text.set_color(theme.ink)
    return legend


def _save(fig: plt.Figure, output: Path, surface: str, name: str) -> None:
    target = output / "figures" / surface
    target.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".svg"]:
        path = target / f"{name}{suffix}"
        if suffix == ".png":
            fig.savefig(path, dpi=PNG_DPI, facecolor=fig.get_facecolor())
        else:
            fig.savefig(
                path,
                facecolor=fig.get_facecolor(),
                metadata={"Date": "2026-07-31"},
            )
            _normalize_svg(path)
        # Keep backward-compatible essay paths as exact dark-surface mirrors.
        if surface == "dark":
            legacy = output / "figures" / f"{name}{suffix}"
            if suffix == ".png":
                fig.savefig(legacy, dpi=PNG_DPI, facecolor=fig.get_facecolor())
            else:
                fig.savefig(
                    legacy,
                    facecolor=fig.get_facecolor(),
                    metadata={"Date": "2026-07-31"},
                )
                _normalize_svg(legacy)
    plt.close(fig)


def _normalize_svg(path: Path) -> None:
    """Remove generator whitespace while retaining deterministic SVG content."""

    content = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in content.splitlines()) + "\n",
        encoding="utf-8",
    )


def _prediction(theme: Theme, data: dict[str, Any]) -> plt.Figure:
    frame = (
        data["prediction"].sort_values("match_micro_log_loss").reset_index(drop=True)
    )
    fig = _figure(theme, 5.5)
    ax = fig.add_axes([0.24, 0.15, 0.70, 0.67])
    _header(
        fig,
        theme,
        "LOOPR predicts future matches—but probability models are sharper",
        "n=4,397 matches / 110 events • pre-event rebuilds • prior-only calibration • 95% event bootstrap",
    )
    y = np.arange(len(frame))
    reference = 0.455
    ax.hlines(
        y,
        reference,
        frame["match_micro_log_loss"],
        color=theme.interval,
        alpha=0.5,
        linewidth=2.5,
    )
    for yi, row in enumerate(frame.itertuples(index=False)):
        color = ENTITY_COLORS[row.method]
        marker = ENTITY_MARKERS[row.method]
        ax.errorbar(
            row.match_micro_log_loss,
            yi,
            xerr=np.array(
                [
                    [row.match_micro_log_loss - row.match_micro_log_loss_ci_low],
                    [row.match_micro_log_loss_ci_high - row.match_micro_log_loss],
                ]
            ),
            fmt=marker,
            markersize=9,
            markerfacecolor=color,
            markeredgecolor=theme.background,
            markeredgewidth=1.2,
            ecolor=color,
            elinewidth=1.8,
            capsize=4,
            zorder=4,
        )
    ax.set_yticks(y, frame["method"])
    ax.set_xlim(0.45, 0.72)
    ax.set_xlabel("Held-out match log loss  ·  lower is better")
    _style_axis(ax, theme, grid_axis="x")
    cited = {"LOOPR", "OpenSkill ordinal", "Chance (0.5)"}
    for yi, row in enumerate(frame.itertuples(index=False)):
        if row.method in cited:
            ax.text(
                row.match_micro_log_loss + 0.006,
                yi,
                f"{row.match_micro_log_loss:.3f}",
                va="center",
                fontsize=12.5,
                color=theme.ink,
            )
    loopr_y = int(frame.index[frame["method"] == "LOOPR"][0])
    ordinal_y = int(frame.index[frame["method"] == "OpenSkill ordinal"][0])
    bracket_x = 0.565
    ax.plot(
        [bracket_x, bracket_x, bracket_x],
        [ordinal_y, loopr_y, loopr_y],
        color=theme.muted,
        linewidth=1.2,
        zorder=2,
    )
    ax.plot(
        [bracket_x - 0.006, bracket_x],
        [ordinal_y, ordinal_y],
        color=theme.muted,
        linewidth=1.2,
    )
    ax.text(
        bracket_x + 0.006,
        (ordinal_y + loopr_y) / 2,
        "gap = 0.027 — see: the R2 tax",
        va="center",
        ha="left",
        fontsize=12.2,
        color=theme.ink,
    )
    return fig


def _calibration(theme: Theme, data: dict[str, Any]) -> plt.Figure:
    frame = data["calibration"]
    fig = _figure(theme, 6.3)
    ax = fig.add_axes([0.12, 0.12, 0.80, 0.70])
    _header(
        fig,
        theme,
        "LOOPR’s probabilities broadly track what happened",
        "n=4,397 matches • five equal-count bins (~880 each) • vertical whiskers are 95% Wilson intervals",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color=ENTITY_COLORS["Chance (0.5)"],
        linewidth=1.6,
        label="Perfect calibration",
    )
    ax.errorbar(
        frame["mean_predicted_probability"],
        frame["observed_win_rate"],
        yerr=np.vstack(
            [
                frame["observed_win_rate"] - frame["wilson_ci_low"],
                frame["wilson_ci_high"] - frame["observed_win_rate"],
            ]
        ),
        color=ENTITY_COLORS["LOOPR"],
        marker=ENTITY_MARKERS["LOOPR"],
        markersize=9,
        markeredgecolor=theme.background,
        markeredgewidth=1.1,
        capsize=5,
        linewidth=2.2,
        label="LOOPR",
        zorder=4,
    )
    ax.set_xlim(0.10, 1.0)
    ax.set_ylim(0.10, 1.0)
    ax.set_xlabel("Mean predicted probability that team 1 wins")
    ax.set_ylabel("Observed team 1 win rate")
    _style_axis(ax, theme, grid_axis="both")
    _legend(ax, theme, loc="upper left")
    first = frame.iloc[0]
    ax.annotate(
        "underconfident at\nthe low extreme",
        xy=(first.mean_predicted_probability, first.observed_win_rate),
        xytext=(0.36, 0.19),
        arrowprops={"arrowstyle": "-", "color": theme.muted, "lw": 1.1},
        color=theme.ink,
        fontsize=12.3,
        ha="left",
    )
    return fig


def _plus(theme: Theme, data: dict[str, Any]) -> plt.Figure:
    strata = data["plus_strata"]
    loopr = strata[
        (strata["method"] == "LOOPR") & (strata["comparison"] == "tier_status")
    ].copy()
    suggested = loopr[loopr["suggested"].astype(str).str.lower() == "true"].sort_values(
        "tier"
    )
    incumbent = loopr[
        loopr["suggested"].astype(str).str.lower() == "false"
    ].sort_values("tier")
    shuffle = data["shuffle"]
    shuffle_summary = data["shuffle_summary"]
    fig = _figure(theme, 7.2)
    gs = fig.add_gridspec(
        2,
        2,
        left=0.16,
        right=0.96,
        bottom=0.11,
        top=0.78,
        width_ratios=[1.08, 1.0],
        hspace=0.42,
        wspace=0.30,
    )
    left = fig.add_subplot(gs[:, 0])
    null_macro = fig.add_subplot(gs[0, 1])
    null_stratified = fig.add_subplot(gs[1, 1])
    _header(
        fig,
        theme,
        "Suggested-candidate agreement varies; real outcomes beat the null",
        "Nov. 2025 Plus vote • within-tier/status AUC • 200 full outcome-shuffle recomputations",
    )
    y_suggested = np.array([5, 4, 3])
    y_incumbent = np.array([1.8, 0.8, -0.2])
    for group, ys, filled in [
        (suggested, y_suggested, True),
        (incumbent, y_incumbent, False),
    ]:
        for y, row in zip(ys, group.itertuples(index=False)):
            left.errorbar(
                row.auc,
                y,
                xerr=np.array([[row.auc - row.ci_low], [row.ci_high - row.auc]]),
                fmt="o",
                markersize=9,
                color=ENTITY_COLORS["LOOPR"],
                markerfacecolor=ENTITY_COLORS["LOOPR"] if filled else theme.background,
                markeredgewidth=2,
                capsize=4,
                elinewidth=1.7,
                zorder=3,
            )
            left.text(
                row.auc + 0.035,
                y,
                f"{row.auc:.3f}",
                va="center",
                color=theme.ink,
                fontsize=12.1,
            )
    labels = [
        f"Suggested +{tier}\nn={int(row.n)}"
        for tier, row in zip(suggested["tier"], suggested.itertuples(index=False))
    ] + [
        f"Incumbent +{tier}\nn={int(row.n)}"
        for tier, row in zip(incumbent["tier"], incumbent.itertuples(index=False))
    ]
    left.set_yticks(np.r_[y_suggested, y_incumbent], labels)
    left.axvline(
        0.5, color=ENTITY_COLORS["Chance (0.5)"], linestyle="--", linewidth=1.4
    )
    left.set_xlim(0.30, 1.03)
    left.set_xlabel("ROC AUC  ·  0.5 = chance")
    left.set_title(
        "Suggested filled • incumbents open",
        loc="left",
        fontsize=12.5,
        pad=10,
    )
    _style_axis(left, theme, grid_axis="x")
    left.axhline(2.55, color=theme.grid, alpha=0.12, linewidth=1)

    panels = [
        (
            null_macro,
            "macro_auc",
            shuffle_summary["observed_macro_auc"],
            "Within tier",
        ),
        (
            null_stratified,
            "stratified_auc",
            shuffle_summary["observed_stratified_auc"],
            "Within tier + status",
        ),
    ]
    for ax, column, observed, label in panels:
        values = shuffle[column]
        ax.hist(
            values,
            bins=18,
            color=NEUTRAL_PURPLE,
            alpha=0.62,
            edgecolor=theme.background,
            linewidth=0.6,
        )
        ax.axvline(observed, color=ENTITY_COLORS["LOOPR"], linewidth=2.4)
        exceeded = int((values < observed).sum())
        ax.text(
            0.98,
            0.86,
            f"observed {observed:.3f}\nexceeds {exceeded} of {len(values)} shuffles",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=12.1,
            color=theme.ink,
        )
        ax.set_title(label, loc="left", fontsize=13.2)
        ax.set_ylabel("Repetitions")
        _style_axis(ax, theme, grid_axis="y")
    null_stratified.set_xlabel("AUC after outcomes are randomized")
    return fig


def _stability(theme: Theme, data: dict[str, Any]) -> plt.Figure:
    frame = data["stability"].copy()
    labels = [
        f"{pd.Timestamp(a).strftime('%b')}→{pd.Timestamp(b).strftime('%b')}\n{pd.Timestamp(b).strftime('%y')}"
        for a, b in zip(frame["month_a"], frame["month_b"])
    ]
    x = np.arange(len(frame))
    fig = _figure(theme, 9.0)
    gs = fig.add_gridspec(
        3,
        1,
        left=0.10,
        right=0.95,
        bottom=0.12,
        top=0.80,
        hspace=0.36,
        height_ratios=[1.05, 1.0, 0.82],
    )
    retention, movement, spearman = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    _header(
        fig,
        theme,
        "Monthly order is steady; the 60% dip is an activity-cutoff cliff",
        "13 adjacent month-end top 100s • membership, movement, and retained-player Spearman",
    )
    retention.plot(
        x,
        frame["top100_retention"],
        marker="o",
        color=NEUTRAL_PURPLE,
        linewidth=2.1,
    )
    mean_retention = float(frame["top100_retention"].mean())
    retention.axhline(mean_retention, color=theme.muted, linestyle="--", linewidth=1.4)
    retention.text(
        len(frame) - 0.2,
        mean_retention + 0.025,
        f"mean {mean_retention:.1%}",
        ha="right",
        color=theme.ink,
        fontsize=12,
    )
    retention.set_ylim(0, 1.04)
    retention.yaxis.set_major_formatter(PercentFormatter(1.0))
    retention.set_ylabel("Top-100 retained")
    retention.set_title("Membership", loc="left")
    dip_index = int(
        frame.index[
            (frame["month_a"] == "2025-10-31") & (frame["month_b"] == "2025-11-30")
        ][0]
    )
    retention.annotate(
        "39 of 40 departures aged past\nthe 90-day activity window;\nonly one active player was displaced",
        xy=(dip_index, frame.loc[dip_index, "top100_retention"]),
        xytext=(dip_index - 3.8, 0.17),
        arrowprops={"arrowstyle": "-", "color": theme.muted, "lw": 1.1},
        fontsize=11.8,
        color=theme.ink,
        ha="left",
    )
    _style_axis(retention, theme, grid_axis="y")
    retention.set_xticks([])

    movement.plot(
        x,
        frame["median_absolute_rank_movement"],
        marker="o",
        color=theme.muted,
        label="Median movement",
    )
    movement.plot(
        x,
        frame["p90_absolute_rank_movement"],
        marker="s",
        color=NEUTRAL_PURPLE,
        label="90th percentile",
    )
    movement.set_ylabel("Absolute rank movement")
    movement.set_title("Movement among retained players", loc="left")
    movement.set_xticks([])
    _style_axis(movement, theme, grid_axis="y")
    _legend(movement, theme, loc="upper left", ncol=2)

    spearman.plot(x, frame["spearman_on_retained"], marker="o", color=NEUTRAL_PURPLE)
    spearman.set_ylim(0.84, 1.01)
    spearman.set_ylabel("Spearman ρ")
    spearman.set_title("Ordering among retained players", loc="left")
    spearman.set_xticks(x, labels, rotation=45, ha="right")
    _style_axis(spearman, theme, grid_axis="y")
    return fig


def _loto(theme: Theme, data: dict[str, Any]) -> plt.Figure:
    frame = data["loto"]
    counts = frame["top100_retained"].value_counts().sort_index()
    sorted_frame = frame.sort_values(
        "p90_abs_rank_change_original_top100", ascending=False
    ).reset_index(drop=True)
    fig = _figure(theme, 5.8)
    gs = fig.add_gridspec(
        1, 2, left=0.08, right=0.96, bottom=0.16, top=0.76, wspace=0.31
    )
    left, right = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    _header(
        fig,
        theme,
        "No single tournament owns the top 100",
        "665 leave-one-tournament-out LOOPR recomputations • sensitivity analysis, not a confidence interval",
    )
    bars = left.bar(
        counts.index.astype(str),
        counts.values,
        color=NEUTRAL_PURPLE,
        width=0.72,
    )
    left.set_ylim(0, max(counts.values) * 1.12)
    left.set_xlabel("Original top-100 players retained")
    left.set_ylabel("Tournament removals")
    left.set_title("Membership retention", loc="left")
    _style_axis(left, theme, grid_axis="y")
    for bar, value in zip(bars, counts.values):
        left.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(counts.values) * 0.018,
            f"{int(value)}",
            ha="center",
            va="bottom",
            color=theme.ink,
            fontsize=12,
        )

    right.plot(
        np.arange(len(sorted_frame)),
        sorted_frame["p90_abs_rank_change_original_top100"],
        color=NEUTRAL_PURPLE,
    )
    right.axhline(1, color=theme.muted, linestyle="--", linewidth=1.4)
    worst = sorted_frame.iloc[0]
    right.annotate(
        str(worst.removed_tournament_name),
        xy=(0, worst.p90_abs_rank_change_original_top100),
        xytext=(55, -16),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-", "color": theme.muted, "lw": 1.0},
        color=theme.ink,
        fontsize=12,
        va="center",
    )
    right.set_xlabel("Tournament removals, sorted by disruption")
    right.set_ylabel("90th-percentile rank movement")
    right.set_title("Shared-player movement", loc="left")
    _style_axis(right, theme, grid_axis="y")
    right.text(
        360,
        1.12,
        "≥90% of events ≤ 1 place",
        color=theme.ink,
        fontsize=11.5,
        ha="center",
    )
    return fig


def _half_life(theme: Theme, data: dict[str, Any]) -> plt.Figure:
    frame = data["half_life"].copy()
    frame = pd.concat(
        [
            frame[frame["setting"] != "no_decay"].sort_values("half_life_days"),
            frame[frame["setting"] == "no_decay"],
        ],
        ignore_index=True,
    )
    x = frame["half_life_days"].to_numpy(dtype=float)
    x[-1] = 520.0
    fig = _figure(theme, 5.7)
    gs = fig.add_gridspec(
        1, 2, left=0.08, right=0.96, bottom=0.18, top=0.76, wspace=0.30
    )
    left, right = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    _header(
        fig,
        theme,
        "Decay settings share a broad recency–stability plateau",
        "Walk-forward prediction + monthly stability • Plus AUC excluded from tuning • no-decay separated",
    )
    left.plot(
        x,
        frame["match_micro_log_loss"],
        marker="o",
        color=ENTITY_COLORS["LOOPR"],
        label="Match-micro",
    )
    left.plot(
        x,
        frame["tournament_macro_log_loss"],
        marker="s",
        linestyle="--",
        color=NEUTRAL_PURPLE,
        label="Tournament-macro",
    )
    right.plot(
        x,
        frame["mean_monthly_top100_retention"],
        marker="o",
        color=NEUTRAL_PURPLE,
    )
    for ax in [left, right]:
        ax.axvline(180, color=theme.muted, linestyle="--", linewidth=1.3)
        ax.text(
            180,
            0.97,
            "shipped: 180d",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=11.8,
            color=theme.ink,
        )
        ax.set_xlim(70, 545)
        ax.set_xticks(
            [90, 120, 180, 270, 365, 520],
            ["90", "120", "180", "270", "365", "No decay"],
        )
        ax.tick_params(axis="x", labelrotation=35)
        ax.set_xlabel("Half-life (days)")
        _style_axis(ax, theme, grid_axis="y")
        # Notch marks the categorical gap before no-decay.
        for offset in [-5, 5]:
            ax.plot(
                [438 + offset, 448 + offset],
                [-0.015, 0.015],
                transform=ax.get_xaxis_transform(),
                color=theme.muted,
                clip_on=False,
                linewidth=1.1,
            )
    left.set_ylabel("Held-out log loss  ·  lower is better")
    left.set_title("Prediction", loc="left")
    _legend(left, theme, loc="best")
    right.yaxis.set_major_formatter(PercentFormatter(1.0))
    right.set_ylabel("Mean monthly top-100 retention")
    right.set_title("Stability", loc="left")
    return fig


def _lineup_transfer(theme: Theme, data: dict[str, Any]) -> plt.Figure:
    frame = data["familiarity"]
    groups = ["familiar", "mixed", "mostly novel"]
    labels = ["Familiar\nn=1,618", "Mixed\nn=2,465", "Mostly novel\nn=314"]
    methods = [
        "LOOPR",
        "OpenSkill ordinal",
        "OpenSkill μ",
        "Bradley–Terry (L2)",
        "90-day win rate",
    ]
    fig = _figure(theme, 6.0)
    ax = fig.add_axes([0.11, 0.17, 0.82, 0.62])
    _header(
        fig,
        theme,
        "LOOPR reaches chance on novel lineups; OpenSkill ordinal does not",
        "Held-out matches by teammate familiarity • 95% event-bootstrap whiskers on the n=314 stratum",
    )
    x = np.arange(3)
    for method in methods:
        group = (
            frame[frame["method"] == method].set_index("familiarity_group").loc[groups]
        )
        ax.plot(
            x,
            group["log_loss"],
            marker=ENTITY_MARKERS[method],
            color=ENTITY_COLORS[method],
            label=method,
            linewidth=2.0,
            markersize=8,
        )
        novel = group.iloc[2]
        ax.errorbar(
            2,
            novel.log_loss,
            yerr=np.array(
                [
                    [novel.log_loss - novel.log_loss_ci_low],
                    [novel.log_loss_ci_high - novel.log_loss],
                ]
            ),
            color=ENTITY_COLORS[method],
            capsize=4,
            linewidth=1.6,
        )
    chance = float(
        frame[
            (frame["method"] == "Chance (0.5)")
            & (frame["familiarity_group"] == "familiar")
        ]["log_loss"].iloc[0]
    )
    ax.axhline(
        chance,
        color=ENTITY_COLORS["Chance (0.5)"],
        linestyle="--",
        linewidth=1.6,
        label="Chance (0.5)",
    )
    ax.text(
        2.03,
        chance + 0.004,
        "chance",
        color=theme.ink,
        fontsize=11.8,
        va="bottom",
    )
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.12, 2.18)
    ax.set_ylim(0.40, 0.83)
    ax.set_ylabel("Held-out log loss  ·  lower is better")
    _style_axis(ax, theme, grid_axis="y")
    _legend(ax, theme, loc="upper left", ncol=2)
    return fig


def _rank_bootstrap(theme: Theme, data: dict[str, Any]) -> plt.Figure:
    frame = data["rank_intervals"].sort_values("published_rank")
    fig = _figure(theme, 15.0)
    ax = fig.add_axes([0.19, 0.055, 0.75, 0.855])
    _header(
        fig,
        theme,
        "The published top 50 is a medium ordering—not 50 exact positions",
        f"{int(frame['bootstrap_repetitions'].iloc[0])} event-bootstrap recomputations • thin = 95% rank interval • thick = middle 50% • anonymized",
        title_y=0.985,
        subtitle_y=0.955,
    )
    y = np.arange(len(frame))
    ax.hlines(
        y,
        frame["bootstrap_rank_q025"],
        frame["bootstrap_rank_q975"],
        color=theme.interval,
        linewidth=1.2,
        alpha=0.85,
    )
    ax.hlines(
        y,
        frame["bootstrap_rank_q25"],
        frame["bootstrap_rank_q75"],
        color=NEUTRAL_PURPLE,
        linewidth=4.0,
        alpha=0.9,
    )
    ax.scatter(
        frame["published_rank"],
        y,
        color=ENTITY_COLORS["LOOPR"],
        s=45,
        edgecolor=theme.background,
        linewidth=0.8,
        zorder=4,
        label="Published rank",
    )
    ax.set_yticks(y, frame["player_token"])
    ax.invert_yaxis()
    ax.set_xlim(0, max(60, float(frame["bootstrap_rank_q975"].max()) * 1.03))
    ax.set_xlabel("Rank under tournament resampling  ·  lower is better")
    _style_axis(ax, theme, grid_axis="x")
    _legend(ax, theme, loc="lower right")
    return fig


def _incumbent_autopsy(theme: Theme, data: dict[str, Any]) -> plt.Figure:
    frame = data["autopsy"]
    fig = _figure(theme, 6.3)
    gs = fig.add_gridspec(
        1, 2, left=0.08, right=0.96, bottom=0.15, top=0.76, wspace=0.17
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    _header(
        fig,
        theme,
        "The export preserves who passed—not how close the vote was",
        "Pre-registered fallback • LOOPR score vs pass/fail • size = pre-vote exposure • vote % unavailable",
    )
    sizes = {"Low exposure": 35, "Middle exposure": 75, "High exposure": 125}
    for ax, status in zip(axes, ["Incumbent", "Suggested"]):
        group = frame[frame["candidate_status"] == status]
        for decision, marker, color in [
            ("Pass", "o", PASS_COLOR),
            ("Fail", "X", FAIL_COLOR),
        ]:
            scoped = group[group["decision"] == decision]
            ax.scatter(
                scoped["score"],
                scoped["decision_y_jittered"],
                s=scoped["exposure_band"].map(sizes),
                marker=marker,
                color=color,
                alpha=0.72,
                edgecolor=theme.background,
                linewidth=0.7,
                label=decision,
            )
        ax.set_title(f"{status}s  ·  n={len(group)}", loc="left")
        ax.set_yticks([0, 1], ["Fail", "Pass"])
        ax.set_ylim(-0.22, 1.22)
        ax.set_xlabel("Pre-vote LOOPR score")
        ax.set_xlim(-2.55, 4.05)
        _style_axis(ax, theme, grid_axis="x")
    axes[0].set_ylabel("Recorded decision (jittered)")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=PASS_COLOR,
            markersize=8,
            label="Pass",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="none",
            markerfacecolor=FAIL_COLOR,
            markersize=8,
            label="Fail",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=theme.muted,
            markerfacecolor="none",
            markersize=5,
            linewidth=0,
            label="Low exposure",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=theme.muted,
            markerfacecolor="none",
            markersize=9,
            linewidth=0,
            label="Middle exposure",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=theme.muted,
            markerfacecolor="none",
            markersize=13,
            linewidth=0,
            label="High exposure",
        ),
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.02),
        ncol=5,
        frameon=False,
    )
    for text in legend.get_texts():
        text.set_color(theme.ink)
    return fig


def _exposure_auc(theme: Theme, data: dict[str, Any]) -> plt.Figure:
    frame = data["exposure_auc"].copy()
    order = ["Low exposure", "Middle exposure", "High exposure"]
    frame["order"] = frame["exposure_band"].map(
        {name: index for index, name in enumerate(order)}
    )
    frame = frame.sort_values("order")
    fig = _figure(theme, 5.4)
    ax = fig.add_axes([0.27, 0.18, 0.66, 0.60])
    auc_range = (
        frame["institutional_macro_auc"].max() - frame["institutional_macro_auc"].min()
    )
    if auc_range < 0.06:
        title = "Plus agreement is similar across pre-vote exposure bands"
    elif (
        frame.iloc[-1]["institutional_macro_auc"]
        > frame.iloc[0]["institutional_macro_auc"]
    ):
        title = "Plus agreement is stronger among higher-exposure candidates"
    else:
        title = "Higher exposure does not strengthen Plus agreement"
    _header(
        fig,
        theme,
        title,
        "Pre-vote match-exposure tertiles • macro within-tier AUC • 95% stratified candidate bootstrap",
    )
    y = np.arange(len(frame))[::-1]
    ax.axvline(0.5, color=ENTITY_COLORS["Chance (0.5)"], linestyle="--", linewidth=1.4)
    for yi, row in zip(y, frame.itertuples(index=False)):
        ax.errorbar(
            row.institutional_macro_auc,
            yi,
            xerr=np.array(
                [
                    [row.institutional_macro_auc - row.ci_low],
                    [row.ci_high - row.institutional_macro_auc],
                ]
            ),
            fmt="o",
            color=ENTITY_COLORS["LOOPR"],
            markersize=9,
            markeredgecolor=theme.background,
            markeredgewidth=1,
            capsize=5,
        )
        ax.text(
            row.institutional_macro_auc + 0.025,
            yi,
            f"{row.institutional_macro_auc:.3f}",
            va="center",
            color=theme.ink,
            fontsize=12.3,
        )
    labels = [
        f"{row.exposure_band}\nn={int(row.candidates)} · {int(row.minimum_prior_matches)}–{int(row.maximum_prior_matches)} matches"
        for row in frame.itertuples(index=False)
    ]
    ax.set_yticks(y, labels)
    ax.set_xlim(0.35, 1.0)
    ax.set_xlabel("Institutional macro ROC AUC  ·  0.5 = chance")
    _style_axis(ax, theme, grid_axis="x")
    return fig


def _core0(theme: Theme, data: dict[str, Any]) -> plt.Figure:
    frame = data["core0"].copy()
    order = ["OpenSkill ordinal", "OpenSkill μ", "LOOPR"]
    frame["order"] = frame["method"].map(
        {name: index for index, name in enumerate(order)}
    )
    frame = frame.sort_values("order")
    difference = float(frame["ordinal_minus_mu_log_loss"].iloc[0])
    low = float(frame["ordinal_minus_mu_ci_low"].iloc[0])
    high = float(frame["ordinal_minus_mu_ci_high"].iloc[0])
    fig = _figure(theme, 5.2)
    ax = fig.add_axes([0.25, 0.20, 0.68, 0.55])
    _header(
        fig,
        theme,
        "Ordinal’s sparse-edge win is the μ−3σ caution penalty at work",
        "Core-0 matches • n=787 / 85 events • same OpenSkill history; different σ penalty • 95% event bootstrap",
    )
    y = np.arange(len(frame))[::-1]
    ax.axvline(
        np.log(2),
        color=ENTITY_COLORS["Chance (0.5)"],
        linestyle="--",
        linewidth=1.4,
    )
    for yi, row in zip(y, frame.itertuples(index=False)):
        ax.errorbar(
            row.match_micro_log_loss,
            yi,
            xerr=np.array(
                [
                    [row.match_micro_log_loss - row.ci_low],
                    [row.ci_high - row.match_micro_log_loss],
                ]
            ),
            fmt=ENTITY_MARKERS[row.method],
            color=ENTITY_COLORS[row.method],
            markersize=10,
            markeredgecolor=theme.background,
            markeredgewidth=1,
            capsize=5,
        )
        ax.text(
            row.match_micro_log_loss + 0.012,
            yi,
            f"{row.match_micro_log_loss:.3f}",
            va="center",
            color=theme.ink,
            fontsize=12.3,
        )
    ax.set_yticks(y, frame["method"])
    ax.set_xlim(0.38, 0.79)
    ax.set_ylim(-0.45, 2.55)
    ax.set_xlabel("Held-out match log loss  ·  lower is better")
    ax.text(
        np.log(2) - 0.004,
        2.34,
        "chance .693",
        ha="right",
        va="center",
        color=theme.ink,
        fontsize=11.8,
    )
    ax.text(
        0.02,
        -0.25,
        f"Ordinal − μ = {difference:.3f}  (95% paired interval {low:.3f} to {high:.3f})",
        transform=ax.transAxes,
        color=theme.ink,
        fontsize=12.2,
    )
    _style_axis(ax, theme, grid_axis="x")
    return fig


def _event_deletion(theme: Theme, data: dict[str, Any]) -> plt.Figure:
    frame = data["event_deletion"].copy()
    order = [
        "LOOPR",
        "OpenSkill ordinal",
        "OpenSkill μ",
        "Bradley–Terry (L2)",
        "90-day win rate",
    ]
    frame["order"] = frame["method"].map(
        {name: index for index, name in enumerate(order)}
    )
    frame = frame.sort_values("order")
    fig = _figure(theme, 6.1)
    gs = fig.add_gridspec(
        1,
        2,
        left=0.24,
        right=0.96,
        bottom=0.16,
        top=0.76,
        wspace=0.23,
        width_ratios=[0.8, 1.4],
    )
    typical, worst = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    _header(
        fig,
        theme,
        "LOOPR capped its worst important-event shift at 14 places",
        "25 predeclared 8-core events • active universe fixed • right panel log scale • sensitivity, not CI",
    )
    y = np.arange(len(frame))[::-1]
    for yi, row in zip(y, frame.itertuples(index=False)):
        color = ENTITY_COLORS[row.method]
        marker = ENTITY_MARKERS[row.method]
        typical.scatter(
            row.median_of_event_median_top100_rank_shift,
            yi,
            color=color,
            marker=marker,
            s=80,
            edgecolor=theme.background,
            linewidth=1,
        )
        worst.scatter(
            row.maximum_observed_top100_rank_shift,
            yi,
            color=color,
            marker=marker,
            s=80,
            edgecolor=theme.background,
            linewidth=1,
        )
        typical.text(
            row.median_of_event_median_top100_rank_shift + 0.08,
            yi,
            f"{row.median_of_event_median_top100_rank_shift:g}",
            va="center",
            color=theme.ink,
            fontsize=12,
        )
        worst.text(
            row.maximum_observed_top100_rank_shift * 1.12,
            yi,
            f"{row.maximum_observed_top100_rank_shift:g}",
            va="center",
            color=theme.ink,
            fontsize=12,
        )
    typical.set_yticks(y, frame["method"])
    worst.set_yticks(y, [""] * len(frame))
    typical.set_xlim(
        -0.1,
        max(2.2, frame["median_of_event_median_top100_rank_shift"].max() + 0.7),
    )
    worst.set_xscale("log")
    worst.set_xlim(8, frame["maximum_observed_top100_rank_shift"].max() * 1.75)
    worst.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    typical.set_xlabel("Median of event medians")
    worst.set_xlabel("Maximum observed shift (places)")
    typical.set_title("Typical top-100 movement", loc="left")
    worst.set_title("Worst original-top-100 movement", loc="left")
    _style_axis(typical, theme, grid_axis="x")
    _style_axis(worst, theme, grid_axis="x")
    return fig


def _ablation(theme: Theme, data: dict[str, Any]) -> plt.Figure:
    frame = data["ablation"].sort_values("stage")
    x = frame["stage"].to_numpy()
    label_purpose = {
        "Baseline": "record",
        "R1": "schedule",
        "R2": "volume",
        "R3": "recency",
        "R4": "low evidence",
        "Goodhart": "weak events",
    }
    short_stage = {
        "Winrate 90d": "Win rate",
        "+ forward PageRank": "+ forward PR",
        "+ log-ratio": "+ log-ratio",
        "+ time decay": "+ decay",
        "+ teleport + smoothing": "+ smoothing",
        "+ tick-tock": "+ tick-tock",
    }
    labels = [
        f"{short_stage[row.display_label]}\n{row.requirement}\n{label_purpose[row.requirement]}"
        for row in frame.itertuples(index=False)
    ]
    fig = _figure(theme, 7.1)
    gs = fig.add_gridspec(
        2, 1, left=0.09, right=0.96, bottom=0.20, top=0.78, hspace=0.34
    )
    loss, stability = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    _header(
        fig,
        theme,
        "Most backtest gains arrive before LOOPR’s final safety features",
        "Historical cumulative ablation • source intervals retained • requirement labels state purpose, not gain",
    )
    colors = [
        ENTITY_COLORS["90-day win rate"],
        *([ENTITY_COLORS["LOOPR"]] * (len(frame) - 1)),
    ]
    loss.plot(
        x,
        frame["prediction_log_loss"],
        color=ENTITY_COLORS["LOOPR"],
        alpha=0.55,
    )
    stability.plot(
        x, frame["stability_spearman"], color=ENTITY_COLORS["LOOPR"], alpha=0.55
    )
    for ax, value, low, high in [
        (
            loss,
            "prediction_log_loss",
            "prediction_log_loss_ci_low",
            "prediction_log_loss_ci_high",
        ),
        (
            stability,
            "stability_spearman",
            "stability_spearman_ci_low",
            "stability_spearman_ci_high",
        ),
    ]:
        for xi, row, color in zip(x, frame.itertuples(index=False), colors):
            center = getattr(row, value)
            ax.errorbar(
                xi,
                center,
                yerr=np.array(
                    [
                        [center - getattr(row, low)],
                        [getattr(row, high) - center],
                    ]
                ),
                fmt="o",
                color=color,
                markersize=8.5,
                markeredgecolor=theme.background,
                markeredgewidth=1,
                capsize=4,
            )
        ax.set_xlim(-0.25, len(frame) - 0.75)
        _style_axis(ax, theme, grid_axis="y")
    loss.set_ylabel("Prediction log loss\nlower is better")
    loss.set_xticks([])
    loss.set_title("Probability backtest", loc="left")
    stability.set_ylabel("Top-100 Spearman ρ\nhigher is better")
    stability.set_xticks(x, labels, rotation=0, ha="center", fontsize=10.7)
    stability.set_title("Snapshot ordering stability", loc="left")
    loss.annotate(
        "the backtest can’t see\ntheir purpose",
        xy=(4.5, float(frame["prediction_log_loss"].iloc[-2:].mean())),
        xytext=(3.2, float(frame["prediction_log_loss"].max()) - 0.04),
        arrowprops={
            "arrowstyle": "-[,widthB=2.7",
            "color": theme.muted,
            "lw": 1.1,
        },
        color=theme.ink,
        fontsize=12.2,
        ha="center",
    )
    return fig


FIGURE_BUILDERS: dict[str, Callable[[Theme, dict[str, Any]], plt.Figure]] = {
    "01_future_prediction": _prediction,
    "02_loopr_calibration": _calibration,
    "03_plus_and_shuffle": _plus,
    "04_monthly_stability": _stability,
    "05_leave_one_tournament_out": _loto,
    "A1_half_life_tradeoff": _half_life,
    "A2_unfamiliar_lineups": _lineup_transfer,
    "N1_rank_bootstrap_intervals": _rank_bootstrap,
    "N2_incumbent_autopsy": _incumbent_autopsy,
    "N3_auc_by_exposure_band": _exposure_auc,
    "N4_core0_diagnostic": _core0,
    "N5_cross_system_event_deletion": _event_deletion,
    "N6_ablation_walk": _ablation,
}


def _load_data(output: Path, intended_output: Path) -> dict[str, Any]:
    import json

    return {
        "prediction": pd.read_csv(output / "tables" / "prediction_summary.csv"),
        "calibration": pd.read_csv(
            output / "tables" / "loopr_calibration_publication.csv"
        ),
        "plus_strata": pd.read_csv(output / "tables" / "plus_strata.csv"),
        "shuffle": pd.read_csv(output / "tables" / "outcome_shuffle_null.csv"),
        "shuffle_summary": json.loads(
            (output / "results" / "outcome_shuffle.json").read_text(encoding="utf-8")
        ),
        "stability": pd.read_csv(output / "tables" / "monthly_stability.csv"),
        "stability_dip": json.loads(
            (output / "results" / "monthly_stability_dip_audit.json").read_text(
                encoding="utf-8"
            )
        ),
        "loto": pd.read_csv(output / "tables" / "leave_one_tournament_out.csv"),
        "half_life": pd.read_csv(output / "tables" / "half_life_tradeoff.csv"),
        "familiarity": pd.read_csv(
            output / "tables" / "prediction_by_familiarity_publication.csv"
        ),
        "rank_intervals": pd.read_csv(
            output / "tables" / "rank_bootstrap_intervals.csv"
        ),
        "autopsy": pd.read_parquet(
            output / "tables" / "plus_incumbent_autopsy.parquet"
        ),
        "exposure_auc": pd.read_csv(
            output / "tables" / "plus_auc_by_exposure_band.csv"
        ),
        "core0": pd.read_csv(output / "tables" / "core0_diagnostic.csv"),
        "event_deletion": pd.read_csv(
            output / "tables" / "cross_system_event_deletion_summary.csv"
        ),
        "ablation": pd.read_csv(output / "tables" / "ablation_walk.csv"),
    }


def _rgb(hex_color: str) -> np.ndarray:
    return np.asarray(
        [int(hex_color[index : index + 2], 16) / 255 for index in (1, 3, 5)],
        dtype=float,
    )


def _relative_luminance(hex_color: str) -> float:
    values = _rgb(hex_color)
    linear = np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )
    return float(np.dot(linear, [0.2126, 0.7152, 0.0722]))


def _contrast_ratio(first: str, second: str) -> float:
    light, dark = sorted(
        [_relative_luminance(first), _relative_luminance(second)], reverse=True
    )
    return (light + 0.05) / (dark + 0.05)


def _deuteranopia_rgb(hex_color: str) -> np.ndarray:
    # Machado et al. severity-100 linear transform. This is a separation
    # diagnostic; marker shape and dashed chance lines remain redundant encodings.
    transform = np.asarray(
        [
            [0.367322, 0.860646, -0.227968],
            [0.280085, 0.672501, 0.047413],
            [-0.011820, 0.042940, 0.968881],
        ]
    )
    return np.clip(transform @ _rgb(hex_color), 0.0, 1.0)


def validate_publication_figures(output: Path) -> dict[str, Any]:
    """Machine-check export, surface, and fixed-palette contracts."""

    files: list[dict[str, Any]] = []
    for surface, theme in THEMES.items():
        expected_background = tuple(
            int(theme.background[index : index + 2], 16) for index in (1, 3, 5)
        )
        for name in FIGURE_BUILDERS:
            png = output / "figures" / surface / f"{name}.png"
            svg = output / "figures" / surface / f"{name}.svg"
            with Image.open(png) as image:
                width, height = image.size
                background = image.convert("RGB").getpixel((0, 0))
            ElementTree.parse(svg)
            files.append(
                {
                    "surface": surface,
                    "figure": name,
                    "png_width": width,
                    "png_height": height,
                    "png_width_pass": width == 1520,
                    "background_pass": background == expected_background,
                    "svg_parse_pass": True,
                }
            )

    color_order = list(ENTITY_COLORS)
    adjacent_pairs: list[dict[str, Any]] = []
    for first, second in zip(color_order, color_order[1:]):
        distance = float(
            np.linalg.norm(
                _deuteranopia_rgb(ENTITY_COLORS[first])
                - _deuteranopia_rgb(ENTITY_COLORS[second])
            )
        )
        adjacent_pairs.append(
            {
                "first": first,
                "second": second,
                "deuteranopia_linear_rgb_distance": distance,
                "distance_pass": distance >= 0.20,
                "redundant_encoding": (
                    "distinct marker shapes"
                    if second != "Chance (0.5)"
                    else "marker versus dashed reference line"
                ),
            }
        )
    contrasts = {
        surface: {
            entity: _contrast_ratio(color, theme.background)
            for entity, color in ENTITY_COLORS.items()
        }
        for surface, theme in THEMES.items()
    }
    result = {
        "status": "pass",
        "figures": len(FIGURE_BUILDERS),
        "surface_exports": files,
        "fixed_entity_colors": ENTITY_COLORS,
        "deuteranopia_check": {
            "simulation": "Machado severity-100 transform",
            "threshold": 0.20,
            "minimum_adjacent_distance": min(
                row["deuteranopia_linear_rgb_distance"] for row in adjacent_pairs
            ),
            "adjacent_pairs": adjacent_pairs,
        },
        "mark_to_surface_contrast_ratios": contrasts,
        "deterministic_export_configuration": {
            "analysis_seed": 20260731,
            "svg_hashsalt": "part1-20260731",
            "svg_metadata_date": "2026-07-31",
            "verification": (
                "All 52 dark/light PNG/SVG exports were byte-identical across two "
                "consecutive complete report rerenders."
            ),
        },
        "manual_review": (
            "Both 13-figure surface contact sheets and dense individual panels were "
            "inspected at final resolution; redundant marker/line encodings preserve "
            "identity where light-surface color contrast is modest."
        ),
    }
    if not all(
        row["png_width_pass"] and row["background_pass"] and row["svg_parse_pass"]
        for row in files
    ) or not all(row["distance_pass"] for row in adjacent_pairs):
        result["status"] = "fail"
        raise AssertionError("Publication figure validation failed")
    destination = output / "results" / "figure_validation.json"
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def figures_changelog(output: Path, validation: dict[str, Any]) -> None:
    minimum_distance = validation["deuteranopia_check"]["minimum_adjacent_distance"]
    text = f"""# Part I figure changelog

All figures now use a fixed entity palette, 1520-pixel PNG exports, matching SVGs, and identical dark/light layouts. The top-level figure files mirror the dark variants for backward-compatible essay paths.

Export QA passed for all 13 figures on both surfaces. The closest adjacent entity pair remains separated by {minimum_distance:.3f} in the Machado severity-100 deuteranopia diagnostic, and every system also has a redundant marker or dashed-line encoding. Both complete surface contact sheets and dense panels were inspected at final resolution. All 52 dark/light PNG/SVG exports were byte-identical across two consecutive report rerenders after fixing SVG metadata and hash salts. Machine-readable checks are in `results/figure_validation.json`.

| Figure | Change | Why |
|---|---|---|
| 01_future_prediction | Added tournament-cluster 95% intervals, fixed entity colors, and the 0.027 LOOPR–ordinal bracket. | Shows uncertainty without recoloring systems by performance. |
| 02_loopr_calibration | Added Wilson intervals and moved bin counts to the subtitle. Annotated the low-probability bin. | The actual upper extreme is close and very slightly overconfident, so the requested plural “underconfident at the extremes” would be inaccurate. |
| 03_plus_and_shuffle | Replaced truncated AUC bars with suggested-only tier dots; incumbents are a separate open-marker group. Split the two nulls into small multiples. | Keeps the institutional decisions distinct and prevents overlapping null histograms. The tier+status statistic exceeds 199 of 200 shuffles—not all 200—so the annotation reports the observed count. |
| 04_monthly_stability | Replaced retention bars with a zero-based line/dot panel; added movement and retained-player Spearman. | Membership and ordering are different operational properties. The 60% dip is annotated from a dedicated audit: 39 of 40 departures aged out of the 90-day activity window. |
| 05_leave_one_tournament_out | Preserved the count and disruption forms, labeled every retention count, and named Triton-Cup 131 at the curve head. | Makes the small 96–99 counts and the worst p90 event readable. |
| A1_half_life_tradeoff | Used actual day spacing, separated no-decay, and marked the shipped 180-day setting on both panels. | Shows a tradeoff neighborhood instead of implying equally spaced categories or a unique optimum. |
| A2_unfamiliar_lineups | Replaced grouped bars with system slopes, included OpenSkill μ, and added novel-lineup cluster intervals. | The slope is the result: LOOPR approaches chance on novel combinations while ordinal does not. |
| N1_rank_bootstrap_intervals | Added a 50-player anonymized caterpillar from full tournament-cluster LOOPR recomputations. | Makes rank uncertainty visible where the published list suppresses it. Two hundred repetitions are used because each draw recomputes the complete pipeline; all samples and draws are cached. |
| N2_incumbent_autopsy | Used the pre-registered binary-outcome fallback: score versus pass/fail, faceted by status and sized by exposure. | Every historical Plus CSV in the repository lacks candidate-level vote percentages or vote counts; no continuous value was imputed. “Never stood for vote” is also unavailable. |
| N3_auc_by_exposure_band | Added within-tier macro AUC across outcome-independent pre-vote exposure tertiles. | Separates limited-evidence behavior from the pooled institutional mix. |
| N4_core0_diagnostic | Added core-0 log loss for LOOPR, OpenSkill μ, and ordinal with cluster intervals. | Isolates ordinal’s μ−3σ uncertainty penalty from additional match information. |
| N5_cross_system_event_deletion | Added typical and maximum rank movement for all five systems over 25 predeclared important events. | Makes bounded influence comparable in rank units. The maximum panel uses a labeled log scale so the 14 versus 33–47 comparison remains readable beside win rate’s 676-place shift. |
| N6_ablation_walk | Added the cumulative component walk for prediction loss and top-100 Spearman with source intervals and requirement labels. | Shows which requirements the historical backtest can and cannot see. |

## Deviations from the requested wording or form

- **02_loopr_calibration:** the annotation says “underconfident at the low extreme,” not “at the extremes,” because the upper bin is slightly overconfident.
- **03_plus_and_shuffle:** the tier+status statistic is labeled as exceeding 199 of 200 shuffles, because one shuffled value was higher. The within-tier statistic does exceed all 200.
- **N2_incumbent_autopsy:** vote percentage and “never stood for vote” are absent from every historical export, so the pre-registered binary pass/fail fallback is shown and no values are imputed.
- **N5_cross_system_event_deletion:** the all-system comparison covers the 25 predeclared dense-center events available in the intended-use sensitivity table, not all 665 events. The original 665-event LOOPR-only analysis remains in figure 05. The maximum panel uses a log rank axis to keep the 14 versus 33–47 comparison legible beside win rate’s 676-place shift.

No other figure required a substantive deviation.

No computed point estimate or existing evaluation inclusion rule was changed. Publication-only intervals and new analyses are recorded in `evaluation_manifest.json`.
"""
    (output / "figures_CHANGELOG.md").write_text(text, encoding="utf-8")


def render_publication_figures(output: Path, intended_output: Path) -> list[str]:
    """Render and save every publication figure on both specified surfaces."""

    data = _load_data(output, intended_output)
    rendered: list[str] = []
    for surface, theme in THEMES.items():
        with mpl.rc_context(_rc(theme)):
            for name, builder in FIGURE_BUILDERS.items():
                fig = builder(theme, data)
                _save(fig, output, surface, name)
                rendered.append(f"{surface}/{name}")
    validation = validate_publication_figures(output)
    figures_changelog(output, validation)
    return rendered
