"""
Dashboard utilities for pretty-printing evaluation results.

This module provides clean tabular summaries of the metrics suite
described in plan.md.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from rankings.evaluation.metrics_extras import reliability_diagram
from rankings.evaluation.tournament_metrics import (
    mean_absolute_seed_error,
    ndcg_at_k,
    pairwise_agreement,
    upset_rate_analysis,
)


def format_metrics_table(
    results: Dict[str, Any],
    metrics: Optional[List[str]] = None,
    precision: int = 4,
) -> str:
    """
    Format cross-validation results as a pretty table.

    Parameters
    ----------
    results : Dict[str, Any]
        Results from cross_validate_ratings()
    metrics : Optional[List[str]]
        Specific metrics to include. If None, uses default set.
    precision : int
        Number of decimal places for numeric values

    Returns
    -------
    str
        Formatted table string

    Examples
    --------
    >>> results = {
    ...     'avg_loss': 0.6931, 'std_loss': 0.0123,
    ...     'avg_c_stat': 0.7234, 'std_c_stat': 0.0456,
    ...     'alpha_std': 0.0789
    ... }
    >>> print(format_metrics_table(results))  # doctest: +SKIP
    ┌─────────────────┬──────────┬──────────┬──────────┬──────────┐
    │ Metric          │ Mean     │ Std      │ Min      │ Max      │
    ├─────────────────┼──────────┼──────────┼──────────┼──────────┤
    │ Loss            │   0.6931 │   0.0123 │     --   │     --   │
    │ Concordance     │   0.7234 │   0.0456 │     --   │     --   │
    │ Alpha Std       │   0.0789 │     --   │     --   │     --   │
    └─────────────────┴──────────┴──────────┴──────────┴──────────┘
    """
    if metrics is None:
        metrics = [
            ("Weighted log-loss (entropy)", "loss"),
            ("Concordance", "c_stat"),
            ("Skill Score", "skill_score"),
            ("Upset O/E", "upset_oe"),
            ("Brier Score", "brier"),
            ("High-Conf Acc", "acc_conf"),
            ("Accuracy", "accuracy"),
            ("Mean Prob", "mean_probability"),
            ("Alpha Std", "alpha_std"),
        ]

    # Build table data
    rows = []
    for display_name, key in metrics:
        row = {"Metric": display_name}

        # Special handling for alpha_std (no mean/std aggregation)
        if key == "alpha_std":
            value = results.get("alpha_std", np.nan)
            row["Mean"] = (
                f"{value:.{precision}f}" if np.isfinite(value) else "--"
            )
            row["Std"] = "--"
            row["Min"] = "--"
            row["Max"] = "--"
        else:
            # Standard metrics with mean/std across splits
            mean_key = f"avg_{key}"
            std_key = f"std_{key}"

            mean_val = results.get(mean_key, np.nan)
            std_val = results.get(std_key, np.nan)

            row["Mean"] = (
                f"{mean_val:.{precision}f}" if np.isfinite(mean_val) else "--"
            )
            row["Std"] = (
                f"{std_val:.{precision}f}" if np.isfinite(std_val) else "--"
            )

            # Try to get min/max from split results
            split_results = results.get("split_results", [])
            if split_results and mean_key.replace("avg_", "") in ["loss"]:
                # For loss, extract from main field
                values = [r.get("loss", np.nan) for r in split_results]
            else:
                # For other metrics, extract from avg_ fields
                values = [r.get(mean_key, np.nan) for r in split_results]

            valid_values = [v for v in values if np.isfinite(v)]
            if valid_values:
                row["Min"] = f"{min(valid_values):.{precision}f}"
                row["Max"] = f"{max(valid_values):.{precision}f}"
            else:
                row["Min"] = "--"
                row["Max"] = "--"

        rows.append(row)

    # Filter out rows with no data
    rows = [r for r in rows if r["Mean"] != "--"]

    if not rows:
        return "No metrics available."

    # Create DataFrame for pretty printing
    df = pl.DataFrame(rows)

    # Format as table with box drawing
    return _format_dataframe_table(df)


def format_split_summary(results: Dict[str, Any]) -> str:
    """
    Format a summary of cross-validation splits.

    Parameters
    ----------
    results : Dict[str, Any]
        Results from cross_validate_ratings()

    Returns
    -------
    str
        Formatted summary string
    """
    n_splits = results.get("n_splits", 0)
    avg_loss = results.get("avg_loss", np.nan)
    std_loss = results.get("std_loss", np.nan)
    alpha_std = results.get("alpha_std", np.nan)

    summary = f"""
Cross-Validation Summary
{'=' * 24}
Number of splits: {n_splits}
Average loss: {avg_loss:.4f} ± {std_loss:.4f}
Alpha stability (σ): {alpha_std:.4f}

Key Performance Indicators:
"""

    # Add key metrics if available
    key_metrics = [
        ("Discrimination (c-stat)", "avg_c_stat"),
        ("Skill vs random", "avg_skill_score"),
        ("Calibration (upset O/E)", "avg_upset_oe"),
        ("High-confidence accuracy", "avg_acc_conf"),
    ]

    for label, key in key_metrics:
        value = results.get(key, np.nan)
        if np.isfinite(value):
            summary += f"  • {label}: {value:.3f}\n"

    return summary.strip()


def format_tournament_metrics(
    per_tournament_metrics: List[Dict[str, Any]], top_n: int = 5
) -> str:
    """
    Format per-tournament metrics showing best and worst performing tournaments.

    Parameters
    ----------
    per_tournament_metrics : List[Dict[str, Any]]
        Per-tournament metrics from evaluation results
    top_n : int
        Number of top/bottom tournaments to show

    Returns
    -------
    str
        Formatted tournament breakdown
    """
    if not per_tournament_metrics:
        return "No per-tournament data available."

    # Sort by loss (lower is better)
    tournaments = sorted(
        per_tournament_metrics, key=lambda x: x.get("weighted_loss", np.inf)
    )

    def format_tournament_row(t: Dict[str, Any]) -> str:
        tid = t.get("tournament_id", "Unknown")
        loss = t.get("weighted_loss", np.nan)
        c_stat = t.get("c_stat", np.nan)
        n_matches = t.get("n_matches", 0)

        return f"  {tid:>10} │ {loss:7.3f} │ {c_stat:6.3f} │ {n_matches:>8}"

    result = f"""
Tournament Performance Breakdown
{'=' * 35}

Best Performing Tournaments:
    Tournament │    Loss │ C-stat │ Matches
    ───────────┼─────────┼────────┼─────────
"""

    for t in tournaments[:top_n]:
        result += format_tournament_row(t) + "\n"

    if len(tournaments) > top_n:
        result += f"\nWorst Performing Tournaments:\n"
        result += "    Tournament │    Loss │ C-stat │ Matches\n"
        result += "    ───────────┼─────────┼────────┼─────────\n"

        for t in tournaments[-top_n:]:
            result += format_tournament_row(t) + "\n"

    return result.strip()


def _format_dataframe_table(df: pl.DataFrame) -> str:
    """
    Format a polars DataFrame as a box-drawing table.

    This is a simple implementation - for production use,
    consider using tabulate or rich libraries.
    """
    if df.height == 0:
        return "Empty table"

    # Convert to list of dicts for easier processing
    rows = df.to_dicts()
    columns = df.columns

    # Calculate column widths
    widths = {}
    for col in columns:
        widths[col] = max(len(col), max(len(str(row[col])) for row in rows))

    # Format header
    header_sep = (
        "┌" + "┬".join("─" * (widths[col] + 2) for col in columns) + "┐"
    )
    header_row = (
        "│" + "│".join(f" {col:^{widths[col]}} " for col in columns) + "│"
    )
    separator = "├" + "┼".join("─" * (widths[col] + 2) for col in columns) + "┤"

    # Format data rows
    data_rows = []
    for row in rows:
        data_row = (
            "│"
            + "│".join(f" {str(row[col]):>{widths[col]}} " for col in columns)
            + "│"
        )
        data_rows.append(data_row)

    # Format footer
    footer = "└" + "┴".join("─" * (widths[col] + 2) for col in columns) + "┘"

    # Combine all parts
    return "\n".join([header_sep, header_row, separator, *data_rows, footer])


def format_calibration_table(results: Dict[str, Any]) -> str:
    """
    Format reliability diagram data as a calibration table.

    Parameters
    ----------
    results : Dict[str, Any]
        Results from cross_validate_ratings() with predictions

    Returns
    -------
    str
        Formatted calibration table
    """
    # Collect all predictions from all tournaments across all splits
    all_predictions = []

    split_results = results.get("split_results", [])
    for split in split_results:
        per_tournament = split.get("per_tournament_metrics", [])
        for tournament in per_tournament:
            predictions = tournament.get("predictions", np.array([]))
            if len(predictions) > 0:
                all_predictions.extend(predictions)

    if not all_predictions:
        return "No prediction data available for calibration analysis."

    # Create reliability diagram
    predictions_array = np.array(all_predictions)
    diagram_df = reliability_diagram(predictions_array, n_buckets=10)

    if diagram_df.height == 0:
        return "No calibration data available."

    # Filter out empty buckets and format
    diagram_df = diagram_df.filter(pl.col("count") > 0)

    # Format for display
    display_df = diagram_df.select(
        [
            pl.col("bucket_center").round(2).alias("Bucket"),
            pl.col("count").alias("Count"),
            pl.col("expected_rate").round(3).alias("Expected"),
            pl.col("observed_rate").round(3).alias("Observed"),
            pl.col("calibration_error").round(3).alias("Cal Error"),
        ]
    )

    result = "Weighted Log-Loss (Entropy) - Calibration Analysis\n"
    result += "=" * 52 + "\n\n"
    result += _format_dataframe_table(display_df)
    result += f"\n\nTotal predictions: {len(all_predictions)}"
    result += f"\nMean calibration error: {diagram_df['calibration_error'].mean():.3f}"

    return result


def print_evaluation_dashboard(
    results: Dict[str, Any], calibration: bool = False
) -> None:
    """
    Print a comprehensive dashboard of evaluation results.

    Parameters
    ----------
    results : Dict[str, Any]
        Results from cross_validate_ratings()
    calibration : bool, default=False
        If True, include calibration analysis table

    Examples
    --------
    >>> results = {'avg_loss': 0.693, 'n_splits': 5}
    >>> print_evaluation_dashboard(results)  # doctest: +SKIP
    """
    print(format_split_summary(results))
    print("\n")
    print(format_metrics_table(results))

    # Show calibration analysis if requested
    if calibration:
        print("\n")
        print(format_calibration_table(results))

    # Show per-tournament breakdown if available
    per_tournament = results.get("split_results", [])
    if per_tournament:
        # Flatten per-tournament metrics from all splits
        all_tournaments = []
        for split in per_tournament:
            tournaments = split.get("per_tournament_metrics", [])
            all_tournaments.extend(tournaments)

        if all_tournaments:
            print("\n")
            print(format_tournament_metrics(all_tournaments))


def format_tournament_prediction_table(
    results: Dict[str, Any],
    metrics: Optional[List[Tuple[str, str]]] = None,
    precision: int = 4,
) -> str:
    """
    Format tournament prediction results as a pretty table.

    Parameters
    ----------
    results : Dict[str, Any]
        Results from tournament prediction evaluation
    metrics : Optional[List[Tuple[str, str]]]
        List of (display_name, key) tuples for metrics to show
    precision : int
        Number of decimal places

    Returns
    -------
    str
        Formatted table string
    """
    if metrics is None:
        metrics = [
            ("NDCG@4", "ndcg_at_4"),
            ("NDCG@8", "ndcg_at_8"),
            ("NDCG (Full)", "ndcg_full"),
            ("Spearman Corr", "spearman_correlation"),
            ("MAE (All)", "mae_all"),
            ("MAE (Top 4)", "mae_top4"),
            ("Match Accuracy", "match_accuracy"),
            ("Match Log Loss", "match_log_loss"),
            ("Calibration Error", "expected_calibration_error"),
            ("Manual Agreement", "manual_pairwise_agreement"),
        ]

    # Build table data
    rows = []
    metrics_dict = results.get("metrics", {})

    for display_name, key in metrics:
        if key in metrics_dict:
            stats = metrics_dict[key]
            row = {
                "Metric": display_name,
                "Mean": f"{stats.get('mean', 0):.{precision}f}",
                "Std": f"{stats.get('std', 0):.{precision}f}",
                "Median": f"{stats.get('median', 0):.{precision}f}",
            }

            # Add confidence interval if available
            if "ci_95" in stats:
                ci_lower, ci_upper = stats["ci_95"]
                row[
                    "95% CI"
                ] = f"[{ci_lower:.{precision}f}, {ci_upper:.{precision}f}]"
            else:
                row["95% CI"] = "--"

            rows.append(row)

    if not rows:
        return "No tournament prediction metrics available"

    # Create DataFrame for pretty printing
    df = pl.DataFrame(rows)

    # Format as string table
    table_lines = []

    # Header
    table_lines.append(
        "┌"
        + "─" * 18
        + "┬"
        + "─" * 10
        + "┬"
        + "─" * 10
        + "┬"
        + "─" * 10
        + "┬"
        + "─" * 20
        + "┐"
    )
    table_lines.append(
        f"│ {'Metric':<16} │ {'Mean':^8} │ {'Std':^8} │ "
        f"{'Median':^8} │ {'95% CI':^18} │"
    )
    table_lines.append(
        "├"
        + "─" * 18
        + "┼"
        + "─" * 10
        + "┼"
        + "─" * 10
        + "┼"
        + "─" * 10
        + "┼"
        + "─" * 20
        + "┤"
    )

    # Data rows
    for row in rows:
        table_lines.append(
            f"│ {row['Metric']:<16} │ {row['Mean']:>8} │ {row['Std']:>8} │ "
            f"{row['Median']:>8} │ {row['95% CI']:>18} │"
        )

    # Footer
    table_lines.append(
        "└"
        + "─" * 18
        + "┴"
        + "─" * 10
        + "┴"
        + "─" * 10
        + "┴"
        + "─" * 10
        + "┴"
        + "─" * 20
        + "┘"
    )

    return "\n".join(table_lines)


def format_seeding_comparison(
    model_seeds: List[Tuple[int, float, float]],
    manual_seeds: Optional[Dict[int, int]] = None,
    final_placements: Optional[Dict[int, int]] = None,
    top_k: int = 10,
) -> str:
    """
    Format seeding comparison table.

    Parameters
    ----------
    model_seeds : List[Tuple[int, float, float]]
        List of (team_id, rating, confidence) from model
    manual_seeds : Optional[Dict[int, int]]
        Manual seeds for comparison
    final_placements : Optional[Dict[int, int]]
        Actual final placements
    top_k : int
        Number of top seeds to show

    Returns
    -------
    str
        Formatted comparison table
    """
    rows = []

    for i, (team_id, rating, confidence) in enumerate(model_seeds[:top_k]):
        seed = i + 1
        row = {
            "Seed": seed,
            "Team ID": team_id,
            "Rating": f"{rating:.3f}",
            "Confidence": f"{confidence:.2%}",
        }

        if manual_seeds and team_id in manual_seeds:
            row["Manual Seed"] = manual_seeds[team_id]
            row["Diff"] = seed - manual_seeds[team_id]
        else:
            row["Manual Seed"] = "--"
            row["Diff"] = "--"

        if final_placements and team_id in final_placements:
            row["Final"] = final_placements[team_id]
            row["Error"] = abs(seed - final_placements[team_id])
        else:
            row["Final"] = "--"
            row["Error"] = "--"

        rows.append(row)

    if not rows:
        return "No seeding data available"

    # Format as table
    table_lines = []
    table_lines.append("Tournament Seeding Comparison")
    table_lines.append("=" * 70)
    table_lines.append(
        f"{'Seed':<6} {'Team':<8} {'Rating':<10} {'Conf':<8} "
        f"{'Manual':<8} {'Diff':<6} {'Final':<7} {'Error':<6}"
    )
    table_lines.append("-" * 70)

    for row in rows:
        table_lines.append(
            f"{row['Seed']:<6} {row['Team ID']:<8} {row['Rating']:<10} "
            f"{row['Confidence']:<8} {row['Manual Seed']:<8} "
            f"{row['Diff']:<6} {row['Final']:<7} {row['Error']:<6}"
        )

    return "\n".join(table_lines)


def format_upset_analysis(
    upset_rates: Dict[str, Dict[str, float]],
    precision: int = 3,
) -> str:
    """
    Format upset rate analysis table.

    Parameters
    ----------
    upset_rates : Dict[str, Dict[str, float]]
        Upset analysis by probability bucket
    precision : int
        Number of decimal places

    Returns
    -------
    str
        Formatted upset analysis table
    """
    if not upset_rates:
        return "No upset rate data available"

    table_lines = []
    table_lines.append("Upset Rate Analysis by Probability")
    table_lines.append("=" * 60)
    table_lines.append(
        f"{'Prob Range':<12} {'N':<6} {'Expected':<10} "
        f"{'Actual':<10} {'Diff':<10} {'Upset %':<10}"
    )
    table_lines.append("-" * 60)

    for bucket_name, stats in sorted(upset_rates.items()):
        n_matches = stats.get("n_matches", 0)
        expected = stats.get("expected_win_rate", 0)
        actual = stats.get("actual_win_rate", 0)
        diff = stats.get("calibration_diff", 0)
        upset_rate = stats.get("upset_rate", 0)

        table_lines.append(
            f"{bucket_name:<12} {n_matches:<6} "
            f"{expected:.{precision}f}".ljust(10)
            + " "
            + f"{actual:.{precision}f}".ljust(10)
            + " "
            + f"{diff:+.{precision}f}".ljust(10)
            + " "
            + f"{upset_rate:.1%}".ljust(10)
        )

    return "\n".join(table_lines)


def print_tournament_prediction_dashboard(
    results: Dict[str, Any],
    show_seeding: bool = True,
    show_upsets: bool = True,
    show_comparison: bool = True,
) -> None:
    """
    Print comprehensive tournament prediction dashboard.

    Parameters
    ----------
    results : Dict[str, Any]
        Tournament prediction evaluation results
    show_seeding : bool
        Show seeding comparison table
    show_upsets : bool
        Show upset analysis
    show_comparison : bool
        Show model comparison if available
    """
    print("\n" + "=" * 70)
    print("TOURNAMENT PREDICTION EVALUATION DASHBOARD")
    print("=" * 70)

    # Overall metrics
    print("\nOverall Performance Metrics:")
    print("-" * 70)
    print(format_tournament_prediction_table(results))

    # Seeding comparison
    if show_seeding and "example_seeds" in results:
        print("\n\nExample Seeding Comparison (Latest Tournament):")
        print("-" * 70)
        print(
            format_seeding_comparison(
                results["example_seeds"],
                results.get("example_manual_seeds"),
                results.get("example_placements"),
            )
        )

    # Upset analysis
    if show_upsets and "upset_analysis" in results:
        print("\n\nCalibration and Upset Analysis:")
        print("-" * 70)
        print(format_upset_analysis(results["upset_analysis"]))

    # Model comparison
    if show_comparison and "model_comparison" in results:
        print("\n\nModel Comparison vs Manual Seeding:")
        print("-" * 70)
        comparison = results["model_comparison"]

        print(
            f"Pairwise Agreement: {comparison.get('pairwise_agreement', 0):.1%}"
        )
        print(
            f"Better than Manual: {comparison.get('better_than_manual_pct', 0):.1%}"
        )

        if "mcnemar_p_value" in comparison:
            p_val = comparison["mcnemar_p_value"]
            sig = (
                "***"
                if p_val < 0.001
                else "**"
                if p_val < 0.01
                else "*"
                if p_val < 0.05
                else ""
            )
            print(f"McNemar Test p-value: {p_val:.4f} {sig}")

    # Work saved metrics
    if "work_saved" in results:
        print("\n\nPotential Work Saved for Organizers:")
        print("-" * 70)
        work = results["work_saved"]
        print(f"Automated Decisions: {work.get('automated_pct', 0):.1%}")
        print(f"Borderline Teams: {work.get('borderline_teams', 0):.0f}")
        print(f"Est. Time Saved: {work.get('time_saved_hours', 0):.1f} hours")

    print("\n" + "=" * 70)
