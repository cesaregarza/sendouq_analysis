"""
Cross-validation for tournament prediction and seeding.

This module extends the cross-validation framework to evaluate
tournament prediction capabilities using proper temporal validation.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

from rankings.evaluation.tournament_metrics import tournament_prediction_summary
from rankings.tournament_prediction import (
    MatchPredictor,
    MonteCarloSimulator,
    TeamRatingConfig,
    TeamStrengthCalculator,
    TournamentSeeder,
)


@dataclass
class TournamentPredictionSplit:
    """Data split for tournament prediction evaluation."""

    train_tournaments: list[int]  # Tournament IDs for training
    test_tournament: int  # Single tournament ID for testing
    train_start_date: int  # Unix timestamp
    train_end_date: int  # Unix timestamp
    test_date: int  # Unix timestamp


def create_tournament_prediction_splits(
    tournaments_df: pl.DataFrame,
    min_train_tournaments: int = 10,
    gap_weeks: int = 0,
) -> list[TournamentPredictionSplit]:
    """
    Create time-based splits for tournament prediction evaluation.

    Each split uses all tournaments before a target tournament for training,
    then evaluates on that single tournament.

    Parameters
    ----------
    tournaments_df : pl.DataFrame
        DataFrame with tournament metadata including start_time
    min_train_tournaments : int
        Minimum number of training tournaments required
    gap_weeks : int
        Gap in weeks between training and test data

    Returns
    -------
    list[TournamentPredictionSplit]
        List of train/test splits
    """
    # Sort tournaments by start time
    tournaments = tournaments_df.sort("start_time")

    splits = []
    tournament_list = tournaments.to_dicts()

    for i in range(min_train_tournaments, len(tournament_list)):
        test_tournament = tournament_list[i]
        test_id = test_tournament["tournament_id"]
        test_date = test_tournament["start_time"]

        # Calculate cutoff date with gap
        cutoff_date = test_date
        if gap_weeks > 0:
            from datetime import datetime, timedelta

            cutoff_date = int(
                (
                    datetime.fromtimestamp(test_date)
                    - timedelta(weeks=gap_weeks)
                ).timestamp()
            )

        # Get training tournaments
        train_tournaments = [
            t["tournament_id"]
            for t in tournament_list[:i]
            if t["start_time"] < cutoff_date
        ]

        if len(train_tournaments) >= min_train_tournaments:
            split = TournamentPredictionSplit(
                train_tournaments=train_tournaments,
                test_tournament=test_id,
                train_start_date=tournament_list[0]["start_time"],
                train_end_date=cutoff_date,
                test_date=test_date,
            )
            splits.append(split)

    return splits


def evaluate_tournament_prediction(
    split: TournamentPredictionSplit,
    player_ratings: dict[int, float],
    matches_df: pl.DataFrame,
    teams_df: pl.DataFrame,
    config: TeamRatingConfig | None = None,
) -> dict[str, Any]:
    """
    Evaluate tournament prediction on a single split.

    Parameters
    ----------
    split : TournamentPredictionSplit
        Train/test split specification
    player_ratings : dict[int, float]
        Player ratings trained on training data
    matches_df : pl.DataFrame
        Match data for the test tournament
    teams_df : pl.DataFrame
        Team roster data for the test tournament
    config : TeamRatingConfig | None
        Configuration for team strength calculation

    Returns
    -------
    dict[str, Any]
        Evaluation metrics for this split
    """
    # Initialize components
    team_calculator = TeamStrengthCalculator(config)
    match_predictor = MatchPredictor()
    seeder = TournamentSeeder(team_calculator, match_predictor)

    # Get test tournament data
    test_matches = matches_df.filter(
        pl.col("tournament_id") == split.test_tournament
    )
    test_teams = teams_df.filter(
        pl.col("tournament_id") == split.test_tournament
    )

    # Build team rosters
    team_rosters = {}
    for team in test_teams.to_dicts():
        team_id = team["team_id"]
        # Get player IDs for this team
        player_ids = team.get("player_ids", [])
        if player_ids:
            team_rosters[team_id] = player_ids

    if not team_rosters:
        return {"error": "No team rosters found"}

    # Generate seeds
    seeds = seeder.generate_seeds(player_ratings, team_rosters)
    predicted_seeds = {
        team_id: i + 1 for i, (team_id, _, _) in enumerate(seeds)
    }

    # Get actual placements (if available)
    final_placements = {}
    if "final_placement" in test_teams.columns:
        for team in test_teams.to_dicts():
            if team.get("final_placement"):
                final_placements[team["team_id"]] = team["final_placement"]

    # Calibrate match predictor on training matches
    if len(split.train_tournaments) > 0:
        train_matches = matches_df.filter(
            pl.col("tournament_id").is_in(split.train_tournaments)
        )

        # Prepare calibration data
        calibration_data = []
        for match in train_matches.to_dicts():
            team_a = match.get("winner_team_id")
            team_b = match.get("loser_team_id")

            if team_a and team_b:
                # Get team ratings
                roster_a = team_rosters.get(team_a, [])
                roster_b = team_rosters.get(team_b, [])

                if roster_a and roster_b:
                    rating_a = team_calculator.team_log_rating(
                        player_ratings, roster_a
                    )
                    rating_b = team_calculator.team_log_rating(
                        player_ratings, roster_b
                    )

                    # Add both perspectives
                    calibration_data.append((rating_a, rating_b, True))
                    calibration_data.append((rating_b, rating_a, False))

        if calibration_data:
            calibration_result = match_predictor.calibrate(calibration_data)

    # Evaluate match predictions
    match_predictions = []
    match_outcomes = []

    for match in test_matches.to_dicts():
        team_a = match.get("winner_team_id")
        team_b = match.get("loser_team_id")

        if (
            team_a
            and team_b
            and team_a in team_rosters
            and team_b in team_rosters
        ):
            roster_a = team_rosters[team_a]
            roster_b = team_rosters[team_b]

            rating_a = team_calculator.team_log_rating(player_ratings, roster_a)
            rating_b = team_calculator.team_log_rating(player_ratings, roster_b)

            # Predict match outcome
            prob_a_wins = match_predictor.win_probability(rating_a, rating_b)

            match_predictions.append(prob_a_wins)
            match_outcomes.append(True)  # Team A (winner) won

            # Also add reverse perspective
            match_predictions.append(1 - prob_a_wins)
            match_outcomes.append(False)  # Team B (loser) lost

    # Calculate metrics
    metrics = tournament_prediction_summary(
        predicted_seeds=predicted_seeds,
        final_placements=final_placements,
        match_predictions=match_predictions if match_predictions else None,
        match_outcomes=match_outcomes if match_outcomes else None,
    )

    # Add split metadata
    metrics["tournament_id"] = split.test_tournament
    metrics["n_train_tournaments"] = len(split.train_tournaments)
    metrics["n_teams"] = len(team_rosters)
    metrics["n_matches"] = len(test_matches)

    return metrics


def cross_validate_tournament_predictions(
    tournaments_df: pl.DataFrame,
    matches_df: pl.DataFrame,
    teams_df: pl.DataFrame,
    player_ratings_by_date: dict[int, dict[int, float]],
    n_splits: int | None = None,
    config: TeamRatingConfig | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run cross-validation for tournament predictions.

    Parameters
    ----------
    tournaments_df : pl.DataFrame
        Tournament metadata
    matches_df : pl.DataFrame
        Match data
    teams_df : pl.DataFrame
        Team roster data
    player_ratings_by_date : dict[int, dict[int, float]
        Player ratings keyed by tournament date
    n_splits : int | None
        Number of splits to evaluate (None for all)
    config : TeamRatingConfig | None
        Team rating configuration
    verbose : bool
        Print progress

    Returns
    -------
    dict[str, Any]
        Aggregated cross-validation results
    """
    # Create splits
    splits = create_tournament_prediction_splits(tournaments_df)

    if n_splits is not None:
        splits = splits[:n_splits]

    if verbose:
        logger.info(f"Evaluating {len(splits)} tournament prediction splits...")

    # Evaluate each split
    split_results = []
    for i, split in enumerate(splits):
        if verbose and i % 10 == 0:
            logger.info(f"  Split {i+1}/{len(splits)}...")

        # Get player ratings for this split
        # Use ratings from just before the test tournament
        ratings = player_ratings_by_date.get(split.test_tournament, {})

        if not ratings:
            continue

        # Evaluate
        metrics = evaluate_tournament_prediction(
            split, ratings, matches_df, teams_df, config
        )

        split_results.append(metrics)

    # Aggregate results
    aggregated = aggregate_cv_results(split_results)

    if verbose:
        logger.info(f"Completed evaluation of {len(split_results)} tournaments")
        print_cv_summary(aggregated)

    return aggregated


def aggregate_cv_results(split_results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate cross-validation results across splits.

    Parameters
    ----------
    split_results : list[dict[str, Any]
        Results from each split

    Returns
    -------
    dict[str, Any]
        Aggregated metrics with mean, std, and confidence intervals
    """
    if not split_results:
        return {}

    # Collect all metrics
    metric_values = {}
    for result in split_results:
        for key, value in result.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                if key not in metric_values:
                    metric_values[key] = []
                metric_values[key].append(value)

    # Calculate statistics
    aggregated = {"n_tournaments": len(split_results), "metrics": {}}

    for metric, values in metric_values.items():
        if values:
            aggregated["metrics"][metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "median": np.median(values),
                "min": np.min(values),
                "max": np.max(values),
                "q25": np.percentile(values, 25),
                "q75": np.percentile(values, 75),
                "values": values,  # Keep raw values for further analysis
            }

    # Add confidence intervals (95%)
    for metric in aggregated["metrics"]:
        values = aggregated["metrics"][metric]["values"]
        n = len(values)
        if n > 1:
            se = aggregated["metrics"][metric]["std"] / np.sqrt(n)
            ci_lower = aggregated["metrics"][metric]["mean"] - 1.96 * se
            ci_upper = aggregated["metrics"][metric]["mean"] + 1.96 * se
            aggregated["metrics"][metric]["ci_95"] = (ci_lower, ci_upper)

    return aggregated


def print_cv_summary(results: dict[str, Any]) -> None:
    """
    Print a summary of cross-validation results.

    Parameters
    ----------
    results : dict[str, Any]
        Aggregated CV results
    """
    print("\n" + "=" * 60)
    print("Tournament Prediction Cross-Validation Summary")
    print("=" * 60)

    print(f"\nEvaluated on {results.get('n_tournaments', 0)} tournaments")

    if "metrics" in results:
        # Key metrics to highlight
        key_metrics = [
            ("ndcg_at_4", "NDCG@4 (Seeding)", True),
            ("spearman_correlation", "Spearman Correlation", False),
            ("mae_top4", "MAE Top 4", False),
            ("match_accuracy", "Match Accuracy", True),
            ("match_log_loss", "Match Log Loss", False),
            ("expected_calibration_error", "Calibration Error", False),
        ]

        print("\nKey Metrics:")
        print("-" * 40)

        for metric_key, display_name, higher_better in key_metrics:
            if metric_key in results["metrics"]:
                stats = results["metrics"][metric_key]
                arrow = "↑" if higher_better else "↓"
                print(
                    f"{display_name:25} {arrow} {stats['mean']:.3f} ± {stats['std']:.3f}"
                )
                if "ci_95" in stats:
                    print(
                        f"{'':25}   95% CI: [{stats['ci_95'][0]:.3f}, {stats['ci_95'][1]:.3f}]"
                    )

    print("=" * 60)


def compare_models_cv(
    model_results: dict[str, dict[str, Any]],
    baseline_name: str = "manual",
) -> dict[str, Any]:
    """
    Compare multiple models using cross-validation results.

    Parameters
    ----------
    model_results : dict[str, dict[str, Any]]
        Results for each model (model_name -> cv_results)
    baseline_name : str
        Name of baseline model for comparison

    Returns
    -------
    dict[str, Any]
        Comparison statistics
    """
    comparison = {}

    if baseline_name not in model_results:
        logger.warning(f"Baseline model '{baseline_name}' not found")
        return comparison

    baseline = model_results[baseline_name]

    for model_name, results in model_results.items():
        if model_name == baseline_name:
            continue

        model_comparison = {
            "model": model_name,
            "vs_baseline": baseline_name,
            "improvements": {},
        }

        # Compare each metric
        for metric in results.get("metrics", {}):
            if metric in baseline.get("metrics", {}):
                model_mean = results["metrics"][metric]["mean"]
                baseline_mean = baseline["metrics"][metric]["mean"]

                improvement = model_mean - baseline_mean
                pct_improvement = (
                    (improvement / abs(baseline_mean)) * 100
                    if baseline_mean != 0
                    else 0
                )

                # Determine if improvement is significant
                # Simple t-test approximation
                model_values = results["metrics"][metric]["values"]
                baseline_values = baseline["metrics"][metric]["values"]

                if len(model_values) > 1 and len(baseline_values) > 1:
                    from scipy import stats

                    t_stat, p_value = stats.ttest_ind(
                        model_values, baseline_values
                    )

                    model_comparison["improvements"][metric] = {
                        "absolute": improvement,
                        "percentage": pct_improvement,
                        "significant": p_value < 0.05,
                        "p_value": p_value,
                    }

        comparison[model_name] = model_comparison

    return comparison
