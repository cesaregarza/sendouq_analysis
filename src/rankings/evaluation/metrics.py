"""
Diagnostic metrics for tournament rating evaluation.

This module implements additional metrics beyond the main loss function
to help diagnose rating system performance.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from scipy.stats import spearmanr


def compute_brier_score(predictions: np.ndarray, outcomes: np.ndarray) -> float:
    """
    Compute Brier score for binary predictions.

    The Brier score measures the mean squared difference between
    predicted probabilities and actual outcomes.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted probabilities (0-1)
    outcomes : np.ndarray
        Actual outcomes (0 or 1)

    Returns
    -------
    float
        Brier score (lower is better, 0 is perfect)
    """
    return np.mean((predictions - outcomes) ** 2)


def compute_accuracy(
    predictions: np.ndarray, outcomes: np.ndarray, threshold: float = 0.5
) -> float:
    """
    Compute prediction accuracy.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted probabilities
    outcomes : np.ndarray
        Actual outcomes (0 or 1)
    threshold : float
        Decision threshold

    Returns
    -------
    float
        Accuracy (0-1)
    """
    predicted_classes = (predictions >= threshold).astype(int)
    return np.mean(predicted_classes == outcomes)


def compute_spearman_correlation(
    pre_tournament_ratings: Dict[int, float], final_placements: Dict[int, int]
) -> Tuple[float, float]:
    """
    Compute Spearman correlation between ratings and tournament placements.

    Parameters
    ----------
    pre_tournament_ratings : Dict[int, float]
        Ratings before tournament (team_id -> rating)
    final_placements : Dict[int, int]
        Final tournament placements (team_id -> placement)
        Lower placement = better (1st place = 1)

    Returns
    -------
    Tuple[float, float]
        (correlation, p-value)
    """
    # Get common teams
    common_teams = set(pre_tournament_ratings.keys()) & set(
        final_placements.keys()
    )

    if len(common_teams) < 3:
        return 0.0, 1.0

    # Extract paired values
    ratings = []
    placements = []

    for team_id in common_teams:
        ratings.append(pre_tournament_ratings[team_id])
        placements.append(final_placements[team_id])

    # Higher rating should correspond to lower (better) placement
    # So we expect negative correlation
    corr, p_value = spearmanr(ratings, placements)

    # Return absolute correlation for easier interpretation
    return -corr, p_value


def compute_round_metrics(
    matches_df: pl.DataFrame,
    predictions: Dict[int, float],
    round_col: str = "round_name",
) -> pl.DataFrame:
    """
    Compute metrics broken down by tournament round.

    Parameters
    ----------
    matches_df : pl.DataFrame
        Matches with predictions
    predictions : Dict[int, float]
        Match ID to predicted probability
    round_col : str
        Column containing round information

    Returns
    -------
    pl.DataFrame
        Metrics by round
    """
    # Add predictions to matches
    match_ids = (
        matches_df["match_id"].to_list()
        if "match_id" in matches_df.columns
        else list(range(len(matches_df)))
    )
    pred_values = [predictions.get(mid, 0.5) for mid in match_ids]

    matches_with_pred = matches_df.with_columns(
        [
            pl.Series("prediction", pred_values),
            pl.lit(1).alias("actual"),  # Winner always won
        ]
    )

    # Compute metrics by round
    round_metrics = (
        matches_with_pred.group_by(round_col)
        .agg(
            [
                pl.count().alias("n_matches"),
                pl.col("prediction").mean().alias("mean_prediction"),
                (pl.col("prediction") > 0.5).mean().alias("accuracy"),
                ((pl.col("prediction") - pl.col("actual")) ** 2)
                .mean()
                .alias("brier_score"),
                (
                    -pl.col("actual") * pl.col("prediction").log()
                    - (1 - pl.col("actual")) * (1 - pl.col("prediction")).log()
                )
                .mean()
                .alias("log_loss"),
            ]
        )
        .sort(round_col)
    )

    return round_metrics


def evaluate_tournament_predictions(
    tournament_matches: pl.DataFrame,
    rating_map: Dict[int, float],
    tournament_placements: Optional[Dict[int, int]] = None,
    alpha: float = 1.0,
) -> Dict[str, any]:
    """
    Comprehensive evaluation of predictions for a single tournament.

    Parameters
    ----------
    tournament_matches : pl.DataFrame
        Matches from the tournament
    rating_map : Dict[int, float]
        Pre-tournament ratings
    tournament_placements : Optional[Dict[int, int]]
        Final placements (if available)
    alpha : float
        Temperature parameter

    Returns
    -------
    Dict[str, any]
        Comprehensive metrics
    """
    from .loss import compute_match_probability

    # Compute predictions for each match
    predictions = []
    outcomes = []
    match_details = []

    for row in tournament_matches.iter_rows(named=True):
        winner_id = row["winner_team_id"]
        loser_id = row["loser_team_id"]

        r_winner = rating_map.get(winner_id, 0.0)
        r_loser = rating_map.get(loser_id, 0.0)

        # Probability that winner beats loser
        p = compute_match_probability(r_winner, r_loser, alpha)

        predictions.append(p)
        outcomes.append(1)  # Winner always won

        match_details.append(
            {
                "winner_id": winner_id,
                "loser_id": loser_id,
                "prediction": p,
                "winner_rating": r_winner,
                "loser_rating": r_loser,
                "upset": p < 0.5,  # Winner was predicted to lose
            }
        )

    predictions = np.array(predictions)
    outcomes = np.array(outcomes)

    # Compute metrics
    metrics = {
        "n_matches": len(predictions),
        "accuracy": compute_accuracy(predictions, outcomes),
        "brier_score": compute_brier_score(predictions, outcomes),
        "mean_prediction": np.mean(predictions),
        "upset_rate": np.mean(predictions < 0.5),
        "prediction_std": np.std(predictions),
    }

    # Add placement correlation if available
    if tournament_placements:
        corr, p_value = compute_spearman_correlation(
            rating_map, tournament_placements
        )
        metrics["placement_correlation"] = corr
        metrics["placement_p_value"] = p_value

    # Round-specific metrics if round info available
    if "round_name" in tournament_matches.columns:
        round_metrics = compute_round_metrics(
            tournament_matches, {i: p for i, p in enumerate(predictions)}
        )
        metrics["round_metrics"] = round_metrics

    # Match details for further analysis
    metrics["match_details"] = match_details

    return metrics


def compute_accuracy_at_threshold(
    predictions: np.ndarray, outcomes: np.ndarray, threshold: float = 0.60
) -> float:
    """
    Compute accuracy for predictions above a confidence threshold.

    As mentioned in plan.md, this metric ignores true coin-flips and focuses
    on matches where the model has reasonable confidence.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted probabilities
    outcomes : np.ndarray
        Actual outcomes (0 or 1)
    threshold : float
        Minimum prediction confidence (default 0.60)

    Returns
    -------
    float
        Accuracy for high-confidence predictions only
    """
    confident_mask = predictions >= threshold
    if not np.any(confident_mask):
        return np.nan

    confident_predictions = predictions[confident_mask]
    confident_outcomes = outcomes[confident_mask]

    return np.mean(confident_outcomes)


def compute_expected_upset_rate(
    predictions: np.ndarray, outcomes: np.ndarray
) -> Tuple[float, float]:
    """
    Compare observed upsets vs expected given predicted probabilities.

    From plan.md: highlights systematic under-/over-confidence in the model.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted win probabilities for the observed winner
    outcomes : np.ndarray
        Actual outcomes (should be all 1s for winners)

    Returns
    -------
    Tuple[float, float]
        (expected_upset_rate, observed_upset_rate)
    """
    # Expected upset rate is mean of (1 - p) for all matches
    expected_upsets = np.mean(1 - predictions)

    # Observed upset rate is fraction where prediction < 0.5
    observed_upsets = np.mean(predictions < 0.5)

    return expected_upsets, observed_upsets


def evaluate_by_rating_separation(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    rating_diffs: np.ndarray,
    min_gap: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate predictions only on matches with meaningful rating separation.

    From plan.md: skip or down-weight matches where rating gap is too small,
    focusing on matches where predictions should be meaningful.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted probabilities
    outcomes : np.ndarray
        Actual outcomes
    rating_diffs : np.ndarray
        Absolute rating differences between opponents
    min_gap : float
        Minimum rating gap to include (in units of rating std dev)

    Returns
    -------
    Dict[str, float]
        Metrics for well-separated matches only
    """
    separated_mask = rating_diffs >= min_gap

    if not np.any(separated_mask):
        return {"n_matches": 0}

    sep_predictions = predictions[separated_mask]
    sep_outcomes = outcomes[separated_mask]

    return {
        "n_matches": np.sum(separated_mask),
        "accuracy": compute_accuracy(sep_predictions, sep_outcomes),
        "brier_score": compute_brier_score(sep_predictions, sep_outcomes),
        "mean_prediction": np.mean(sep_predictions),
        "fraction_of_total": np.mean(separated_mask),
    }


def aggregate_tournament_metrics(
    tournament_metrics: List[Dict[str, any]]
) -> Dict[str, any]:
    """
    Aggregate metrics across multiple tournaments.

    Parameters
    ----------
    tournament_metrics : List[Dict[str, any]]
        List of metric dictionaries from evaluate_tournament_predictions

    Returns
    -------
    Dict[str, any]
        Aggregated metrics
    """
    if not tournament_metrics:
        return {}

    # Extract scalar metrics
    scalar_keys = [
        "accuracy",
        "brier_score",
        "mean_prediction",
        "upset_rate",
        "prediction_std",
        "placement_correlation",
    ]

    aggregated = {}

    for key in scalar_keys:
        values = [m[key] for m in tournament_metrics if key in m]
        if values:
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)
            aggregated[f"{key}_min"] = np.min(values)
            aggregated[f"{key}_max"] = np.max(values)

    # Count tournaments
    aggregated["n_tournaments"] = len(tournament_metrics)
    aggregated["n_total_matches"] = sum(
        m["n_matches"] for m in tournament_metrics
    )

    # Aggregate round metrics if available
    all_round_metrics = []
    for m in tournament_metrics:
        if "round_metrics" in m and isinstance(
            m["round_metrics"], pl.DataFrame
        ):
            all_round_metrics.append(m["round_metrics"])

    if all_round_metrics:
        combined_rounds = pl.concat(all_round_metrics)
        aggregated["combined_round_metrics"] = (
            combined_rounds.group_by("round_name")
            .agg(
                [
                    pl.col("n_matches").sum(),
                    pl.col("accuracy").mean(),
                    pl.col("brier_score").mean(),
                    pl.col("log_loss").mean(),
                ]
            )
            .sort("round_name")
        )

    return aggregated
