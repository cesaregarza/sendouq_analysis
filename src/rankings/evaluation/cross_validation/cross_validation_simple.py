"""Simple cross-validation with fast alpha optimization.

This module provides a simplified cross-validation approach that:
- Uses clear temporal splitting logic
- Implements fast alpha optimization via sampling
- Supports confidence-weighted loss functions
- Provides comprehensive evaluation metrics
"""

import logging
from typing import Any, Dict, List, Optional, Type

import numpy as np
import polars as pl
from tqdm import tqdm

from rankings.analysis.utils.summaries import derive_team_ratings_from_players

from ..loss import compute_weighted_log_loss
from ..metrics_extras import concordance
from .simple_splits import create_simple_time_splits

logger = logging.getLogger(__name__)


def _create_teams_from_matches(matches_df: pl.DataFrame) -> pl.DataFrame:
    """Create teams dataframe from matches."""
    # Get unique team IDs from both winner and loser columns
    winner_teams = matches_df.select(
        pl.col("winner_team_id").alias("team_id")
    ).unique()
    loser_teams = matches_df.select(
        pl.col("loser_team_id").alias("team_id")
    ).unique()

    # Combine and get unique teams
    all_teams = pl.concat([winner_teams, loser_teams]).unique()

    return all_teams


def _sample_matches_for_alpha_optimization(
    matches_df: pl.DataFrame,
    sample_size: int = 10000,
    seed: int = 42,
) -> pl.DataFrame:
    """Sample matches for fast alpha optimization.

    Args:
        matches_df: Full matches DataFrame
        sample_size: Number of matches to sample
        seed: Random seed for reproducibility

    Returns:
        Sampled matches DataFrame
    """
    n_matches = matches_df.height

    if n_matches <= sample_size:
        return matches_df

    # Sample uniformly across time to maintain temporal distribution
    return matches_df.sample(n=sample_size, seed=seed).sort("start_time")


def _optimize_alpha_fast(
    matches_df: pl.DataFrame,
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    weight_scheme: str = "entropy_squared",
    sample_size: int = 10000,
    alpha_range: tuple = (0.1, 2.0),
    n_steps: int = 20,
    warm_start: Optional[float] = None,
) -> float:
    """Optimize alpha parameter using sampling for speed.

    Args:
        matches_df: Matches DataFrame
        probabilities: Win probabilities for team 1
        outcomes: Actual outcomes (1 for team 1 win, 0 for team 2 win)
        weight_scheme: Weighting scheme for loss ("none", "entropy", "entropy_squared")
        sample_size: Number of matches to use for optimization
        alpha_range: Range of alpha values to search
        n_steps: Number of steps in grid search
        warm_start: Previous alpha value for warm starting

    Returns:
        Optimal alpha value
    """
    # Sample matches if needed
    n_matches = len(probabilities)
    if n_matches > sample_size:
        sample_indices = np.random.choice(
            n_matches, size=sample_size, replace=False
        )
        sample_probs = probabilities[sample_indices]
        sample_outcomes = outcomes[sample_indices]
    else:
        sample_probs = probabilities
        sample_outcomes = outcomes

    # Calculate weights based on scheme
    if weight_scheme == "entropy":
        # Weight by entropy (uncertainty)
        entropy = -sample_probs * np.log(sample_probs + 1e-10) - (
            1 - sample_probs
        ) * np.log(1 - sample_probs + 1e-10)
        weights = entropy
    elif weight_scheme == "entropy_squared":
        # Weight by squared entropy (more aggressive)
        entropy = -sample_probs * np.log(sample_probs + 1e-10) - (
            1 - sample_probs
        ) * np.log(1 - sample_probs + 1e-10)
        weights = entropy**2
    else:
        # No weighting
        weights = np.ones_like(sample_probs)

    # Normalize weights
    weights = weights / np.mean(weights)

    # Define alpha search range
    if warm_start is not None:
        # Narrow search around warm start
        alpha_min = max(alpha_range[0], warm_start * 0.8)
        alpha_max = min(alpha_range[1], warm_start * 1.2)
    else:
        alpha_min, alpha_max = alpha_range

    alphas = np.linspace(alpha_min, alpha_max, n_steps)

    # Grid search for optimal alpha
    best_alpha = 1.0
    best_loss = float("inf")

    for alpha in alphas:
        # Calculate loss with this alpha
        # Calculate loss with this alpha
        eps = 1e-15
        clipped_probs = np.clip(sample_probs, eps, 1 - eps)

        # Apply alpha scaling
        if alpha != 1.0:
            scaled_probs = 0.5 + (clipped_probs - 0.5) * alpha
            scaled_probs = np.clip(scaled_probs, eps, 1 - eps)
        else:
            scaled_probs = clipped_probs

        # Calculate loss
        base_loss = -np.log(scaled_probs) * sample_outcomes - np.log(
            1 - scaled_probs
        ) * (1 - sample_outcomes)
        weighted_loss = base_loss * weights
        loss = np.mean(weighted_loss)

        if loss < best_loss:
            best_loss = loss
            best_alpha = alpha

    return best_alpha


def cross_validate_simple(
    engine_class: Type,
    engine_params: Dict[str, Any],
    matches_df: pl.DataFrame,
    players_df: pl.DataFrame,
    ranking_entity: str = "player",
    prediction_entity: str = "team",
    n_folds: int = 5,
    test_tournaments: int = 20,
    samples_per_fold: int = 5,
    fit_alpha: bool = True,
    weight_scheme: str = "entropy_squared",
    min_train_tournaments: int = 50,
    progress_bar: bool = True,
) -> Dict[str, Any]:
    """Run simple cross-validation with fast alpha optimization.

    Args:
        engine_class: Rating engine class to use
        engine_params: Parameters for the rating engine
        matches_df: DataFrame with match data
        players_df: DataFrame with player data
        ranking_entity: Entity to rank ("player" or "team")
        prediction_entity: Entity to predict ("team" or "player")
        n_folds: Number of cross-validation folds
        test_tournaments: Number of consecutive tournaments in each test set
        samples_per_fold: Number of samples per fold
        fit_alpha: Whether to optimize the alpha parameter
        weight_scheme: Loss weighting scheme ("none", "entropy", "entropy_squared")
        min_train_tournaments: Minimum tournaments for training
        progress_bar: Whether to show progress bar

    Returns:
        Dictionary with evaluation results including:
        - overall_metrics: Aggregated metrics across all folds
        - fold_results: Results for each fold
        - all_sample_results: Individual results for each sample
    """
    # Create train/test splits
    splits = create_simple_time_splits(
        matches_df=matches_df,
        n_splits=n_folds,
        test_tournaments_per_split=test_tournaments,
        test_samples_per_split=samples_per_fold,
        min_train_tournaments=min_train_tournaments,
    )

    # Run evaluation on each split
    all_results = []
    fold_results = []

    # Track alpha values for warm starting
    alpha_history = []

    # Progress bar setup
    iterator = enumerate(splits)
    if progress_bar:
        iterator = tqdm(iterator, total=len(splits), desc="Evaluating splits")

    for i, (train_df, test_df, test_tournament_ids) in iterator:
        fold_idx = i // samples_per_fold

        # Create and train engine
        engine = engine_class(**engine_params)

        # Compute rankings based on entity type
        if ranking_entity == "player":
            if players_df is None:
                raise ValueError("players_df required for player rankings")
            ratings_df = engine.rank_players(train_df, players_df)

            # Convert to team ratings if we will predict on teams
            if prediction_entity == "team":
                team_ratings_df = derive_team_ratings_from_players(
                    players_df, ratings_df, agg="mean"
                )
                rating_map = dict(
                    zip(
                        team_ratings_df["team_id"],
                        team_ratings_df["team_rating"],
                    )
                )
            else:
                # Create rating map from player ratings
                rating_map = dict(
                    zip(ratings_df["player_id"], ratings_df["player_rank"])
                )
        else:
            # Rank teams directly
            ratings_df = engine.rank_teams(train_df)
            # Create rating map from team ratings
            rating_map = dict(
                zip(ratings_df["team_id"], ratings_df["team_rank"])
            )

        # Compute predictions on test set using rating map
        # We need to compute probabilities based on the rating differences
        if prediction_entity == "team":
            winner_col = "winner_team_id"
            loser_col = "loser_team_id"
        else:
            winner_col = "winner_player_id"
            loser_col = "loser_player_id"

        # Get predictions for all matches in test set
        # For each match, we evaluate BOTH directions to get unbiased evaluation
        test_predictions = []
        skipped_matches = 0

        for _, row in enumerate(test_df.iter_rows(named=True)):
            winner_id = row[winner_col]
            loser_id = row[loser_col]

            winner_rating = rating_map.get(winner_id, 0.0)
            loser_rating = rating_map.get(loser_id, 0.0)

            # Skip matches where either team is completely unrated
            if winner_rating == 0.0 and loser_rating == 0.0:
                skipped_matches += 1
                continue

            # Bradley-Terry probability
            # Use the actual Bradley-Terry model: P(A > B) = score_A / (score_A + score_B)
            prob_winner_wins = winner_rating / (
                winner_rating + loser_rating + 1e-10
            )

            # Add both directions for unbiased evaluation
            # 1. Predict winner vs loser (should have high probability)
            test_predictions.append(
                {
                    "probability": prob_winner_wins,
                    "outcome": 1,  # Winner actually won
                    "winner_rating": winner_rating,
                    "loser_rating": loser_rating,
                }
            )

            # 2. Predict loser vs winner (should have low probability)
            test_predictions.append(
                {
                    "probability": 1
                    - prob_winner_wins,  # P(loser beats winner)
                    "outcome": 0,  # Loser actually lost
                    "winner_rating": winner_rating,
                    "loser_rating": loser_rating,
                }
            )

        # Check if we have any predictions
        if len(test_predictions) == 0:
            logger.warning(
                f"All {skipped_matches} test matches were skipped (unrated teams)"
            )
            # Skip this fold
            continue

        # Extract arrays for loss calculation
        probabilities = np.array([p["probability"] for p in test_predictions])
        outcomes = np.array([p["outcome"] for p in test_predictions])

        # Calculate loss (with optional alpha optimization)
        if fit_alpha:
            # Use warm start if we have history
            warm_start = (
                np.median(alpha_history) if len(alpha_history) >= 3 else None
            )

            alpha = _optimize_alpha_fast(
                matches_df=test_df,
                probabilities=probabilities,
                outcomes=outcomes,
                weight_scheme=weight_scheme,
                warm_start=warm_start,
            )
            alpha_history.append(alpha)
        else:
            alpha = 1.0

        # Calculate weighted loss
        if weight_scheme == "entropy":
            entropy = -probabilities * np.log(probabilities + 1e-10) - (
                1 - probabilities
            ) * np.log(1 - probabilities + 1e-10)
            weights = entropy / np.mean(entropy)
        elif weight_scheme == "entropy_squared":
            entropy = -probabilities * np.log(probabilities + 1e-10) - (
                1 - probabilities
            ) * np.log(1 - probabilities + 1e-10)
            weights = (entropy**2) / np.mean(entropy**2)
        else:
            weights = np.ones_like(probabilities)

        # Calculate loss with alpha scaling
        eps = 1e-15
        clipped_probs = np.clip(probabilities, eps, 1 - eps)

        # Apply alpha scaling
        if alpha != 1.0:
            scaled_probs = 0.5 + (clipped_probs - 0.5) * alpha
            scaled_probs = np.clip(scaled_probs, eps, 1 - eps)
        else:
            scaled_probs = clipped_probs

        # Calculate loss
        base_loss = -np.log(scaled_probs) * outcomes - np.log(
            1 - scaled_probs
        ) * (1 - outcomes)
        weighted_loss = base_loss * weights
        loss = np.mean(weighted_loss)

        # Calculate additional metrics
        # Accuracy: fraction of correct predictions (prob > 0.5 when outcome = 1)
        predictions = (scaled_probs > 0.5).astype(int)
        accuracy = np.mean(predictions == outcomes)

        # Calculate c-stat properly now that we have both outcomes
        # C-stat measures how often the model ranks the winner higher than the loser
        # With our setup, it's the average probability when outcome=1
        winner_probs = scaled_probs[outcomes == 1]
        c_stat = np.mean(winner_probs) if len(winner_probs) > 0 else 0.5

        # Store results
        result = {
            "fold": fold_idx,
            "sample": i % samples_per_fold,
            "loss": loss,
            "accuracy": accuracy,
            "c_stat": c_stat,
            "n_train": train_df.height,
            "n_test": test_df.height,
            "n_test_evaluated": len(test_predictions)
            // 2,  # Divide by 2 since we add 2 predictions per match
            "n_test_skipped": skipped_matches,
            "n_train_tournaments": train_df["tournament_id"].n_unique(),
            "n_test_tournaments": len(test_tournament_ids),
        }

        if fit_alpha:
            result["alpha"] = alpha

        all_results.append(result)

    # Aggregate results by fold
    for fold_idx in range(n_folds):
        fold_samples = [r for r in all_results if r["fold"] == fold_idx]

        fold_result = {
            "fold": fold_idx,
            "n_samples": len(fold_samples),
            "loss": np.mean([r["loss"] for r in fold_samples]),
            "loss_std": np.std([r["loss"] for r in fold_samples]),
            "accuracy": np.mean([r["accuracy"] for r in fold_samples]),
            "c_stat": np.mean([r["c_stat"] for r in fold_samples]),
        }

        if fit_alpha:
            fold_result["alpha"] = np.mean([r["alpha"] for r in fold_samples])
            fold_result["alpha_std"] = np.std(
                [r["alpha"] for r in fold_samples]
            )

        fold_results.append(fold_result)

    # Calculate overall metrics
    overall_metrics = {
        "n_folds": n_folds,
        "n_samples_per_fold": samples_per_fold,
        "n_total_samples": len(all_results),
        "avg_loss": np.mean([r["loss"] for r in all_results]),
        "std_loss": np.std([r["loss"] for r in all_results]),
        "avg_accuracy": np.mean([r["accuracy"] for r in all_results]),
        "avg_c_stat": np.mean([r["c_stat"] for r in all_results]),
    }

    if fit_alpha:
        overall_metrics["avg_alpha"] = np.mean(
            [r["alpha"] for r in all_results]
        )
        overall_metrics["std_alpha"] = np.std([r["alpha"] for r in all_results])

    return {
        "overall_metrics": overall_metrics,
        "fold_results": fold_results,
        "all_sample_results": all_results,
        "weight_scheme": weight_scheme,
        "engine_params": engine_params,
    }
