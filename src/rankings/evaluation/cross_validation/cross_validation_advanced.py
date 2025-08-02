"""
Cross-validation functionality for tournament rating evaluation.

This module implements time-based cross-validation splits and evaluation
procedures for tournament data.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import polars as pl

from rankings.analysis.engine import RatingEngine
from rankings.analysis.utils.summaries import derive_team_ratings_from_players
from rankings.core.constants import MIN_TOURNAMENTS_BEFORE_CV
from rankings.core.logging import get_logger, log_dataframe_stats, log_timing
from rankings.evaluation.loss import (
    compute_cross_tournament_loss,
    compute_tournament_loss,
    filter_matches_by_ranked_threshold,
    fit_alpha_parameter,
    fit_alpha_parameter_optimized,
    fit_alpha_parameter_sampled,
)

# TODO: Incorporate improved splits functionality
from rankings.evaluation.metrics_extras import (
    accuracy_threshold,
    alpha_std,
    concordance,
    placement_spearman,
    skill_score,
    upset_oe,
)


def create_time_based_folds(
    matches_df: pl.DataFrame,
    n_splits: int = 5,
    min_test_tournaments: int = 1,
    min_tournaments_before: int = MIN_TOURNAMENTS_BEFORE_CV,
    tournament_id_col: str = "tournament_id",
    timestamp_col: str = "last_game_finished_at",
) -> List[Tuple[pl.DataFrame, pl.DataFrame, List[int]]]:
    """
    Create time-based cross-validation splits for tournaments.

    Uses a rolling-origin approach where:
    - Training data includes all matches before test tournament start
    - Test data is a single tournament or set of concurrent tournaments

    Parameters
    ----------
    matches_df : pl.DataFrame
        Matches data with tournament IDs and timestamps
    n_splits : int
        Number of CV splits to create
    min_test_tournaments : int
        Minimum tournaments in test set
    min_tournaments_before : int
        Minimum tournaments that must exist before first test tournament
    tournament_id_col : str
        Column name for tournament ID
    timestamp_col : str
        Column name for match timestamp

    Returns
    -------
    List[Tuple[pl.DataFrame, pl.DataFrame, List[int]]]
        List of (train_df, test_df, test_tournament_ids) tuples
    """
    # Get tournament start times
    tournament_times = (
        matches_df.group_by(tournament_id_col)
        .agg(pl.col(timestamp_col).min().alias("start_time"))
        .sort("start_time")
    )

    # Filter to tournaments with enough history
    valid_tournaments = tournament_times.slice(min_tournaments_before)

    if valid_tournaments.height < n_splits * min_test_tournaments:
        raise ValueError(
            f"Not enough tournaments for {n_splits} splits. "
            f"Need at least {n_splits * min_test_tournaments} valid tournaments, "
            f"found {valid_tournaments.height}"
        )

    # Create splits
    splits = []
    tournaments_per_split = valid_tournaments.height // n_splits

    for i in range(n_splits):
        # Determine test tournaments
        test_start_idx = i * tournaments_per_split
        test_end_idx = test_start_idx + min_test_tournaments

        test_tournaments = valid_tournaments.slice(
            test_start_idx, min_test_tournaments
        )
        test_tournament_ids = test_tournaments[tournament_id_col].to_list()

        # Get cutoff time (start of earliest test tournament)
        cutoff_time = test_tournaments["start_time"].min()

        # Create train/test split
        train_df = matches_df.filter(pl.col(timestamp_col) < cutoff_time)
        test_df = matches_df.filter(
            pl.col(tournament_id_col).is_in(test_tournament_ids)
        )

        if train_df.height > 0 and test_df.height > 0:
            splits.append((train_df, test_df, test_tournament_ids))

    return splits


def evaluate_on_split(
    engine_class: Type[RatingEngine],
    engine_params: Dict[str, Any],
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    players_df: Optional[pl.DataFrame] = None,
    teams_df: Optional[pl.DataFrame] = None,
    ranking_entity: str = "player",
    prediction_entity: str = "team",
    agg_func: str = "mean",
    fit_alpha: bool = True,
    alpha: float = 1.0,
    alpha_bounds: tuple[float, float] = (0.1, 5.0),
    score_transform: str = "bradley_terry",
    weight_scheme: str = "entropy",
    weight_threshold: float = 0.1,
    weight_exp_base: float = 2.0,
    warm_start_alpha: Optional[float] = None,
    alpha_fit_method: str = "optimized",
) -> Dict[str, float]:
    """
    Evaluate rating engine on a single train/test split.

    Parameters
    ----------
    engine_class : Type[RatingEngine]
        Rating engine class to instantiate
    engine_params : Dict[str, Any]
        Parameters for engine initialization
    train_df : pl.DataFrame
        Training matches
    test_df : pl.DataFrame
        Test matches
    players_df : Optional[pl.DataFrame]
        Player metadata (if ranking players)
    teams_df : Optional[pl.DataFrame]
        Team metadata (if ranking teams)
    ranking_entity : str
        Entity type to rank ("player" or "team")
    prediction_entity : str
        Entity type to predict on ("player" or "team")
    agg_func : str
        Aggregation function for converting player ratings to team ratings
    fit_alpha : bool
        Whether to fit alpha on training data
    alpha : float
        Fixed alpha value if not fitting
    alpha_bounds : tuple[float, float]
        Bounds for alpha fitting
    score_transform : str
        Transform to apply: "bradley_terry", "logistic", or "identity"
    warm_start_alpha : float, optional
        Alpha value from previous fold to use as warm start
    alpha_fit_method : str
        Method for fitting alpha: "original", "sampled", or "optimized" (default)
    Returns
    -------
    Dict[str, float]
        Evaluation metrics including loss
    """
    logger = get_logger(__name__)
    logger.info(
        f"Evaluating split: rank {ranking_entity}, predict {prediction_entity}, agg={agg_func}"
    )
    log_dataframe_stats(logger, train_df, "train_matches")
    log_dataframe_stats(logger, test_df, "test_matches")

    # Initialize engine and compute ratings on training data
    logger.debug(
        f"Initializing {engine_class.__name__} with params: {engine_params}"
    )
    engine = engine_class(**engine_params)

    # --- ranking / training ---
    with log_timing(logger, f"{ranking_entity} ranking on training data"):
        if ranking_entity == "player":
            if players_df is None:
                raise ValueError("players_df required for player rankings")
            ratings_df = engine.rank_players(train_df, players_df)
            # convert to team ratings if we will predict on teams
            if prediction_entity == "team":
                logger.debug(
                    f"Converting player ratings to team ratings using {agg_func}"
                )
                ratings_df = derive_team_ratings_from_players(
                    players_df, ratings_df, agg=agg_func
                ).rename({"team_rating": "score", "team_id": "id"})
            else:
                ratings_df = ratings_df.rename(
                    {"player_rank": "score", "id": "id"}
                )
        else:
            if teams_df is None:
                # Create teams from matches
                logger.debug("Creating teams from matches")
                teams_df = _create_teams_from_matches(train_df)
            ratings_df = engine.rank_teams(train_df, teams_df)
            if prediction_entity == "player":
                raise ValueError("Cannot predict on players when ranking teams")
            ratings_df = ratings_df.rename(
                {"team_rank": "score", "team_id": "id"}
            )

    log_dataframe_stats(logger, ratings_df, "computed_ratings")

    # Create rating map
    rating_map = dict(
        zip(ratings_df["id"].to_list(), ratings_df["score"].to_list())
    )
    logger.debug(f"Created rating map with {len(rating_map)} entities")

    # Determine column names for prediction
    if prediction_entity == "team":
        winner_col = "winner_team_id"
        loser_col = "loser_team_id"
    else:
        winner_col = "winner_user_id"
        loser_col = "loser_user_id"

    # Apply 75% threshold filtering for evaluation
    if prediction_entity == "team" and players_df is not None:
        logger.debug("Applying 75% threshold filtering for evaluation")
        original_test_size = test_df.height

        # Note: We don't filter training data here as it's already been used for ranking
        # We only filter test data for fair evaluation
        test_df = filter_matches_by_ranked_threshold(
            test_df,
            players_df,
            rating_map,
            threshold=0.75,
            winner_col=winner_col,
            loser_col=loser_col,
        )

        logger.debug(
            f"Filtered test matches by 75% threshold: {original_test_size} -> {test_df.height} "
            f"({test_df.height/original_test_size*100:.1f}% kept)"
        )

    # Fit alpha if requested
    if fit_alpha:
        logger.debug(f"Fitting alpha parameter using {alpha_fit_method} method")

        # Select alpha fitting function
        if alpha_fit_method == "original":
            alpha_func = fit_alpha_parameter
        elif alpha_fit_method == "sampled":
            alpha_func = fit_alpha_parameter_sampled
        else:  # default to optimized
            alpha_func = fit_alpha_parameter_optimized

        alpha = alpha_func(
            train_df,
            rating_map,
            winner_id_col=winner_col,
            loser_id_col=loser_col,
            alpha_bounds=alpha_bounds,
            score_transform=score_transform,
            scheme=weight_scheme,
            global_prior=getattr(engine, "global_prior_", None),
            players_df=players_df,
            apply_threshold_filter=(prediction_entity == "team"),
            threshold=0.75,
            weight_threshold=weight_threshold,
            weight_exp_base=weight_exp_base,
        )
    else:
        logger.debug(f"Using fixed alpha={alpha}")

    # Evaluate on test tournaments
    test_tournaments = test_df["tournament_id"].unique().to_list()
    logger.info(f"Evaluating on {len(test_tournaments)} test tournaments")
    tournament_losses = []
    all_metrics = []

    for tournament_id in test_tournaments:
        tournament_matches = test_df.filter(
            pl.col("tournament_id") == tournament_id
        )

        loss, metrics = compute_tournament_loss(
            tournament_matches,
            rating_map,
            alpha=alpha,
            score_transform=score_transform,
            scheme=weight_scheme,
            winner_id_col=winner_col,
            loser_id_col=loser_col,
            return_predictions=True,
            global_prior=getattr(engine, "global_prior_", None),
            players_df=players_df,
            weight_threshold=weight_threshold,
            weight_exp_base=weight_exp_base,
            apply_threshold_filter=False,  # Already filtered above
            threshold=0.75,
        )

        # Compute additional metrics
        try:
            metrics["c_stat"] = concordance(
                tournament_matches,
                rating_map,
                winner_col=winner_col,
                loser_col=loser_col,
            )
        except Exception:
            metrics["c_stat"] = np.nan

        try:
            metrics["skill_score"] = skill_score(
                tournament_matches,
                rating_map,
                alpha=alpha,
                score_transform=score_transform,
                winner_col=winner_col,
                loser_col=loser_col,
            )
        except Exception:
            metrics["skill_score"] = np.nan

        try:
            predictions = metrics.get("predictions", np.array([]))
            metrics["upset_oe"] = upset_oe(predictions)
            metrics["acc_conf"] = accuracy_threshold(predictions, 0.65)
        except Exception:
            metrics["upset_oe"] = np.nan
            metrics["acc_conf"] = np.nan

        tournament_losses.append(loss)
        metrics["tournament_id"] = tournament_id
        all_metrics.append(metrics)

    # Aggregate results
    avg_loss = compute_cross_tournament_loss(tournament_losses)
    logger.info(
        f"Split evaluation complete: avg_loss={avg_loss:.4f}, alpha={alpha:.4f}"
    )

    results = {
        "loss": avg_loss,
        "alpha": alpha,
        "n_test_tournaments": len(test_tournaments),
        "n_train_matches": train_df.height,
        "n_test_matches": test_df.height,
        "tournament_losses": tournament_losses,
        "per_tournament_metrics": all_metrics,
    }

    # Add average metrics
    if all_metrics:
        for key in [
            "accuracy",
            "mean_probability",
            "c_stat",
            "skill_score",
            "upset_oe",
            "acc_conf",
        ]:
            values = [
                m.get(key, 0)
                for m in all_metrics
                if key in m and np.isfinite(m[key])
            ]
            if values:
                results[f"avg_{key}"] = np.mean(values)

    return results


def cross_validate_ratings(
    engine_class: Type[RatingEngine],
    engine_params: Dict[str, Any],
    matches_df: pl.DataFrame,
    players_df: Optional[pl.DataFrame] = None,
    teams_df: Optional[pl.DataFrame] = None,
    ranking_entity: str = "player",
    prediction_entity: str = "team",
    agg_func: str = "mean",
    n_splits: int = 5,
    fit_alpha: bool = True,
    alpha_bounds: tuple[float, float] = (0.1, 5.0),
    score_transform: str = "bradley_terry",
    regularization_lambda: float = 0.0,
    default_params: Optional[Dict[str, Any]] = None,
    weight_scheme: str = "entropy",
    weight_threshold: float = 0.1,
    weight_exp_base: float = 2.0,
    min_test_tournaments: int = 10,
    test_size_ratio: float = 0.2,
    alpha_fit_method: str = "optimized",
) -> Dict[str, Any]:
    """
    Perform full cross-validation evaluation of rating engine.

    Parameters
    ----------
    engine_class : Type[RatingEngine]
        Rating engine class
    engine_params : Dict[str, Any]
        Engine parameters to evaluate
    matches_df : pl.DataFrame
        All matches data
    players_df : Optional[pl.DataFrame]
        Player metadata
    teams_df : Optional[pl.DataFrame]
        Team metadata
    ranking_entity : str
        Entity type to rank ("player" or "team")
    prediction_entity : str
        Entity type to predict on ("player" or "team")
    agg_func : str
        Aggregation function for converting player ratings to team ratings
    n_splits : int
        Number of CV folds
    fit_alpha : bool
        Whether to fit alpha parameter
    alpha_bounds : tuple[float, float]
        Bounds for alpha fitting
    score_transform : str
        Transform to apply: "bradley_terry", "logistic", or "identity"
    regularization_lambda : float
        L2 regularization strength
    default_params : Optional[Dict[str, Any]]
        Default parameters for regularization
    min_test_tournaments : int
        Minimum number of tournaments per test fold (default: 10)
    test_size_ratio : float
        Approximate ratio of tournaments for testing (default: 0.2)
    alpha_fit_method : str
        Method for fitting alpha: "original", "sampled", or "optimized" (default)

    Returns
    -------
    Dict[str, Any]
        Cross-validation results including average loss
    """
    logger = get_logger(__name__)
    logger.info(f"Starting {n_splits}-fold cross-validation")
    logger.info(f"Engine: {engine_class.__name__}, params: {engine_params}")
    logger.info(
        f"Config: rank {ranking_entity}, predict {prediction_entity}, agg={agg_func}"
    )
    log_dataframe_stats(logger, matches_df, "cv_matches")

    # Create splits using improved method
    logger.debug("Creating time-based splits with improved method")
    try:
        # TODO: Re-implement improved splits functionality
        splits = create_time_based_folds(
            matches_df,
            n_splits=n_splits,
            min_test_tournaments=min_test_tournaments,
            test_size_ratio=test_size_ratio,
        )
        logger.info(f"Created {len(splits)} CV splits with improved method")
    except ValueError as e:
        logger.warning(
            f"Improved splits failed: {e}. Falling back to legacy method"
        )
        splits = create_time_based_folds(matches_df, n_splits=n_splits)
        logger.info(f"Created {len(splits)} CV splits with legacy method")

    # Evaluate on each split
    split_results = []
    previous_alpha = None  # For warm starts

    with log_timing(logger, f"{n_splits}-fold cross-validation"):
        for i, (train_df, test_df, test_ids) in enumerate(splits):
            logger.info(f"Evaluating split {i+1}/{n_splits}...")

            results = evaluate_on_split(
                engine_class=engine_class,
                engine_params=engine_params,
                train_df=train_df,
                test_df=test_df,
                players_df=players_df,
                teams_df=teams_df,
                ranking_entity=ranking_entity,
                prediction_entity=prediction_entity,
                agg_func=agg_func,
                fit_alpha=fit_alpha,
                alpha_bounds=alpha_bounds,
                score_transform=score_transform,
                weight_scheme=weight_scheme,
                weight_threshold=weight_threshold,
                weight_exp_base=weight_exp_base,
                warm_start_alpha=previous_alpha,
                alpha_fit_method=alpha_fit_method,
            )

            results["split_id"] = i
            results["test_tournament_ids"] = test_ids
            split_results.append(results)
            logger.debug(f"Split {i+1} loss: {results['loss']:.4f}")

            # Update warm start for next fold
            if fit_alpha:
                previous_alpha = results.get("alpha", 1.0)

    # Compute overall metrics
    losses = [r["loss"] for r in split_results]
    avg_loss = np.mean(losses)
    std_loss = np.std(losses)

    # Compute alpha_std if alphas were fitted
    alphas = [r.get("alpha", np.nan) for r in split_results]
    alpha_variability = alpha_std(alphas) if fit_alpha else np.nan

    # Add regularization if specified
    if regularization_lambda > 0 and default_params:
        logger.debug(
            f"Applying L2 regularization with lambda={regularization_lambda}"
        )
        reg_term = 0.0
        for key, value in engine_params.items():
            if key in default_params:
                default = default_params[key]
                if isinstance(value, (int, float)) and isinstance(
                    default, (int, float)
                ):
                    reg_term += (value - default) ** 2

        regularized_loss = avg_loss + regularization_lambda * reg_term
        logger.debug(
            f"Regularization term: {reg_term:.6f}, regularized loss: {regularized_loss:.4f}"
        )
    else:
        regularized_loss = avg_loss

    logger.info(
        f"Cross-validation complete: avg_loss={avg_loss:.4f} ± {std_loss:.4f}"
    )

    # Aggregate other metrics across splits
    aggregated_metrics = {}
    for metric in [
        "c_stat",
        "skill_score",
        "upset_oe",
        "acc_conf",
        "accuracy",
        "mean_probability",
    ]:
        values = []
        for r in split_results:
            avg_key = f"avg_{metric}"
            if avg_key in r and np.isfinite(r[avg_key]):
                values.append(r[avg_key])

        if values:
            aggregated_metrics[f"avg_{metric}"] = np.mean(values)
            aggregated_metrics[f"std_{metric}"] = np.std(values)

    return {
        "avg_loss": avg_loss,
        "regularized_loss": regularized_loss,
        "std_loss": std_loss,
        "alpha_std": alpha_variability,
        "split_results": split_results,
        "engine_params": engine_params,
        "n_splits": n_splits,
        **aggregated_metrics,
    }


def _create_teams_from_matches(matches_df: pl.DataFrame) -> pl.DataFrame:
    """Create teams dataframe from matches."""
    winner_teams = matches_df.select(
        pl.col("winner_team_id").alias("team_id")
    ).unique()

    loser_teams = matches_df.select(
        pl.col("loser_team_id").alias("team_id")
    ).unique()

    all_teams = pl.concat([winner_teams, loser_teams]).unique()

    # Add required columns
    return all_teams.with_columns(
        [pl.lit(None).alias("name"), pl.lit(1).alias("player_count")]
    )
