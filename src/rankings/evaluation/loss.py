"""
Loss functions for evaluating tournament rating predictions.

This module implements the cross-entropy loss function described in plan.md,
which measures how well ratings predict match outcomes.
"""

import logging

import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar

from rankings.analysis.transforms import bt_prob
from rankings.core.logging import get_logger, log_timing


def get_team_ranked_ratio(
    team_id: int,
    tournament_id: int | None,
    players_df: pl.DataFrame,
    rating_map: dict[int, float],
) -> float:
    """
    Calculate the percentage of ranked players in a team.

    Parameters
    ----------
    team_id : int
        Team ID
    tournament_id : int
        Tournament ID
    players_df : pl.DataFrame
        Players dataframe with team rosters
    rating_map : dict[int, float]
        Mapping from player ID to rating

    Returns
    -------
    float
        Ratio of ranked players (0.0 to 1.0)
    """
    if tournament_id is None:
        # If no tournament_id, just filter by team
        roster = players_df.filter(pl.col("team_id") == team_id)
    else:
        roster = players_df.filter(
            (pl.col("team_id") == team_id)
            & (pl.col("tournament_id") == tournament_id)
        )

    if roster.height == 0:
        return 0.0

    total_players = roster.height
    ranked_players = sum(
        1 for uid in roster["user_id"].to_list() if uid in rating_map
    )

    return ranked_players / total_players


def filter_matches_by_ranked_threshold(
    matches_df: pl.DataFrame,
    players_df: pl.DataFrame,
    rating_map: dict[int, float],
    threshold: float = 0.75,
    winner_col: str = "winner_team_id",
    loser_col: str = "loser_team_id",
) -> pl.DataFrame:
    """
    Filter matches where both teams have at least threshold % ranked players.

    Parameters
    ----------
    matches_df : pl.DataFrame
        Matches to filter
    players_df : pl.DataFrame
        Players dataframe with team rosters
    rating_map : dict[int, float]
        Mapping from entity ID to rating
    threshold : float
        Minimum ratio of ranked players required (default: 0.75)
    winner_col : str
        Column name for winner ID
    loser_col : str
        Column name for loser ID

    Returns
    -------
    pl.DataFrame
        Filtered matches
    """
    filtered = []

    for row in matches_df.iter_rows(named=True):
        winner_ratio = get_team_ranked_ratio(
            row[winner_col], row["tournament_id"], players_df, rating_map
        )
        loser_ratio = get_team_ranked_ratio(
            row[loser_col], row["tournament_id"], players_df, rating_map
        )

        # Keep if both teams meet the threshold
        if winner_ratio >= threshold and loser_ratio >= threshold:
            filtered.append(row)

    if not filtered:
        return pl.DataFrame([], schema=matches_df.schema)

    return pl.DataFrame(filtered, schema=matches_df.schema)


def compute_match_probability(
    rating_a: float,
    rating_b: float,
    *,
    alpha: float = 1.0,
    score_transform: str = "bradley_terry",
    global_prior: float | None = None,
) -> float:
    """
    Compute probability that team/player A beats team/player B.

    Parameters
    ----------
    rating_a : float
        Rating of team/player A
    rating_b : float
        Rating of team/player B
    alpha : float
        Temperature parameter (higher = more deterministic)
    score_transform : str
        Transform to apply: "bradley_terry", "logistic", or "identity"
    global_prior : float, optional
        Prior rating for unknown players (defaults to 0.05)

    Returns
    -------
    float
        Probability that A beats B
    """
    prior = global_prior if global_prior is not None else 0.05
    rating_a = float(rating_a) if rating_a is not None else prior
    rating_b = float(rating_b) if rating_b is not None else prior

    if score_transform == "bradley_terry":
        # Use Bradley-Terry model with positive scores
        return bt_prob(rating_a, rating_b, alpha=alpha)
    elif score_transform == "logistic":
        # Legacy logistic on raw score difference
        z = alpha * (rating_a - rating_b)
        return 1.0 / (1.0 + np.exp(-z))
    elif score_transform == "identity":
        # Assume ratings are already probabilities
        return rating_a
    else:
        raise ValueError(f"Unknown score_transform: {score_transform}")


def compute_match_loss(
    p_win: float, actual_winner_is_a: bool, eps: float = 1e-15
) -> float:
    """
    Compute cross-entropy loss for a single match.

    Parameters
    ----------
    p_win : float
        Predicted probability that A wins
    actual_winner_is_a : bool
        True if A actually won, False if B won
    eps : float
        Small value to prevent log(0)

    Returns
    -------
    float
        Negative log-likelihood loss
    """
    p_win = np.clip(p_win, eps, 1.0 - eps)

    if actual_winner_is_a:
        return -np.log(p_win)
    else:
        return -np.log(1.0 - p_win)


def compute_tournament_loss(
    matches_df: pl.DataFrame,
    rating_map: dict[int, float],
    *,
    alpha: float = 1.0,
    score_transform: str = "bradley_terry",
    scheme: str = "entropy",
    winner_id_col: str = "winner_team_id",
    loser_id_col: str = "loser_team_id",
    return_predictions: bool = False,
    global_prior: float | None = None,
    players_df: pl.DataFrame | None = None,
    apply_threshold_filter: bool = True,
    threshold: float = 0.75,
    weight_threshold: float = 0.1,
    weight_exp_base: float = 2.0,
) -> tuple[float, dict[str, any]]:
    """
    Compute aggregated loss for a tournament.

    Parameters
    ----------
    matches_df : pl.DataFrame
        Matches dataframe with winner and loser ID columns
    rating_map : Dict[int, float]
        Mapping from team/player ID to rating
    alpha : float
        Temperature parameter
    score_transform : str
        Transform to apply: "bradley_terry", "logistic", or "identity"
    scheme : str
        Weighting scheme: "none", "var_inv", "entropy", "entropy_squared",
        "entropy_exp", or "threshold"
    winner_id_col : str
        Column name for winner ID
    loser_id_col : str
        Column name for loser ID
    return_predictions : bool
        If True, include prediction probabilities in return dict
    global_prior : float, optional
        Prior rating for unknown players (defaults to 0.05)
    players_df : pl.DataFrame, optional
        Players dataframe for threshold filtering
    apply_threshold_filter : bool
        If True, filter matches by ranked player threshold
    threshold : float
        Minimum ratio of ranked players required (default: 0.75)
    weight_threshold : float
        For threshold scheme, minimum distance from 0.5 to include (default: 0.1)
    weight_exp_base : float
        Base for exponential weighting scheme (default: 2.0)

    Returns
    -------
    Tuple[float, Dict[str, any]]
        (tournament_loss, metrics_dict with optional predictions)
    """
    logger = get_logger(__name__)
    original_count = matches_df.height
    logger.debug(
        f"Computing tournament loss for {original_count} matches with alpha={alpha}"
    )

    # Apply threshold filtering if requested
    if (
        apply_threshold_filter
        and players_df is not None
        and winner_id_col == "winner_team_id"
    ):
        matches_df = filter_matches_by_ranked_threshold(
            matches_df,
            players_df,
            rating_map,
            threshold,
            winner_id_col,
            loser_id_col,
        )
        filtered_count = matches_df.height
        logger.debug(
            f"Filtered matches by {threshold:.0%} threshold: {original_count} -> {filtered_count} "
            f"({filtered_count/original_count*100:.1f}% kept)"
        )

    # Extract winner and loser IDs
    match_data = matches_df.select(
        [
            pl.col(winner_id_col).alias("winner"),
            pl.col(loser_id_col).alias("loser"),
        ]
    )

    predictions = []

    with log_timing(
        logger,
        f"loss computation for {matches_df.height} matches",
        level=logging.DEBUG,
    ):
        for row in match_data.iter_rows(named=True):
            winner_id = row["winner"]
            loser_id = row["loser"]

            # Get ratings (default to global_prior if not found)
            prior = global_prior if global_prior is not None else 0.05
            r_winner = rating_map.get(winner_id, prior)
            r_loser = rating_map.get(loser_id, prior)

            # Compute probability that winner beats loser
            p_win = compute_match_probability(
                r_winner,
                r_loser,
                alpha=alpha,
                score_transform=score_transform,
                global_prior=global_prior,
            )
            predictions.append(p_win)

    # Convert to numpy arrays
    predictions = np.array(predictions)

    if len(predictions) == 0:
        logger.warning("No matches found, returning infinite loss")
        return np.inf, {"n_matches": 0}

    # Compute loss using enhanced weighting schemes
    from .enhanced_weighting import compute_confidence_weights

    base = -np.log(np.clip(predictions, 1e-15, 1.0 - 1e-15))  # Bernoulli y=1

    # Use enhanced weighting schemes
    w = compute_confidence_weights(
        predictions,
        scheme=scheme,
        threshold=weight_threshold,
        exp_base=weight_exp_base,
    )

    final_loss = np.average(base, weights=w)

    # Compute additional metrics
    metrics = {
        "n_matches": len(predictions),
        "n_matches_original": original_count,
        "n_matches_filtered": original_count - matches_df.height
        if apply_threshold_filter and players_df is not None
        else 0,
        "mean_probability": np.mean(predictions),
        "accuracy": np.mean(predictions > 0.5),
        "unweighted_loss": np.mean(base),
        "weighted_loss": final_loss,
    }

    # Include predictions if requested
    if return_predictions:
        metrics["predictions"] = predictions

        # Add bucketised analysis
        from .metrics import compute_brier_score

        outcomes = np.ones_like(predictions)  # All winners
        metrics["brier_score"] = compute_brier_score(predictions, outcomes)
        metrics["bucketised_metrics"] = bucketised_metrics(predictions, base)

    logger.debug(
        f"Tournament loss computed: {final_loss:.4f} (accuracy: {metrics['accuracy']:.3f})"
    )
    return final_loss, metrics


def compute_cross_tournament_loss(tournament_losses: list[float]) -> float:
    """
    Compute average loss across multiple tournaments.

    Parameters
    ----------
    tournament_losses : List[float]
        Loss values for each tournament

    Returns
    -------
    float
        Average tournament loss
    """
    if not tournament_losses:
        return np.inf
    return np.mean(tournament_losses)


def compute_weighted_log_loss(
    p: np.ndarray, eps: float = 1e-15, scheme: str = "entropy"
) -> float:
    """
    Compute weighted cross-entropy loss to handle coin-flip matches appropriately.

    Addresses the issue from plan.md where models are over-penalized for being
    uncertain on inherently uncertain matches.

    Parameters
    ----------
    p : np.ndarray
        Predicted win probabilities for the observed winner
    eps : float
        Small value to prevent log(0)
    scheme : str
        Weighting scheme:
        - 'entropy': weight = |p - 0.5| * 2 (zero at 0.5, max at 0/1)
        - 'var_inv': weight = 1 / (p * (1-p)) (inverse variance weighting)
        - 'none': classic unweighted negative log-likelihood

    Returns
    -------
    float
        Weighted cross-entropy loss
    """
    p = np.clip(p, eps, 1 - eps)
    base = -np.log(p)  # cross-entropy because y=1 for winner

    if scheme == "entropy":
        w = np.abs(p - 0.5) * 2  # ranges 0…1
    elif scheme == "var_inv":
        w = 1.0 / (p * (1 - p))
    else:
        w = 1.0

    return (base * w).mean()


def bucketised_metrics(
    p: np.ndarray, loss: np.ndarray, n_buckets: int = 5
) -> pl.DataFrame:
    """
    Evaluate model performance by predicted win probability buckets.

    This addresses the plan.md recommendation to report log-loss in buckets
    to better understand where the model performs well vs poorly.

    Parameters
    ----------
    p : np.ndarray
        Predicted win probabilities
    loss : np.ndarray
        Loss values for each prediction
    n_buckets : int
        Number of probability buckets (default 5)

    Returns
    -------
    pl.DataFrame
        DataFrame with bucket, count, and mean loss for each bucket
    """
    bins = np.linspace(0.5, 1.0, n_buckets + 1)
    labels = [f"{b:.2f}-{bins[i+1]:.2f}" for i, b in enumerate(bins[:-1])]
    idx = np.digitize(p, bins, right=False) - 1

    # Handle edge cases
    idx = np.clip(idx, 0, n_buckets - 1)

    df = pl.DataFrame({"bucket": idx, "loss": loss})
    return (
        df.group_by("bucket")
        .agg(
            [
                pl.len().alias("count"),
                pl.col("loss").mean().alias("mean_loss"),
            ]
        )
        .sort("bucket")
        .with_columns(
            pl.col("bucket")
            .map_elements(lambda i: labels[i], return_dtype=pl.String)
            .alias("bucket_range")
        )
        .select(["bucket_range", "count", "mean_loss"])
    )


def fit_alpha_parameter(
    train_matches_df: pl.DataFrame,
    rating_map: dict[int, float],
    alpha_bounds: tuple[float, float] = (0.1, 5.0),
    score_transform: str = "bradley_terry",
    winner_id_col: str = "winner_team_id",
    loser_id_col: str = "loser_team_id",
    scheme: str = "entropy",
    global_prior: float | None = None,
    players_df: pl.DataFrame | None = None,
    apply_threshold_filter: bool = True,
    threshold: float = 0.75,
    weight_threshold: float = 0.1,
    weight_exp_base: float = 2.0,
) -> float:
    """
    Fit the alpha parameter using maximum likelihood on training data.

    Parameters
    ----------
    train_matches_df : pl.DataFrame
        Training matches
    rating_map : Dict[int, float]
        Pre-computed ratings
    alpha_bounds : Tuple[float, float]
        Bounds for alpha search
    score_transform : str
        Transform to apply: "bradley_terry", "logistic", or "identity"
    winner_id_col : str
        Column name for winner ID
    loser_id_col : str
        Column name for loser ID
    scheme : str
        Weighting scheme: "none", "var_inv", "entropy", "entropy_squared",
        "entropy_exp", or "threshold"
    global_prior : float, optional
        Prior rating for unknown players (defaults to 0.05)
    players_df : pl.DataFrame, optional
        Players dataframe for threshold filtering
    apply_threshold_filter : bool
        If True, filter matches by ranked player threshold
    threshold : float
        Minimum ratio of ranked players required (default: 0.75)
    weight_threshold : float
        For threshold scheme, minimum distance from 0.5 to include (default: 0.1)
    weight_exp_base : float
        Base for exponential weighting scheme (default: 2.0)
    Returns
    -------
    float
        Optimal alpha value
    """
    logger = get_logger(__name__)
    logger.info(
        f"Fitting alpha parameter on {train_matches_df.height} training matches"
    )
    logger.debug(f"Alpha bounds: {alpha_bounds}")

    def neg_log_likelihood(alpha: float) -> float:
        loss, metrics = compute_tournament_loss(
            train_matches_df,
            rating_map,
            alpha=alpha,
            score_transform=score_transform,
            scheme=scheme,
            winner_id_col=winner_id_col,
            loser_id_col=loser_id_col,
            return_predictions=False,
            global_prior=global_prior,
            players_df=players_df,
            apply_threshold_filter=apply_threshold_filter,
            threshold=threshold,
            weight_threshold=weight_threshold,
            weight_exp_base=weight_exp_base,
        )
        # Use actual number of matches after filtering
        n_matches = metrics.get("n_matches", len(train_matches_df))
        return loss * n_matches

    # Use scipy's bounded minimization
    with log_timing(logger, "alpha parameter optimization"):
        result = minimize_scalar(
            neg_log_likelihood, bounds=alpha_bounds, method="bounded"
        )

    optimal_alpha = result.x
    logger.info(f"Optimal alpha found: {optimal_alpha:.6f}")
    logger.info(
        f"Loss at optimal alpha: {neg_log_likelihood(optimal_alpha):.6f}"
    )

    return optimal_alpha


def analyze_exclusion_impact(
    matches_df: pl.DataFrame,
    players_df: pl.DataFrame,
    rating_map: dict[int, float],
    threshold: float = 0.75,
    winner_col: str = "winner_team_id",
    loser_col: str = "loser_team_id",
) -> dict[str, any]:
    """
    Analyze the impact of match exclusion based on ranked player threshold.

    Parameters
    ----------
    matches_df : pl.DataFrame
        Matches to analyze
    players_df : pl.DataFrame
        Players dataframe with team rosters
    rating_map : dict[int, float]
        Mapping from entity ID to rating
    threshold : float
        Minimum ratio of ranked players required
    winner_col : str
        Column name for winner ID
    loser_col : str
        Column name for loser ID

    Returns
    -------
    dict[str, any]
        Analysis results including counts and percentages
    """
    logger = get_logger(__name__)

    total_matches = matches_df.height

    # Get filtered matches
    filtered_df = filter_matches_by_ranked_threshold(
        matches_df, players_df, rating_map, threshold, winner_col, loser_col
    )
    kept_matches = filtered_df.height
    excluded_matches = total_matches - kept_matches

    # Analyze by tournament
    tournament_stats = []
    for tid in matches_df["tournament_id"].unique():
        tournament_matches = matches_df.filter(pl.col("tournament_id") == tid)
        tournament_filtered = filtered_df.filter(pl.col("tournament_id") == tid)

        tournament_stats.append(
            {
                "tournament_id": tid,
                "total_matches": tournament_matches.height,
                "kept_matches": tournament_filtered.height,
                "excluded_matches": tournament_matches.height
                - tournament_filtered.height,
                "kept_percentage": tournament_filtered.height
                / tournament_matches.height
                * 100
                if tournament_matches.height > 0
                else 0,
            }
        )

    tournament_df = pl.DataFrame(tournament_stats).sort("kept_percentage")

    # Summary statistics
    summary = {
        "total_matches": total_matches,
        "kept_matches": kept_matches,
        "excluded_matches": excluded_matches,
        "kept_percentage": kept_matches / total_matches * 100
        if total_matches > 0
        else 0,
        "excluded_percentage": excluded_matches / total_matches * 100
        if total_matches > 0
        else 0,
        "threshold": threshold,
        "tournament_stats": tournament_df,
        "most_affected_tournaments": tournament_df.head(10),
        "least_affected_tournaments": tournament_df.tail(10),
    }

    logger.info(f"Match exclusion analysis with {threshold:.0%} threshold:")
    logger.info(f"  Total matches: {total_matches:,}")
    logger.info(
        f"  Kept matches: {kept_matches:,} ({summary['kept_percentage']:.1f}%)"
    )
    logger.info(
        f"  Excluded: {excluded_matches:,} ({summary['excluded_percentage']:.1f}%)"
    )

    return summary
