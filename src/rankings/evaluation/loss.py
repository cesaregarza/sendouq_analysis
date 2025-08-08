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


def filter_matches_by_unrated_count(
    matches_df: pl.DataFrame,
    players_df: pl.DataFrame,
    rating_map: dict[int, float],
    max_unrated_per_team: int = 1,
    winner_col: str = "winner_team_id",
    loser_col: str = "loser_team_id",
) -> pl.DataFrame:
    """
    Filter matches where both teams have at most max_unrated_per_team unrated players.

    This implements the "≥2 unrated rule" from plan.md - excludes matches where
    a team has 2 or more unrated players.

    Parameters
    ----------
    matches_df : pl.DataFrame
        Matches to filter
    players_df : pl.DataFrame
        Players dataframe with team rosters
    rating_map : dict[int, float]
        Mapping from entity ID to rating
    max_unrated_per_team : int
        Maximum number of unrated players allowed per team (default: 1)
    winner_col : str
        Column name for winner ID
    loser_col : str
        Column name for loser ID

    Returns
    -------
    pl.DataFrame
        Filtered matches
    """
    # Build a (tournament_id, team_id) -> unrated_count map
    roster = players_df.select(["tournament_id", "team_id", "user_id"])

    # Check if each player is unrated
    is_unrated = roster["user_id"].map_elements(
        lambda u: u not in rating_map, return_dtype=pl.Boolean
    )
    roster = roster.with_columns(is_unrated.alias("unrated"))

    # Count unrated players per team
    counts = roster.group_by(["tournament_id", "team_id"]).agg(
        pl.col("unrated").sum().alias("unrated_count")
    )

    # Join counts to matches for both winner and loser teams
    matches_with_counts = (
        matches_df.join(
            counts,
            left_on=["tournament_id", winner_col],
            right_on=["tournament_id", "team_id"],
            how="left",
        )
        .rename({"unrated_count": "winner_unrated"})
        .join(
            counts,
            left_on=["tournament_id", loser_col],
            right_on=["tournament_id", "team_id"],
            how="left",
        )
        .rename({"unrated_count": "loser_unrated"})
        .fill_null(0)  # Teams with no players in roster get 0 unrated
    )

    # Filter matches where both teams have <= max_unrated_per_team
    return matches_with_counts.filter(
        (pl.col("winner_unrated") <= max_unrated_per_team)
        & (pl.col("loser_unrated") <= max_unrated_per_team)
    ).drop(["winner_unrated", "loser_unrated"])


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
        Weighting scheme: "none", "var_inv", "entropy", "entropy_sqrt",
        "entropy_squared", "entropy_exp", or "threshold"
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

    # Compute loss using weighting schemes
    base = -np.log(np.clip(predictions, 1e-15, 1.0 - 1e-15))  # Bernoulli y=1

    # Apply weighting scheme
    if scheme == "none":
        w = np.ones_like(predictions)
    elif scheme == "entropy":
        w = np.abs(predictions - 0.5) * 2  # ranges 0…1
    elif scheme == "entropy_sqrt":
        w = np.sqrt(np.abs(predictions - 0.5) * 2)  # sqrt of confidence
    elif scheme == "entropy_squared":
        w = (np.abs(predictions - 0.5) * 2) ** 2  # squared confidence
    elif scheme == "var_inv":
        w = 1.0 / (predictions * (1 - predictions))
    elif scheme == "entropy_exp":
        w = weight_exp_base ** (np.abs(predictions - 0.5) * 2)
    elif scheme == "threshold":
        # Only weight matches that are sufficiently different from 0.5
        w = np.where(np.abs(predictions - 0.5) > weight_threshold, 1.0, 0.1)
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")

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
        from rankings.evaluation.metrics import compute_brier_score

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
    elif scheme == "entropy_sqrt":
        w = np.sqrt(np.abs(p - 0.5) * 2)  # sqrt of confidence
    elif scheme == "entropy_squared":
        w = (np.abs(p - 0.5) * 2) ** 2  # squared confidence
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
        Weighting scheme: "none", "var_inv", "entropy", "entropy_sqrt",
        "entropy_squared", "entropy_exp", or "threshold"
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


def fit_alpha_parameter_sampled(
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
    sample_size: int = 1000,
    n_grid_points: int = 20,
) -> float:
    """
    Fit alpha parameter using sampling and grid search.

    Key optimizations:
    1. Sample matches instead of using all
    2. Pre-compute aggregated ratings once
    3. Use grid search with vectorized operations
    4. Cache filtering results

    Parameters
    ----------
    sample_size : int
        Number of matches to sample for optimization (default: 1000)
    n_grid_points : int
        Number of points in initial grid search (default: 20)

    Other parameters same as fit_alpha_parameter

    Returns
    -------
    float
        Optimal alpha value
    """
    logger = get_logger(__name__)
    logger.info(
        f"Sampled alpha fitting on {train_matches_df.height} training matches"
    )

    # Sample matches if dataset is large
    if train_matches_df.height > sample_size:
        train_sample = train_matches_df.sample(n=sample_size, seed=42)
        logger.debug(f"Sampled {sample_size} matches for optimization")
    else:
        train_sample = train_matches_df

    # Get team ratings directly from rating_map
    # This assumes team_id exists in rating_map or we aggregate from members
    prior = global_prior if global_prior is not None else 0.05

    # Skip filtering in sampled version for speed

    # Extract winner/loser ratings
    if (
        "winner_members" in train_sample.columns
        and "loser_members" in train_sample.columns
    ):
        # Aggregate from team members
        winner_ratings = (
            train_sample["winner_members"]
            .map_elements(
                lambda members: np.mean(
                    [rating_map.get(p, prior) for p in members]
                )
                if members is not None and len(members) > 0
                else prior,
                return_dtype=pl.Float64,
            )
            .to_numpy()
        )

        loser_ratings = (
            train_sample["loser_members"]
            .map_elements(
                lambda members: np.mean(
                    [rating_map.get(p, prior) for p in members]
                )
                if members is not None and len(members) > 0
                else prior,
                return_dtype=pl.Float64,
            )
            .to_numpy()
        )
    else:
        # Direct lookup from rating_map
        winner_ratings = np.array(
            [rating_map.get(tid, prior) for tid in train_sample[winner_id_col]]
        )
        loser_ratings = np.array(
            [rating_map.get(tid, prior) for tid in train_sample[loser_id_col]]
        )

    # Compute match weights once
    if scheme != "none":
        # Compute base probabilities with alpha=1 for weighting
        base_probs = compute_probabilities(
            winner_ratings,
            loser_ratings,
            alpha=1.0,
            score_transform=score_transform,
            global_prior=global_prior,
        )
        weights = compute_weights(
            base_probs,
            scheme=scheme,
            weight_threshold=weight_threshold,
            weight_exp_base=weight_exp_base,
        )
    else:
        weights = np.ones(len(winner_ratings))

    # Vectorized loss computation
    def compute_loss_vectorized(alpha: float) -> float:
        probs = compute_probabilities(
            winner_ratings,
            loser_ratings,
            alpha=alpha,
            score_transform=score_transform,
            global_prior=global_prior,
        )
        # All matches have winner as first team by construction
        losses = -np.log(np.clip(probs, 1e-15, 1 - 1e-15))
        return np.sum(losses * weights) / np.sum(weights)

    # Grid search for rough optimum
    alphas = np.linspace(alpha_bounds[0], alpha_bounds[1], n_grid_points)
    losses = [compute_loss_vectorized(alpha) for alpha in alphas]
    best_idx = np.argmin(losses)

    # Refine with bounded optimization around best grid point
    if best_idx == 0:
        refined_bounds = (alpha_bounds[0], alphas[1])
    elif best_idx == len(alphas) - 1:
        refined_bounds = (alphas[-2], alpha_bounds[1])
    else:
        refined_bounds = (alphas[best_idx - 1], alphas[best_idx + 1])

    with log_timing(logger, "refined alpha optimization"):
        result = minimize_scalar(
            compute_loss_vectorized, bounds=refined_bounds, method="bounded"
        )

    optimal_alpha = result.x
    logger.info(f"Optimal alpha (sampled): {optimal_alpha:.6f}")

    return optimal_alpha


def compute_probabilities(
    rating_a: np.ndarray,
    rating_b: np.ndarray,
    alpha: float,
    score_transform: str = "bradley_terry",
    global_prior: float | None = None,
) -> np.ndarray:
    """Vectorized probability computation."""
    if score_transform == "bradley_terry":
        # Bradley-Terry: P(A > B) = s_A^alpha / (s_A^alpha + s_B^alpha)
        a_alpha = np.power(rating_a, alpha)
        b_alpha = np.power(rating_b, alpha)
        return a_alpha / (a_alpha + b_alpha)
    elif score_transform == "logistic":
        z = alpha * (rating_a - rating_b)
        return 1.0 / (1.0 + np.exp(-z))
    else:
        return rating_a


def compute_weights(
    probs: np.ndarray,
    scheme: str = "entropy",
    weight_threshold: float = 0.1,
    weight_exp_base: float = 2.0,
) -> np.ndarray:
    """Compute match weights based on scheme."""
    if scheme == "none":
        return np.ones_like(probs)
    elif scheme == "var_inv":
        variance = probs * (1 - probs)
        return 1.0 / (variance + 1e-10)
    elif scheme == "entropy":
        entropy = -probs * np.log(probs + 1e-10) - (1 - probs) * np.log(
            1 - probs + 1e-10
        )
        return entropy / np.log(2)
    elif scheme == "entropy_sqrt":
        entropy = -probs * np.log(probs + 1e-10) - (1 - probs) * np.log(
            1 - probs + 1e-10
        )
        return np.sqrt(entropy / np.log(2))
    elif scheme == "entropy_squared":
        entropy = -probs * np.log(probs + 1e-10) - (1 - probs) * np.log(
            1 - probs + 1e-10
        )
        return (entropy / np.log(2)) ** 2
    elif scheme == "entropy_exp":
        entropy = -probs * np.log(probs + 1e-10) - (1 - probs) * np.log(
            1 - probs + 1e-10
        )
        return np.power(weight_exp_base, entropy / np.log(2))
    elif scheme == "threshold":
        distance = np.abs(probs - 0.5)
        return (distance >= weight_threshold).astype(float)
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")


def fit_alpha_parameter_optimized(
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
    sample_size: int = 500,
    n_grid_points: int = 7,
) -> float:
    """
    Optimized alpha parameter fitting with minimal overhead.

    Key optimizations over v1:
    1. Smaller default sample size (500 vs 1000)
    2. Fewer grid points (7 vs 20)
    3. Skip team aggregation - work directly with player ratings
    4. Vectorized team rating computation
    5. Golden section search instead of scipy minimize
    6. Skip filtering if threshold is low

    Returns
    -------
    float
        Optimal alpha value
    """
    logger = get_logger(__name__)
    prior = global_prior if global_prior is not None else 0.05

    # Ultra-fast sampling
    n_matches = train_matches_df.height
    if n_matches > sample_size:
        # Use systematic sampling for better coverage
        step = n_matches // sample_size
        # Simple random sampling
        train_sample = train_matches_df.sample(n=sample_size, seed=42)
    else:
        train_sample = train_matches_df

    # Skip filtering if threshold is permissive
    if apply_threshold_filter and threshold > 0.5 and players_df is not None:
        # Quick approximate filter - just check if teams exist in rating map
        has_ratings = train_sample.with_columns(
            [
                pl.col(winner_id_col)
                .is_in(list(rating_map.keys()))
                .alias("w_rated"),
                pl.col(loser_id_col)
                .is_in(list(rating_map.keys()))
                .alias("l_rated"),
            ]
        )
        train_sample = has_ratings.filter(
            pl.col("w_rated") & pl.col("l_rated")
        ).drop(["w_rated", "l_rated"])

    # Get team player lists if available, else assume team_id = player_id
    if (
        "winner_members" in train_sample.columns
        and "loser_members" in train_sample.columns
    ):
        # Fast vectorized rating aggregation
        def get_team_ratings_vectorized(members_col):
            return train_sample[members_col].map_elements(
                lambda members: np.mean(
                    [rating_map.get(p, prior) for p in members]
                )
                if members is not None and len(members) > 0
                else prior,
                return_dtype=pl.Float64,
            )

        winner_ratings = get_team_ratings_vectorized(
            "winner_members"
        ).to_numpy()
        loser_ratings = get_team_ratings_vectorized("loser_members").to_numpy()
    else:
        # Direct lookup - much faster
        winner_ratings = np.array(
            [rating_map.get(tid, prior) for tid in train_sample[winner_id_col]]
        )
        loser_ratings = np.array(
            [rating_map.get(tid, prior) for tid in train_sample[loser_id_col]]
        )

    # Pre-compute for entropy weighting if needed
    if scheme != "none":
        # Use alpha=1 base probabilities
        if score_transform == "bradley_terry":
            base_probs = winner_ratings / (winner_ratings + loser_ratings)
        else:
            base_probs = 1.0 / (1.0 + np.exp(-(winner_ratings - loser_ratings)))

        # Simplified weight computation
        if scheme == "entropy":
            eps = 1e-10
            weights = -(
                base_probs * np.log(base_probs + eps)
                + (1 - base_probs) * np.log(1 - base_probs + eps)
            )
        elif scheme == "entropy_squared":
            eps = 1e-10
            entropy = -(
                base_probs * np.log(base_probs + eps)
                + (1 - base_probs) * np.log(1 - base_probs + eps)
            )
            weights = entropy * entropy
        else:
            weights = np.ones(len(winner_ratings))

        # Normalize weights
        weights = weights / np.sum(weights)
    else:
        weights = np.ones(len(winner_ratings)) / len(winner_ratings)

    # Ultra-fast loss function
    def loss(alpha: float) -> float:
        if score_transform == "bradley_terry":
            if alpha == 1.0:
                probs = winner_ratings / (winner_ratings + loser_ratings)
            else:
                wa = np.power(winner_ratings, alpha)
                la = np.power(loser_ratings, alpha)
                probs = wa / (wa + la)
        else:
            z = alpha * (winner_ratings - loser_ratings)
            probs = 1.0 / (1.0 + np.exp(-z))

        # Weighted cross-entropy
        return -np.sum(weights * np.log(np.maximum(probs, 1e-15)))

    # Coarse grid search
    alphas = np.linspace(alpha_bounds[0], alpha_bounds[1], n_grid_points)
    losses = [loss(a) for a in alphas]
    best_idx = np.argmin(losses)

    # Golden section search for refinement
    if best_idx == 0:
        a, b = alpha_bounds[0], alphas[1]
    elif best_idx == len(alphas) - 1:
        a, b = alphas[-2], alpha_bounds[1]
    else:
        a, b = alphas[best_idx - 1], alphas[best_idx + 1]

    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    resphi = 2 - phi

    # Initial points
    tol = 1e-5
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1 = loss(x1)
    f2 = loss(x2)

    # Golden section iterations (usually converges in 10-15 iterations)
    for _ in range(15):
        if abs(b - a) < tol:
            break

        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + resphi * (b - a)
            f1 = loss(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - resphi * (b - a)
            f2 = loss(x2)

    optimal_alpha = (a + b) / 2
    logger.info(f"Optimal alpha: {optimal_alpha:.6f}")

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
