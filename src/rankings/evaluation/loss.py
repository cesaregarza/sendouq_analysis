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
    rating_a = max(float(rating_a) if rating_a is not None else prior, prior)
    rating_b = max(float(rating_b) if rating_b is not None else prior, prior)

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
        Weighting scheme: "none", "var_inv", or "entropy"
    winner_id_col : str
        Column name for winner ID
    loser_id_col : str
        Column name for loser ID
    return_predictions : bool
        If True, include prediction probabilities in return dict
    global_prior : float, optional
        Prior rating for unknown players (defaults to 0.05)

    Returns
    -------
    Tuple[float, Dict[str, any]]
        (tournament_loss, metrics_dict with optional predictions)
    """
    logger = get_logger(__name__)
    logger.debug(
        f"Computing tournament loss for {matches_df.height} matches with alpha={alpha}"
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

    # Compute loss using new unified weighting logic
    base = -np.log(np.clip(predictions, 1e-15, 1.0 - 1e-15))  # Bernoulli y=1
    if scheme == "var_inv":
        w = 1.0 / np.maximum(predictions * (1 - predictions), 1e-10)
    elif scheme == "entropy":
        w = 2.0 * np.abs(predictions - 0.5)
    else:
        w = np.ones_like(base)

    final_loss = np.average(base, weights=w)

    # Compute additional metrics
    metrics = {
        "n_matches": len(predictions),
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
                pl.count().alias("count"),
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
        Weighting scheme: "none", "var_inv", or "entropy"
    global_prior : float, optional
        Prior rating for unknown players (defaults to 0.05)
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
        loss, _ = compute_tournament_loss(
            train_matches_df,
            rating_map,
            alpha=alpha,
            score_transform=score_transform,
            scheme=scheme,
            winner_id_col=winner_id_col,
            loser_id_col=loser_id_col,
            return_predictions=False,
            global_prior=global_prior,
        )
        return loss * len(train_matches_df)

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
