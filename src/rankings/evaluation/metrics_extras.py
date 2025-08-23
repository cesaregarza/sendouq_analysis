"""
Extra evaluation metrics for the rankings system.

This module implements the metrics suite described in plan.md:
- Concordance (c_stat): Pure discrimination
- Skill Score: Discrimination vs permuted baseline
- Upset O/E: Calibration via expected vs observed upsets
- Confidence-filtered accuracy: Quality of high-confidence calls
- Spearman correlation: Tournament-level ranking order
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from rankings.core.logging import get_logger
from rankings.evaluation.loss import compute_tournament_loss


def concordance(
    matches_df: pl.DataFrame,
    rating_map: dict[int, float],
    *,
    winner_col: str = "winner_team_id",
    loser_col: str = "loser_team_id",
) -> float:
    """
    Compute concordance (c-statistic) - the fraction of matches where
    the higher-rated team won.

    This is a pure discrimination measure that answers "how often does
    the higher rating win?"

    Parameters
    ----------
    matches_df : pl.DataFrame
        DataFrame containing match results
    rating_map : dict[int, float]
        Mapping from team/player ID to rating
    winner_col : str
        Column name for winner ID
    loser_col : str
        Column name for loser ID

    Returns
    -------
    float
        Concordance value in [0, 1] where 0.5 = random, 1.0 = perfect

    Examples
    --------
    >>> import polars as pl
    >>> matches = pl.DataFrame({
    ...     'winner_team_id': [1, 2],
    ...     'loser_team_id': [2, 3]
    ... })
    >>> ratings = {1: 1600, 2: 1500, 3: 1400}
    >>> concordance(matches, ratings)
    1.0
    """
    if matches_df.height == 0:
        return np.nan

    match_data = matches_df.select(
        [
            pl.col(winner_col).alias("winner"),
            pl.col(loser_col).alias("loser"),
        ]
    )

    correct = 0
    total = 0

    for row in match_data.iter_rows(named=True):
        winner_id = row["winner"]
        loser_id = row["loser"]

        r_winner = rating_map.get(winner_id, 0.0)
        r_loser = rating_map.get(loser_id, 0.0)

        if r_winner > r_loser:
            correct += 1
        total += 1

    return float(correct / total) if total > 0 else np.nan


def skill_score(
    matches_df: pl.DataFrame,
    rating_map: dict[int, float],
    *,
    alpha: float = 1.0,
    score_transform: str = "bradley_terry",
    scheme: str = "none",
    winner_col: str = "winner_team_id",
    loser_col: str = "loser_team_id",
    n_permutations: int = 10,
) -> float:
    """
    Compute skill score: 1 - (model_loss / permuted_baseline_loss).

    This measures discrimination normalized by noise level. A random baseline
    is created by permuting ratings across tournaments.

    Parameters
    ----------
    matches_df : pl.DataFrame
        DataFrame containing match results
    rating_map : dict[int, float]
        Mapping from team/player ID to rating
    alpha : float
        Temperature parameter
    score_transform : str
        Transform to apply: "bradley_terry", "logistic", or "identity"
    scheme : str
        Weighting scheme for loss computation
    winner_col : str
        Column name for winner ID
    loser_col : str
        Column name for loser ID
    n_permutations : int
        Number of permutations for baseline

    Returns
    -------
    float
        Skill score in [0, 1] where 0 = no skill, 1 = perfect skill

    Examples
    --------
    >>> import polars as pl
    >>> matches = pl.DataFrame({
    ...     'winner_team_id': [1, 2, 1],
    ...     'loser_team_id': [2, 3, 3]
    ... })
    >>> ratings = {1: 1600, 2: 1500, 3: 1400}
    >>> skill_score(matches, ratings)  # doctest: +SKIP
    0.23
    """
    if matches_df.height == 0:
        return np.nan

    # Get model loss
    model_loss, _ = compute_tournament_loss(
        matches_df,
        rating_map,
        alpha=alpha,
        score_transform=score_transform,
        scheme=scheme,
        winner_id_col=winner_col,
        loser_id_col=loser_col,
    )

    # Get baseline loss from permuted ratings
    team_ids = list(rating_map.keys())
    rating_values = list(rating_map.values())

    permuted_losses = []
    for _ in range(n_permutations):
        # Permute ratings randomly
        permuted_values = np.random.permutation(rating_values)
        permuted_map = dict(zip(team_ids, permuted_values))

        perm_loss, _ = compute_tournament_loss(
            matches_df,
            permuted_map,
            alpha=alpha,
            score_transform=score_transform,
            scheme=scheme,
            winner_id_col=winner_col,
            loser_id_col=loser_col,
        )
        permuted_losses.append(perm_loss)

    baseline_loss = np.mean(permuted_losses)

    if baseline_loss <= 0 or not np.isfinite(baseline_loss):
        return np.nan

    return float(1.0 - model_loss / baseline_loss)


def upset_oe(predictions: np.ndarray) -> float:
    """
    Compute upset observed/expected ratio.

    An "upset" is when the predicted probability is < 0.5 (underdog wins).
    Good calibration means O/E ≈ 1.0.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted win probabilities for the observed winners

    Returns
    -------
    float
        Observed/Expected upset ratio where 1.0 = perfect calibration

    Examples
    --------
    >>> import numpy as np
    >>> # Perfect calibration: 50% of 0.3 predictions should be upsets
    >>> probs = np.array([0.3, 0.3, 0.7, 0.7])
    >>> upset_oe(probs)  # Should be close to 1.0
    1.0
    """
    if len(predictions) == 0:
        return np.nan

    # Observed upsets: fraction where p < 0.5
    observed = np.mean(predictions < 0.5)

    # Expected upsets: mean of (1-p) for all predictions
    expected = np.mean(1.0 - predictions)

    if expected == 0:
        return np.nan

    return float(observed / expected)


def accuracy_threshold(
    predictions: np.ndarray, threshold: float = 0.65
) -> float:
    """
    Compute accuracy for high-confidence predictions only.

    This measures the quality of predictions where we're confident
    (probability >= threshold).

    Parameters
    ----------
    predictions : np.ndarray
        Predicted win probabilities for the observed winners
    threshold : float
        Minimum confidence threshold

    Returns
    -------
    float
        Accuracy for predictions >= threshold, or NaN if none exist

    Examples
    --------
    >>> import numpy as np
    >>> probs = np.array([0.8, 0.7, 0.4, 0.9])  # All winners
    >>> accuracy_threshold(probs, 0.65)  # Only 0.8, 0.7, 0.9 count
    1.0
    """
    if len(predictions) == 0:
        return np.nan

    high_conf_mask = predictions >= threshold

    if not np.any(high_conf_mask):
        return np.nan

    high_conf_preds = predictions[high_conf_mask]
    # Since these are probabilities for observed winners, accuracy = p > 0.5
    accuracy = np.mean(high_conf_preds > 0.5)

    return float(accuracy)


def placement_spearman(
    pre_ratings: dict[int, float], placements: dict[int, int]
) -> float:
    """
    Compute Spearman correlation between pre-tournament ratings and final placements.

    Higher ratings should correspond to better (lower) placements, so we expect
    negative correlation. We return -correlation so higher = better.

    Parameters
    ----------
    pre_ratings : dict[int, float]
        Pre-tournament ratings by team/player ID
    placements : dict[int, int]
        Final tournament placements by team/player ID (1=best, 2=second, etc.)

    Returns
    -------
    float
        Negative Spearman correlation (higher = better ranking order)

    Examples
    --------
    >>> pre_rats = {1: 1600, 2: 1500, 3: 1400}
    >>> places = {1: 1, 2: 2, 3: 3}  # Perfect order
    >>> placement_spearman(pre_rats, places)
    1.0
    """
    common_ids = set(pre_ratings.keys()) & set(placements.keys())

    if len(common_ids) < 2:
        return np.nan

    ratings = [pre_ratings[id_] for id_ in common_ids]
    places = [placements[id_] for id_ in common_ids]

    # Compute correlation
    corr, _ = spearmanr(ratings, places)

    if not np.isfinite(corr):
        return np.nan

    # Return negative so that higher = better
    return float(-corr)


def alpha_std(alphas: list[float]) -> float:
    """
    Compute standard deviation of alpha values across CV splits.

    This measures stability - lower values indicate more consistent
    optimal alpha across different data splits.

    Parameters
    ----------
    alphas : list[float]
        Alpha values from different CV splits

    Returns
    -------
    float
        Standard deviation of alpha values

    Examples
    --------
    >>> alphas = [1.0, 1.1, 0.9, 1.05]
    >>> alpha_std(alphas)  # doctest: +SKIP
    0.08
    """
    if len(alphas) < 2:
        return np.nan

    return float(np.std(alphas))


def reliability_diagram(
    predictions: np.ndarray, n_buckets: int = 10
) -> pl.DataFrame:
    """
    Create reliability diagram data for calibration analysis.

    This returns bucketed expected vs observed win rates to assess
    how well predicted probabilities match actual outcomes.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted win probabilities for the observed winners
    n_buckets : int, default=10
        Number of probability buckets

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:
        - bucket_min: Lower bound of probability bucket
        - bucket_max: Upper bound of probability bucket
        - bucket_center: Center of probability bucket
        - count: Number of predictions in bucket
        - observed_rate: Actual win rate in bucket (should be ~1.0 since these are winners)
        - expected_rate: Mean predicted probability in bucket
        - calibration_error: |observed_rate - expected_rate|

    Examples
    --------
    >>> import numpy as np
    >>> preds = np.array([0.8, 0.7, 0.6, 0.9, 0.75])
    >>> diagram = reliability_diagram(preds, n_buckets=5)
    >>> print(diagram)  # doctest: +SKIP
    """
    if len(predictions) == 0:
        return pl.DataFrame(
            {
                "bucket_min": [],
                "bucket_max": [],
                "bucket_center": [],
                "count": [],
                "observed_rate": [],
                "expected_rate": [],
                "calibration_error": [],
            }
        )

    # Create probability buckets
    bins = np.linspace(0.0, 1.0, n_buckets + 1)
    bucket_indices = np.digitize(predictions, bins, right=False) - 1
    bucket_indices = np.clip(bucket_indices, 0, n_buckets - 1)

    # Calculate metrics for each bucket
    bucket_data = []
    for i in range(n_buckets):
        bucket_min = bins[i]
        bucket_max = bins[i + 1]
        bucket_center = (bucket_min + bucket_max) / 2

        # Find predictions in this bucket
        mask = bucket_indices == i
        bucket_preds = predictions[mask]

        if len(bucket_preds) == 0:
            count = 0
            observed_rate = np.nan
            expected_rate = np.nan
            calibration_error = np.nan
        else:
            count = len(bucket_preds)
            # Since all inputs are winners, observed rate is always 1.0
            observed_rate = 1.0
            expected_rate = np.mean(bucket_preds)
            calibration_error = abs(observed_rate - expected_rate)

        bucket_data.append(
            {
                "bucket_min": bucket_min,
                "bucket_max": bucket_max,
                "bucket_center": bucket_center,
                "count": count,
                "observed_rate": observed_rate,
                "expected_rate": expected_rate,
                "calibration_error": calibration_error,
            }
        )

    return pl.DataFrame(bucket_data)
