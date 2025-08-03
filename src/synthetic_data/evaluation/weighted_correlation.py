"""
Weighted correlation metrics for ranking evaluation.

Implements weighted Spearman correlation where rank differences at the top
are weighted more heavily than differences at the bottom.
"""

from typing import Optional

import numpy as np
from scipy import stats


def weighted_spearman(
    true_ranks: list[int],
    predicted_ranks: list[int],
    weight_type: str = "exponential",
    alpha: float = 0.1,
) -> float:
    """
    Calculate weighted Spearman correlation.

    Parameters
    ----------
    true_ranks : list[int]
        True rankings (1 = best)
    predicted_ranks : list[int]
        Predicted rankings (1 = best)
    weight_type : str
        Type of weighting: "exponential", "logarithmic", "hyperbolic"
    alpha : float
        Decay parameter for weighting

    Returns
    -------
    float
        Weighted Spearman correlation
    """
    if len(true_ranks) != len(predicted_ranks):
        raise ValueError("Rank lists must have same length")

    n = len(true_ranks)

    # Calculate rank differences
    rank_diffs = np.array(
        [pred - true for true, pred in zip(true_ranks, predicted_ranks)]
    )

    # Calculate weights based on true rank position
    if weight_type == "exponential":
        # Exponential decay: top ranks get much higher weight
        weights = np.array([np.exp(-alpha * (r - 1)) for r in true_ranks])
    elif weight_type == "logarithmic":
        # Logarithmic decay: more gradual weight decrease
        weights = np.array([1.0 / np.log(r + 1) for r in true_ranks])
    elif weight_type == "hyperbolic":
        # Hyperbolic decay: 1/rank weighting
        weights = np.array([1.0 / r for r in true_ranks])
    else:
        raise ValueError(f"Unknown weight type: {weight_type}")

    # Normalize weights
    weights = weights / weights.sum()

    # Calculate weighted squared differences
    weighted_sq_diffs = weights * (rank_diffs**2)

    # Calculate weighted Spearman correlation
    # rho = 1 - (6 * sum(weighted d^2)) / normalizing_factor
    # For weighted case, we need to adjust the normalizing factor

    # Standard Spearman denominator is n(n^2 - 1)
    # For weighted case, we approximate by scaling with effective sample size
    effective_n = 1.0 / np.sum(weights**2)  # Effective sample size

    # Weighted correlation
    weighted_rho = 1.0 - (6.0 * np.sum(weighted_sq_diffs)) / (
        effective_n * (effective_n**2 - 1)
    )

    # Ensure correlation is in valid range
    weighted_rho = np.clip(weighted_rho, -1.0, 1.0)

    return weighted_rho


def top_k_weighted_accuracy(
    true_ranks: list[int],
    predicted_ranks: list[int],
    k_values: list[int] = [10, 20, 50, 100],
) -> dict:
    """
    Calculate top-k accuracy with exponential weighting.

    Being in top-k when you should be is weighted by how high you should rank.
    """
    results = {}

    # Create rank mappings
    true_rank_map = {i: rank for i, rank in enumerate(true_ranks)}
    pred_rank_map = {i: rank for i, rank in enumerate(predicted_ranks)}

    for k in k_values:
        if k > len(true_ranks):
            continue

        # Find players who should be in top k
        true_top_k_indices = [
            i for i, rank in enumerate(true_ranks) if rank <= k
        ]

        # Calculate weighted accuracy
        total_weight = 0
        correct_weight = 0

        for idx in true_top_k_indices:
            true_rank = true_ranks[idx]
            pred_rank = predicted_ranks[idx]

            # Weight by inverse of true rank (top ranks more important)
            weight = 1.0 / true_rank
            total_weight += weight

            # Credit if predicted in top k
            if pred_rank <= k:
                correct_weight += weight

        results[f"top_{k}_weighted"] = (
            correct_weight / total_weight if total_weight > 0 else 0
        )

    return results


def rank_difference_distribution(
    true_ranks: list[int],
    predicted_ranks: list[int],
    percentiles: list[int] = [10, 25, 50, 75, 90, 95, 99],
) -> dict:
    """
    Analyze distribution of rank differences, especially at the top.
    """
    rank_diffs = np.abs(np.array(predicted_ranks) - np.array(true_ranks))

    results = {
        "mean_abs_diff": np.mean(rank_diffs),
        "median_abs_diff": np.median(rank_diffs),
        "max_abs_diff": np.max(rank_diffs),
    }

    # Percentiles
    for p in percentiles:
        results[f"p{p}_abs_diff"] = np.percentile(rank_diffs, p)

    # Top player analysis
    top_10_mask = np.array(true_ranks) <= 10
    if np.any(top_10_mask):
        results["top_10_mean_diff"] = np.mean(rank_diffs[top_10_mask])
        results["top_10_max_diff"] = np.max(rank_diffs[top_10_mask])

    top_50_mask = np.array(true_ranks) <= 50
    if np.any(top_50_mask):
        results["top_50_mean_diff"] = np.mean(rank_diffs[top_50_mask])

    return results
