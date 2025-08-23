"""
Tournament-specific evaluation metrics.

This module extends the evaluation framework with metrics specifically
designed for tournament seeding and prediction evaluation.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import log_loss as sklearn_log_loss


def ndcg_at_k(
    predicted_order: list[int], true_order: list[int], k: int | None = None
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at k.

    NDCG emphasizes getting the top positions correct, which is crucial
    for tournament seeding where top seeds matter most.

    Parameters
    ----------
    predicted_order : list[int]
        Predicted ranking order (team IDs in order)
    true_order : list[int]
        True ranking order based on final placements
    k : int | None
        Cutoff position (None for all positions)

    Returns
    -------
    float
        NDCG@k score (0 to 1, higher is better)
    """
    if k is None:
        k = len(predicted_order)

    # Create relevance scores (higher placement = higher relevance)
    n = len(true_order)
    relevance = {team: n - i for i, team in enumerate(true_order)}

    # Calculate DCG for predicted order
    dcg = 0.0
    for i in range(min(k, len(predicted_order))):
        if i < len(predicted_order):
            team = predicted_order[i]
            rel = relevance.get(team, 0)
            dcg += rel / np.log2(i + 2)  # i+2 because positions are 0-indexed

    # Calculate ideal DCG (perfect ordering)
    ideal_dcg = 0.0
    sorted_relevances = sorted(relevance.values(), reverse=True)
    for i in range(min(k, len(sorted_relevances))):
        ideal_dcg += sorted_relevances[i] / np.log2(i + 2)

    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def weighted_spearman_correlation(
    predicted_seeds: list[int],
    final_placements: list[int],
    weights: list[float] | None = None,
) -> float:
    """
    Calculate weighted Spearman correlation emphasizing top positions.

    Parameters
    ----------
    predicted_seeds : list[int]
        Predicted seed order
    final_placements : list[int]
        Actual final placements for those teams
    weights : list[float] | None
        Weights for each position (default: exponential decay)

    Returns
    -------
    float
        Weighted Spearman correlation (-1 to 1)
    """
    if len(predicted_seeds) != len(final_placements):
        raise ValueError("Seeds and placements must have same length")

    if weights is None:
        # Default exponential decay weights emphasizing top positions
        weights = [1.0 / (1.0 + i) for i in range(len(predicted_seeds))]

    # Convert to ranks
    x_ranks = stats.rankdata(predicted_seeds)
    y_ranks = stats.rankdata(final_placements)

    # Calculate weighted correlation on ranks
    weights = np.array(weights)
    x_mean = np.average(x_ranks, weights=weights)
    y_mean = np.average(y_ranks, weights=weights)

    cov = np.average((x_ranks - x_mean) * (y_ranks - y_mean), weights=weights)
    x_std = np.sqrt(np.average((x_ranks - x_mean) ** 2, weights=weights))
    y_std = np.sqrt(np.average((y_ranks - y_mean) ** 2, weights=weights))

    return cov / (x_std * y_std) if x_std > 0 and y_std > 0 else 0.0


def mean_absolute_seed_error(
    predicted_seeds: dict[int, int],
    final_placements: dict[int, int],
    top_k: int | None = None,
) -> float:
    """
    Calculate mean absolute error between seeds and final placements.

    Parameters
    ----------
    predicted_seeds : dict[int, int]
        Mapping of team_id to predicted seed
    final_placements : dict[int, int]
        Mapping of team_id to final placement
    top_k : int | None
        Only consider top k teams (useful for focusing on top seeds)

    Returns
    -------
    float
        Mean absolute error (lower is better)
    """
    errors = []

    for team_id, seed in predicted_seeds.items():
        if team_id in final_placements:
            if top_k is None or seed <= top_k:
                error = abs(seed - final_placements[team_id])
                errors.append(error)

    return np.mean(errors) if errors else 0.0


def calibration_error(
    predictions: list[float], outcomes: list[bool], n_bins: int = 10
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Calculate expected calibration error for probability predictions.

    Parameters
    ----------
    predictions : list[float]
        Predicted probabilities
    outcomes : list[bool]
        Actual outcomes
    n_bins : int
        Number of bins for calibration

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray]
        (ECE, bin_accuracies, bin_confidences)
    """
    predictions = np.array(predictions)
    outcomes = np.array(outcomes, dtype=float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        mask = (predictions >= bin_boundaries[i]) & (
            predictions < bin_boundaries[i + 1]
        )

        if i == n_bins - 1:  # Include 1.0 in last bin
            mask = (predictions >= bin_boundaries[i]) & (
                predictions <= bin_boundaries[i + 1]
            )

        if mask.sum() > 0:
            bin_accuracy = outcomes[mask].mean()
            bin_confidence = predictions[mask].mean()
            bin_count = mask.sum()

            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
            bin_counts.append(bin_count)
        else:
            bin_accuracies.append(0)
            bin_confidences.append(bin_boundaries[i] + 0.05)
            bin_counts.append(0)

    # Calculate ECE (weighted by bin counts)
    total_count = sum(bin_counts)
    if total_count > 0:
        ece = (
            sum(
                count * abs(acc - conf)
                for acc, conf, count in zip(
                    bin_accuracies, bin_confidences, bin_counts
                )
            )
            / total_count
        )
    else:
        ece = 0.0

    return ece, np.array(bin_accuracies), np.array(bin_confidences)


def upset_rate_analysis(
    predictions: list[float],
    outcomes: list[bool],
    prob_buckets: list[tuple[float, float]] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Analyze upset rates by probability bucket.

    Parameters
    ----------
    predictions : list[float]
        Predicted probabilities
    outcomes : list[bool]
        Actual outcomes (True if prediction was correct)
    prob_buckets : list[tuple[float, float]] | None
        List of (min_prob, max_prob) tuples for buckets

    Returns
    -------
    dict[str, dict[str, float]]
        Upset analysis by probability bucket
    """
    if prob_buckets is None:
        prob_buckets = [
            (0.5, 0.55),
            (0.55, 0.65),
            (0.65, 0.75),
            (0.75, 0.85),
            (0.85, 1.0),
        ]

    upset_rates = {}

    for min_prob, max_prob in prob_buckets:
        bucket_predictions = []
        bucket_outcomes = []

        for pred, outcome in zip(predictions, outcomes):
            if min_prob <= pred < max_prob or (max_prob == 1.0 and pred == 1.0):
                bucket_predictions.append(pred)
                bucket_outcomes.append(outcome)

        if bucket_predictions:
            # Calculate actual win rate vs expected
            actual_win_rate = np.mean(bucket_outcomes)
            expected_win_rate = np.mean(bucket_predictions)

            bucket_name = f"{min_prob:.0%}-{max_prob:.0%}"
            upset_rates[bucket_name] = {
                "actual_win_rate": actual_win_rate,
                "expected_win_rate": expected_win_rate,
                "calibration_diff": actual_win_rate - expected_win_rate,
                "n_matches": len(bucket_outcomes),
                "upset_rate": 1
                - actual_win_rate,  # Rate at which favorites lose
            }

    return upset_rates


def pairwise_agreement(order1: list[int], order2: list[int]) -> float:
    """
    Calculate pairwise agreement between two orderings.

    Useful for comparing model seeds with manual seeds.

    Parameters
    ----------
    order1 : list[int]
        First ordering (e.g., model seeds)
    order2 : list[int]
        Second ordering (e.g., manual seeds)

    Returns
    -------
    float
        Percentage of pairwise comparisons that agree (0 to 1)
    """
    common_items = set(order1) & set(order2)

    if len(common_items) < 2:
        return 1.0  # Trivial agreement

    agree = 0
    total = 0

    # Check all pairs
    common_list = list(common_items)
    for i in range(len(common_list)):
        for j in range(i + 1, len(common_list)):
            item1 = common_list[i]
            item2 = common_list[j]

            # Check if relative order is same
            idx1_1 = order1.index(item1)
            idx1_2 = order1.index(item2)
            idx2_1 = order2.index(item1)
            idx2_2 = order2.index(item2)

            if (idx1_1 < idx1_2) == (idx2_1 < idx2_2):
                agree += 1
            total += 1

    return agree / total if total > 0 else 0.0


def kendall_tau_distance(
    order1: list[int], order2: list[int]
) -> tuple[float, int]:
    """
    Calculate Kendall tau distance between two orderings.

    Parameters
    ----------
    order1 : list[int]
        First ordering
    order2 : list[int]
        Second ordering

    Returns
    -------
    tuple[float, int]
        (normalized_distance, number_of_discordant_pairs)
    """
    common_items = list(set(order1) & set(order2))
    n = len(common_items)

    if n < 2:
        return 0.0, 0

    discordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            item_i = common_items[i]
            item_j = common_items[j]

            # Get positions in both orderings
            pos1_i = order1.index(item_i)
            pos1_j = order1.index(item_j)
            pos2_i = order2.index(item_i)
            pos2_j = order2.index(item_j)

            # Check if pair is discordant
            if (pos1_i < pos1_j) != (pos2_i < pos2_j):
                discordant += 1

    max_discordant = n * (n - 1) // 2
    normalized_distance = (
        discordant / max_discordant if max_discordant > 0 else 0.0
    )

    return normalized_distance, discordant


def mcnemar_test(
    model_correct: list[bool], manual_correct: list[bool]
) -> tuple[float, float]:
    """
    McNemar's test for comparing two prediction methods.

    Tests if model predictions are significantly different from manual.

    Parameters
    ----------
    model_correct : list[bool]
        List of whether model prediction was correct
    manual_correct : list[bool]
        List of whether manual prediction was correct

    Returns
    -------
    tuple[float, float]
        (statistic, p-value)
    """
    # Build contingency table
    both_correct = sum(
        1 for m, h in zip(model_correct, manual_correct) if m and h
    )
    model_only = sum(
        1 for m, h in zip(model_correct, manual_correct) if m and not h
    )
    manual_only = sum(
        1 for m, h in zip(model_correct, manual_correct) if not m and h
    )
    both_wrong = sum(
        1 for m, h in zip(model_correct, manual_correct) if not m and not h
    )

    # McNemar's statistic
    n = model_only + manual_only
    if n == 0:
        return 0.0, 1.0

    # Use continuity correction
    statistic = (abs(model_only - manual_only) - 1) ** 2 / n if n > 1 else 0

    # Chi-square test with 1 degree of freedom
    from scipy.stats import chi2

    p_value = 1 - chi2.cdf(statistic, df=1)

    return statistic, p_value


def tournament_prediction_summary(
    predicted_seeds: dict[int, int],
    final_placements: dict[int, int],
    match_predictions: list[float] | None = None,
    match_outcomes: list[bool] | None = None,
    manual_seeds: dict[int, int] | None = None,
) -> dict[str, float]:
    """
    Generate comprehensive tournament prediction summary.

    Parameters
    ----------
    predicted_seeds : dict[int, int]
        Model's predicted seeds
    final_placements : dict[int, int]
        Actual final placements
    match_predictions : list[float] | None
        Match-level win probabilities
    match_outcomes : list[bool] | None
        Actual match outcomes
    manual_seeds : dict[int, int] | None
        Manual seeds for comparison

    Returns
    -------
    dict[str, float]
        Comprehensive metrics summary
    """
    summary = {}

    # Seeding metrics
    common_teams = set(predicted_seeds.keys()) & set(final_placements.keys())
    if common_teams:
        pred_order = sorted(common_teams, key=lambda t: predicted_seeds[t])
        final_order = sorted(common_teams, key=lambda t: final_placements[t])

        # Calculate various seeding metrics
        summary["ndcg_at_4"] = ndcg_at_k(pred_order, final_order, k=4)
        summary["ndcg_at_8"] = ndcg_at_k(pred_order, final_order, k=8)
        summary["ndcg_full"] = ndcg_at_k(pred_order, final_order)

        # Spearman correlation
        pred_values = [predicted_seeds[t] for t in common_teams]
        final_values = [final_placements[t] for t in common_teams]
        corr, p_val = stats.spearmanr(pred_values, final_values)
        summary["spearman_correlation"] = corr
        summary["spearman_pvalue"] = p_val

        # Mean absolute error
        summary["mae_all"] = mean_absolute_seed_error(
            predicted_seeds, final_placements
        )
        summary["mae_top4"] = mean_absolute_seed_error(
            predicted_seeds, final_placements, top_k=4
        )
        summary["mae_top8"] = mean_absolute_seed_error(
            predicted_seeds, final_placements, top_k=8
        )

    # Match prediction metrics
    if match_predictions is not None and match_outcomes is not None:
        summary["match_accuracy"] = np.mean(
            [(p > 0.5) == o for p, o in zip(match_predictions, match_outcomes)]
        )
        summary["match_log_loss"] = sklearn_log_loss(
            match_outcomes, match_predictions
        )
        summary["match_brier_score"] = np.mean(
            [(p - o) ** 2 for p, o in zip(match_predictions, match_outcomes)]
        )

        # Calibration
        ece, _, _ = calibration_error(match_predictions, match_outcomes)
        summary["expected_calibration_error"] = ece

    # Comparison with manual seeds
    if manual_seeds is not None:
        manual_common = set(predicted_seeds.keys()) & set(manual_seeds.keys())
        if manual_common:
            model_order = sorted(
                manual_common, key=lambda t: predicted_seeds[t]
            )
            manual_order = sorted(manual_common, key=lambda t: manual_seeds[t])

            summary["manual_pairwise_agreement"] = pairwise_agreement(
                model_order, manual_order
            )

            # Kendall tau distance
            tau_dist, _ = kendall_tau_distance(model_order, manual_order)
            summary["manual_kendall_distance"] = tau_dist

    return summary


def confidence_interval_coverage(
    predictions: list[tuple[float, float, float]],
    outcomes: list[bool],
    confidence_level: float = 0.95,
) -> dict[str, float]:
    """
    Evaluate confidence interval coverage for probabilistic predictions.

    Parameters
    ----------
    predictions : list[tuple[float, float, float]
        List of (point_estimate, lower_bound, upper_bound)
    outcomes : list[bool]
        Actual binary outcomes
    confidence_level : float
        Expected coverage level

    Returns
    -------
    dict[str, float]
        Coverage statistics
    """
    if not predictions:
        return {}

    covered = 0
    interval_widths = []

    for (point, lower, upper), outcome in zip(predictions, outcomes):
        outcome_val = 1.0 if outcome else 0.0

        # Check if outcome falls within interval
        if lower <= outcome_val <= upper:
            covered += 1

        interval_widths.append(upper - lower)

    n = len(predictions)
    empirical_coverage = covered / n

    # Calculate coverage error
    coverage_error = abs(empirical_coverage - confidence_level)

    # Standard error for coverage
    se_coverage = np.sqrt(empirical_coverage * (1 - empirical_coverage) / n)

    return {
        "empirical_coverage": empirical_coverage,
        "expected_coverage": confidence_level,
        "coverage_error": coverage_error,
        "coverage_se": se_coverage,
        "mean_interval_width": np.mean(interval_widths),
        "std_interval_width": np.std(interval_widths),
        "n_predictions": n,
    }
