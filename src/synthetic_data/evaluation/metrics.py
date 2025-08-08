"""
Evaluation metrics for ranking quality assessment.

This module provides metrics to evaluate the quality of rankings,
including pairwise AUC for ranking quality and calibration metrics.
"""

from typing import Optional, Tuple

import numpy as np
import polars as pl
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def pairwise_auc(
    pred_score_a: np.ndarray,
    pred_score_b: np.ndarray,
    outcome_ab: np.ndarray,
) -> float:
    """
    Calculate pairwise AUC for ranking quality.

    Measures how well the predicted scores order match outcomes.

    Parameters
    ----------
    pred_score_a : np.ndarray
        Predicted scores for player/team A
    pred_score_b : np.ndarray
        Predicted scores for player/team B
    outcome_ab : np.ndarray
        Binary outcomes (1 if A beat B, 0 otherwise)

    Returns
    -------
    float
        AUC score (0.5 = random, 1.0 = perfect)
    """
    # Convert to numpy arrays if needed
    pred_score_a = np.asarray(pred_score_a)
    pred_score_b = np.asarray(pred_score_b)
    outcome_ab = np.asarray(outcome_ab)

    # Calculate which predictions were correct
    pred_a_wins = (pred_score_a > pred_score_b).astype(float)

    # Handle ties
    ties = pred_score_a == pred_score_b
    pred_a_wins[ties] = 0.5

    # Calculate concordance (fraction of correct predictions)
    correct = (pred_a_wins == 1) & (outcome_ab == 1)
    incorrect = (pred_a_wins == 0) & (outcome_ab == 0)
    ties_contribution = ties.sum() * 0.5

    total_pairs = len(outcome_ab)
    if total_pairs == 0:
        return 0.5

    auc = (correct.sum() + incorrect.sum() + ties_contribution) / total_pairs

    # Alternative calculation using score differences
    # This is more robust for continuous scores
    score_diff = pred_score_a - pred_score_b

    try:
        # Use sklearn's ROC AUC with score differences
        auc_sklearn = roc_auc_score(outcome_ab, score_diff)
        return auc_sklearn
    except (ValueError, TypeError):
        # Fall back to manual calculation if sklearn fails
        return auc


def calibrate_probabilities(
    scores: np.ndarray,
    outcomes: np.ndarray,
    method: str = "platt",
) -> Tuple[LogisticRegression, float]:
    """
    Calibrate ranking scores to probabilities.

    Uses Platt scaling (logistic regression) to map scores to probabilities.

    Parameters
    ----------
    scores : np.ndarray
        Raw ranking scores or score differences
    outcomes : np.ndarray
        Binary outcomes
    method : str
        Calibration method ("platt" for logistic regression)

    Returns
    -------
    tuple
        (calibration_model, calibration_score)
    """
    scores = np.asarray(scores).reshape(-1, 1)
    outcomes = np.asarray(outcomes)

    if method == "platt":
        # Platt scaling: fit logistic regression
        calibrator = LogisticRegression(solver="lbfgs", max_iter=100)
        calibrator.fit(scores, outcomes)

        # Evaluate calibration quality
        probs = calibrator.predict_proba(scores)[:, 1]

        # Calculate ECE (Expected Calibration Error)
        ece = expected_calibration_error(probs, outcomes)

        return calibrator, ece
    else:
        raise ValueError(f"Unknown calibration method: {method}")


def expected_calibration_error(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Calculate Expected Calibration Error (ECE).

    Measures how well predicted probabilities match actual frequencies.

    Parameters
    ----------
    probabilities : np.ndarray
        Predicted probabilities
    outcomes : np.ndarray
        Binary outcomes
    n_bins : int
        Number of bins for calibration

    Returns
    -------
    float
        ECE score (lower is better, 0 = perfect calibration)
    """
    probabilities = np.asarray(probabilities)
    outcomes = np.asarray(outcomes)

    # Get calibration curve
    fraction_positive, mean_predicted = calibration_curve(
        outcomes, probabilities, n_bins=n_bins, strategy="uniform"
    )

    # Calculate ECE
    # Weight by number of samples in each bin
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(probabilities)

    for i in range(len(fraction_positive)):
        # Find samples in this bin
        if i < len(fraction_positive) - 1:
            bin_mask = (probabilities >= bin_edges[i]) & (
                probabilities < bin_edges[i + 1]
            )
        else:
            bin_mask = (probabilities >= bin_edges[i]) & (
                probabilities <= bin_edges[i + 1]
            )

        n_in_bin = bin_mask.sum()
        if n_in_bin > 0:
            bin_weight = n_in_bin / total_samples
            bin_error = abs(fraction_positive[i] - mean_predicted[i])
            ece += bin_weight * bin_error

    return ece


def ranking_correlation(
    pred_rankings: np.ndarray,
    true_rankings: np.ndarray,
    method: str = "spearman",
) -> float:
    """
    Calculate correlation between predicted and true rankings.

    Parameters
    ----------
    pred_rankings : np.ndarray
        Predicted rankings
    true_rankings : np.ndarray
        True rankings
    method : str
        Correlation method ("spearman" or "kendall")

    Returns
    -------
    float
        Correlation coefficient
    """
    from scipy.stats import kendalltau, spearmanr

    pred_rankings = np.asarray(pred_rankings)
    true_rankings = np.asarray(true_rankings)

    if method == "spearman":
        corr, _ = spearmanr(pred_rankings, true_rankings)
    elif method == "kendall":
        corr, _ = kendalltau(pred_rankings, true_rankings)
    else:
        raise ValueError(f"Unknown correlation method: {method}")

    return corr


def top_k_accuracy(
    pred_rankings: np.ndarray,
    true_rankings: np.ndarray,
    k: int = 10,
) -> float:
    """
    Calculate top-k accuracy (Jaccard similarity of top-k sets).

    Parameters
    ----------
    pred_rankings : np.ndarray
        Predicted rankings (lower = better)
    true_rankings : np.ndarray
        True rankings (lower = better)
    k : int
        Number of top items to consider

    Returns
    -------
    float
        Jaccard similarity (0-1, higher = better)
    """
    pred_rankings = np.asarray(pred_rankings)
    true_rankings = np.asarray(true_rankings)

    # Get top-k indices
    pred_top_k = set(np.argsort(pred_rankings)[:k])
    true_top_k = set(np.argsort(true_rankings)[:k])

    # Calculate Jaccard similarity
    intersection = len(pred_top_k & true_top_k)
    union = len(pred_top_k | true_top_k)

    if union == 0:
        return 0.0

    return intersection / union
