"""Probability model transformations for ranking systems.

This module implements the Bradley-Terry probability model and other
transformations used in the ranking engine.
"""

from __future__ import annotations

import math


def bt_prob(
    score_a: float,
    score_b: float,
    *,
    alpha: float = 1.0,
    epsilon: float = 1e-12,
) -> float:
    """
    Bradley-Terry win probability using PageRank scores.

    Parameters
    ----------
    score_a, score_b : float
        Positive scores for competitors A and B.
    alpha : float, optional
        Temperature parameter. alpha = 1 reproduces classical BT.
        Values > 1 make outcomes more deterministic, < 1 more random.
    epsilon : float, optional
        Small value to ensure positive scores.

    Returns
    -------
    float
        Probability that A beats B.

    Examples
    --------
    >>> bt_prob(0.6, 0.4)  # Classical Bradley-Terry
    0.6
    >>> bt_prob(0.01, 0.002, alpha=2.0)  # With temperature scaling
    0.9615384615384615
    """
    score_a = max(score_a, epsilon)
    score_b = max(score_b, epsilon)

    if alpha == 1.0:
        return score_a / (score_a + score_b)
    else:
        log_ratio = alpha * (math.log(score_a) - math.log(score_b))
        return 1.0 / (1.0 + math.exp(-log_ratio))


def log_center(scores: list[float], *, epsilon: float = 1e-12) -> list[float]:
    """
    Center scores in log space for numerical stability.

    This can help with extreme score ratios by centering around
    the geometric mean.

    Parameters
    ----------
    scores : list[float]
        List of positive scores
    epsilon : float, optional
        Small value to ensure positive scores

    Returns
    -------
    list[float]
        Log-centered scores
    """
    scores = [max(score, epsilon) for score in scores]
    log_scores = [math.log(score) for score in scores]
    log_mean = sum(log_scores) / len(log_scores)
    centered_log_scores = [log_score - log_mean for log_score in log_scores]
    return [
        math.exp(centered_log_score)
        for centered_log_score in centered_log_scores
    ]
