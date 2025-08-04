"""
Probability model transformations for ranking systems.

This module implements the Bradley-Terry probability model and other
transformations used in the ranking engine.
"""

import math


def bt_prob(
    sa: float, sb: float, *, alpha: float = 1.0, eps: float = 1e-12
) -> float:
    """
    Bradley-Terry win probability using PageRank scores.

    Parameters
    ----------
    sa, sb : float
        Positive scores for competitors A and B.
    alpha : float, optional
        Temperature parameter. alpha = 1 reproduces classical BT.
        Values > 1 make outcomes more deterministic, < 1 more random.
    eps : float, optional
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
    0.8807970779778824
    """
    # Ensure positive scores
    sa = max(sa, eps)
    sb = max(sb, eps)

    if alpha == 1.0:
        # Classical Bradley-Terry formula: sa / (sa + sb)
        return sa / (sa + sb)
    else:
        # Logistic on log-ratio for alpha ≠ 1
        z = alpha * (math.log(sa) - math.log(sb))
        return 1.0 / (1.0 + math.exp(-z))


def log_center(scores: list[float], *, eps: float = 1e-12) -> list[float]:
    """
    Center scores in log space for numerical stability.

    This can help with extreme score ratios by centering around
    the geometric mean.

    Parameters
    ----------
    scores : list[float]
        List of positive scores
    eps : float, optional
        Small value to ensure positive scores

    Returns
    -------
    list[float]
        Log-centered scores
    """
    scores = [max(s, eps) for s in scores]
    log_scores = [math.log(s) for s in scores]
    log_mean = sum(log_scores) / len(log_scores)
    centered_log_scores = [ls - log_mean for ls in log_scores]
    return [math.exp(cls) for cls in centered_log_scores]
