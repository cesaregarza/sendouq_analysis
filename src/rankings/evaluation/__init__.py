"""Evaluation module for tournament ranking models."""

from rankings.evaluation.cross_validation import (
    create_time_based_splits,
    cross_validate_ratings,
    evaluate_on_split,
)
from rankings.evaluation.loss import (
    bucketised_metrics,
    compute_cross_tournament_loss,
    compute_match_loss,
    compute_match_probability,
    compute_tournament_loss,
    compute_weighted_log_loss,
    fit_alpha_parameter,
)
from rankings.evaluation.metrics import (
    aggregate_tournament_metrics,
    compute_accuracy,
    compute_accuracy_at_threshold,
    compute_brier_score,
    compute_expected_upset_rate,
    compute_round_metrics,
    compute_spearman_correlation,
    evaluate_by_rating_separation,
    evaluate_tournament_predictions,
)
from rankings.evaluation.optimizer import (
    BayesianOptimizer,
    GridSearchOptimizer,
    optimize_rating_engine,
)

__all__ = [
    # Loss functions
    "compute_match_probability",
    "compute_match_loss",
    "compute_tournament_loss",
    "compute_cross_tournament_loss",
    "compute_weighted_log_loss",
    "bucketised_metrics",
    "fit_alpha_parameter",
    # Metrics
    "compute_brier_score",
    "compute_accuracy",
    "compute_accuracy_at_threshold",
    "compute_expected_upset_rate",
    "evaluate_by_rating_separation",
    "compute_spearman_correlation",
    "compute_round_metrics",
    "evaluate_tournament_predictions",
    "aggregate_tournament_metrics",
    # Cross-validation
    "create_time_based_splits",
    "evaluate_on_split",
    "cross_validate_ratings",
    # Optimization
    "GridSearchOptimizer",
    "BayesianOptimizer",
    "optimize_rating_engine",
]
