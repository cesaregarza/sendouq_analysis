"""Cross-validation module for ranking evaluation.

This module provides various cross-validation approaches:
- Simple temporal cross-validation with fast optimization
- Advanced cross-validation with multiple splitting strategies
- Utilities for creating and visualizing splits
"""

from .cross_validation_advanced import (
    create_time_based_folds,
    cross_validate_ratings,
    evaluate_on_split,
)
from .cross_validation_simple import cross_validate_simple
from .simple_splits import (
    create_simple_time_splits,
    get_split_info,
    visualize_splits,
)

__all__ = [
    # Simple CV
    "cross_validate_simple",
    # Advanced CV
    "cross_validate_ratings",
    "create_time_based_folds",
    "evaluate_on_split",
    # Splitting utilities
    "create_simple_time_splits",
    "visualize_splits",
    "get_split_info",
]
