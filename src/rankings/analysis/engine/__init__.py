"""Rankings engine module for tournament-based rating calculations.

This module provides the RatingEngine class and related utilities for computing
player and team rankings using a PageRank-based algorithm with tournament
strength weighting and teleport-proportional smoothing.
"""

from __future__ import annotations

from .core import RatingEngine
from .teleport import make_participation_inverse_teleport

__all__ = [
    "RatingEngine",
    "make_participation_inverse_teleport",
]
