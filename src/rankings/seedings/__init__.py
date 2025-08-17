"""
Entropy-controlled division assignment system for team seedings.

This module implements a parameter-free approach to division assignment
that uses Shannon entropy to measure team balance and adjust ratings
accordingly.
"""

from .entropy_seeding import (
    EntropyDivisionAssigner,
    assign_divisions,
    compute_entropy_controlled_rating,
    compute_team_entropy,
)

__all__ = [
    "EntropyDivisionAssigner",
    "compute_entropy_controlled_rating",
    "compute_team_entropy",
    "assign_divisions",
]
