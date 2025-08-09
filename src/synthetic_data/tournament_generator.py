"""Compatibility shim: expose TournamentGenerator & TournamentFormat.

Prefer importing from synthetic_data.core.tournament_generator directly in new code.
"""

from .core.tournament_generator import (  # re-export
    TournamentFormat,
    TournamentGenerator,
)

__all__ = ["TournamentGenerator", "TournamentFormat"]
