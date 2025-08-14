"""
Continuous tournament scraping module.

This module provides a robust continuous scraping system that handles:
- Tournament lifecycle tracking (scheduled -> in-progress -> completed)
- Intelligent retry strategies for different tournament states
- Differentiation between 404s (deleted vs not-yet-created)
- Automatic cleanup of stale scheduled tournaments
- Efficient incremental updates
"""

from rankings.continuous.manager import ContinuousScraper
from rankings.continuous.state import TournamentState, TournamentStateTracker
from rankings.continuous.strategies import ScrapingStrategy

__all__ = [
    "ContinuousScraper",
    "TournamentState",
    "TournamentStateTracker",
    "ScrapingStrategy",
]
