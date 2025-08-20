"""
Sendou.ink Tournament Rankings

This module provides comprehensive tournament ranking capabilities for Sendou.ink data,
organized into focused submodules for better maintainability:

Core Components:
- core: Fundamental parsing and configuration
- scraping: Tournament data acquisition from Sendou.ink API
- analysis: Advanced rating engines and ranking algorithms

Examples:
    Scraping:
        >>> from rankings import scrape_tournament, scrape_latest_tournaments
        >>> tournament_data = scrape_tournament(1955)
        >>> results = scrape_latest_tournaments(count=50)

    Parsing and Ranking:
        >>> from rankings import parse_tournaments_data, RatingEngine
        >>> tables = parse_tournaments_data(tournament_data)
        >>> engine = RatingEngine(beta=1.0, influence_agg_method="top_20_sum")
        >>> player_rankings = engine.rank_players(tables['matches'], tables['players'])
        >>> tournament_strength = engine.tournament_strength

    For more specific functionality, import from submodules directly:
        >>> from rankings.analysis import generate_tournament_report
        >>> from rankings.evaluation import cross_validate_ratings
        >>> from rankings.calibration import MatchCalibrator
"""

# Core functionality - Main API
from rankings.analysis import RatingEngine
from rankings.calibration import CalibrationResult, MatchCalibrator
from rankings.continuous import ContinuousScraper
from rankings.core import parse_tournaments_data

# Key constants for convenience
from rankings.core.constants import (
    DEFAULT_BETA,
    DEFAULT_DAMPING_FACTOR,
    DEFAULT_DECAY_HALF_LIFE_DAYS,
    MIN_TOURNAMENTS_BEFORE_CV,
)
from rankings.evaluation import cross_validate_ratings, optimize_rating_engine
from rankings.scraping import (
    load_scraped_tournaments,
    scrape_latest_tournaments,
    scrape_tournament,
    scrape_tournaments_from_calendar,
)

__version__ = "0.2.0"

__all__ = [
    # Core API - Essential functions
    "parse_tournaments_data",
    "RatingEngine",
    # Scraping - Main functions
    "scrape_tournament",
    "scrape_latest_tournaments",
    "scrape_tournaments_from_calendar",
    "load_scraped_tournaments",
    # Continuous scraping
    "ContinuousScraper",
    # Evaluation - Main functions
    "cross_validate_ratings",
    "optimize_rating_engine",
    # Calibration
    "MatchCalibrator",
    "CalibrationResult",
    # Essential constants
    "DEFAULT_BETA",
    "DEFAULT_DAMPING_FACTOR",
    "DEFAULT_DECAY_HALF_LIFE_DAYS",
    "MIN_TOURNAMENTS_BEFORE_CV",
    # Version
    "__version__",
]

# Note: For advanced functionality, import directly from submodules:
# - rankings.analysis: Analysis utilities and formatting functions
# - rankings.evaluation: Loss functions, metrics, and cross-validation
# - rankings.calibration: Advanced calibration methods
# - rankings.scraping: Batch processing and storage utilities
# - rankings.continuous: Tournament state tracking and strategies
