"""Sendou.ink Tournament Rankings."""

from __future__ import annotations

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
