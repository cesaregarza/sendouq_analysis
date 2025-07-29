"""
Configuration constants for tournament ranking and scraping algorithms.

This module centralizes all default parameters used by both scraping and advanced
ranking implementations to ensure consistency and easy tuning.
"""

import math
from datetime import datetime
from zoneinfo import ZoneInfo

# =============================================================================
# Time and Decay Parameters
# =============================================================================

# Default reference time for calculations
DEFAULT_REFERENCE_DATE = datetime(
    2025, 7, 23, tzinfo=ZoneInfo("America/Chicago")
)

# Time decay parameters
DEFAULT_DECAY_HALF_LIFE_DAYS: float = 30.0
DEFAULT_DECAY_RATE: float = math.log(2.0) / DEFAULT_DECAY_HALF_LIFE_DAYS
SECONDS_PER_DAY: float = 86_400.0

# =============================================================================
# Scraping Configuration
# =============================================================================

# Sendou.ink URLs
SENDOU_BASE_URL = "https://sendou.ink/to/"
SENDOU_DATA_SUFFIX = "?_data=features%2Ftournament%2Froutes%2Fto.%24id"
CALENDAR_URL = "https://sendou.ink/calendar.ics?tournament=true"

# Scraping defaults
DEFAULT_TIMEOUT = 10.0
DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_FACTOR = 1.5
DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_FAILURES = 10

# =============================================================================
# PageRank Algorithm Parameters
# =============================================================================

# PageRank damping factor (probability of following links vs random teleport)
DEFAULT_DAMPING_FACTOR: float = 0.85

# Convergence criteria
DEFAULT_PAGERANK_TOLERANCE: float = 1e-8
DEFAULT_MAX_ITERATIONS: int = 100

# Advanced engine specific parameters
DEFAULT_TICK_TOCK_TOLERANCE: float = 1e-4
DEFAULT_MAX_TICK_TOCK: int = 5
DEFAULT_MAX_PAGERANK_ITER: int = 150

# =============================================================================
# Tournament Strength Parameters
# =============================================================================

# Tournament strength weighting
DEFAULT_TOURNAMENT_STRENGTH_WEIGHT: float = 1.0
MIN_TEAMS_FOR_STRENGTH: int = 8

# Advanced engine parameters
DEFAULT_BETA: float = 0.0  # Tournament strength exponent (0.0 = no strength weighting, 1.0 = full weighting)
DEFAULT_INFLUENCE_AGG_METHOD: str = (
    "mean"  # "mean", "sum", "median", "top_20_sum"
)
DEFAULT_STRENGTH_AGG: str = (
    "mean"  # "mean", "median", "trimmed_mean", "topN_sum"
)
DEFAULT_STRENGTH_K: int = 5  # Parameter for trimmed_mean or topN_sum

# =============================================================================
# Teleport Vector Types
# =============================================================================

# Teleport vector options for PageRank
TELEPORT_UNIFORM = "uniform"
TELEPORT_VOLUME_INVERSE = "volume_inverse"

# =============================================================================
# Activity Thresholds
# =============================================================================

# Minimum tournaments for inclusion in rankings
MIN_TOURNAMENTS_FOR_RANKING: int = 5

# Minimum tournaments that must exist before first test tournament in cross-validation
MIN_TOURNAMENTS_BEFORE_CV: int = 10

# Minimum matches for edge weight consideration
MIN_MATCHES_FOR_EDGE: int = 1
