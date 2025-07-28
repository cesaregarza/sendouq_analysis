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
"""

# Analysis functionality
from rankings.analysis import (
    RatingEngine,
    add_match_timestamps,
    add_player_names,
    add_team_names,
    add_tournament_names,
    analyze_player_performance,
    create_match_summary_with_names,
    display_player_rankings,
    format_influential_matches,
    format_tournament_influence_summary,
    generate_tournament_report,
    get_most_influential_matches,
    get_player_match_history,
    get_tournament_name_lookup,
    prepare_player_summary,
    prepare_tournament_summary,
)

# Core functionality
from rankings.core import parse_tournaments_data

# Key constants for convenience
from rankings.core.constants import (
    CALENDAR_URL,
    DEFAULT_DAMPING_FACTOR,
    DEFAULT_DECAY_HALF_LIFE_DAYS,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_PAGERANK_TOLERANCE,
    DEFAULT_TOURNAMENT_STRENGTH_WEIGHT,
    SENDOU_BASE_URL,
)

# Scraping functionality
from rankings.scraping import (  # API functions; Discovery functions; Batch processing; Storage
    build_tournament_url,
    discover_tournaments_from_calendar,
    extract_tournament_id_from_url,
    get_latest_tournament_id,
    get_tournament_summary,
    load_scraped_tournaments,
    scrape_latest_tournaments,
    scrape_to_database,
    scrape_tournament,
    scrape_tournament_batch,
    scrape_tournament_range,
    scrape_tournaments_from_calendar,
    validate_tournament_data,
)

__version__ = "0.2.0"

__all__ = [
    # Core
    "parse_tournaments_data",
    # Scraping - API
    "scrape_tournament",
    "build_tournament_url",
    "validate_tournament_data",
    "extract_tournament_id_from_url",
    # Scraping - Discovery
    "discover_tournaments_from_calendar",
    "get_latest_tournament_id",
    # Scraping - Batch processing
    "scrape_tournament_batch",
    "scrape_tournament_range",
    "scrape_latest_tournaments",
    "scrape_tournaments_from_calendar",
    "scrape_to_database",
    # Scraping - Storage
    "load_scraped_tournaments",
    "get_tournament_summary",
    # Analysis - Engine
    "RatingEngine",
    # Analysis - Utility functions
    "get_most_influential_matches",
    "add_player_names",
    "add_team_names",
    "add_tournament_names",
    "add_match_timestamps",
    "analyze_player_performance",
    "create_match_summary_with_names",
    "display_player_rankings",
    "format_influential_matches",
    "format_tournament_influence_summary",
    "generate_tournament_report",
    "get_player_match_history",
    "get_tournament_name_lookup",
    "prepare_player_summary",
    "prepare_tournament_summary",
    # Constants
    "DEFAULT_DECAY_HALF_LIFE_DAYS",
    "DEFAULT_DAMPING_FACTOR",
    "DEFAULT_TOURNAMENT_STRENGTH_WEIGHT",
    "DEFAULT_PAGERANK_TOLERANCE",
    "DEFAULT_MAX_ITERATIONS",
    "SENDOU_BASE_URL",
    "CALENDAR_URL",
]
