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
    bt_prob,
    create_match_summary_with_names,
    derive_team_ratings_from_players,
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
    MIN_TOURNAMENTS_BEFORE_CV,
    SENDOU_BASE_URL,
)

# Evaluation functionality
from rankings.evaluation import (  # Loss functions; Metrics; Cross-validation; Optimization
    BayesianOptimizer,
    GridSearchOptimizer,
    bucketised_metrics,
    compute_accuracy,
    compute_accuracy_at_threshold,
    compute_brier_score,
    compute_cross_tournament_loss,
    compute_expected_upset_rate,
    compute_match_loss,
    compute_match_probability,
    compute_round_metrics,
    compute_spearman_correlation,
    compute_tournament_loss,
    compute_weighted_log_loss,
    create_time_based_folds,
    cross_validate_ratings,
    evaluate_by_rating_separation,
    evaluate_on_split,
    fit_alpha_parameter,
    optimize_rating_engine,
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
    # Analysis - Probability models
    "bt_prob",
    # Analysis - Utility functions
    "get_most_influential_matches",
    "add_player_names",
    "add_team_names",
    "add_tournament_names",
    "add_match_timestamps",
    "analyze_player_performance",
    "create_match_summary_with_names",
    "derive_team_ratings_from_players",
    "display_player_rankings",
    "format_influential_matches",
    "format_tournament_influence_summary",
    "generate_tournament_report",
    "get_player_match_history",
    "get_tournament_name_lookup",
    "prepare_player_summary",
    "prepare_tournament_summary",
    # Evaluation - Loss functions
    "compute_match_probability",
    "compute_match_loss",
    "compute_tournament_loss",
    "compute_cross_tournament_loss",
    "compute_weighted_log_loss",
    "bucketised_metrics",
    "fit_alpha_parameter",
    # Evaluation - Metrics
    "compute_brier_score",
    "compute_accuracy",
    "compute_accuracy_at_threshold",
    "compute_expected_upset_rate",
    "evaluate_by_rating_separation",
    "compute_spearman_correlation",
    "compute_round_metrics",
    # Evaluation - Cross-validation
    "create_time_based_folds",
    "evaluate_on_split",
    "cross_validate_ratings",
    # Evaluation - Optimization
    "GridSearchOptimizer",
    "BayesianOptimizer",
    "optimize_rating_engine",
    # Constants
    "DEFAULT_DECAY_HALF_LIFE_DAYS",
    "DEFAULT_DAMPING_FACTOR",
    "DEFAULT_TOURNAMENT_STRENGTH_WEIGHT",
    "DEFAULT_PAGERANK_TOLERANCE",
    "DEFAULT_MAX_ITERATIONS",
    "MIN_TOURNAMENTS_BEFORE_CV",
    "SENDOU_BASE_URL",
    "CALENDAR_URL",
]
