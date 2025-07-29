"""
Tournament analysis and ranking algorithms.

This module contains the advanced analysis capabilities:
- engine: Advanced RatingEngine with tick-tock algorithm
- utils: Analysis utilities and result formatting
"""

from rankings.analysis.engine import RatingEngine
from rankings.analysis.utils import (
    add_match_timestamps,
    add_player_names,
    add_team_names,
    add_tournament_names,
    analyze_player_performance,
    create_match_summary_with_names,
    derive_team_ratings_from_players,
    display_player_rankings,
    format_influential_matches,
    format_top_rankings,
    format_tournament_influence_summary,
    generate_tournament_report,
    get_most_influential_matches,
    get_player_match_history,
    get_tournament_name_lookup,
    prepare_player_summary,
    prepare_tournament_summary,
)

__all__ = [
    # Engine
    "RatingEngine",
    # Utils
    "prepare_player_summary",
    "prepare_tournament_summary",
    "derive_team_ratings_from_players",
    "format_top_rankings",
    "get_most_influential_matches",
    "add_player_names",
    "add_team_names",
    "add_tournament_names",
    "add_match_timestamps",
    "create_match_summary_with_names",
    "get_player_match_history",
    "get_tournament_name_lookup",
    "analyze_player_performance",
    "display_player_rankings",
    "format_influential_matches",
    "format_tournament_influence_summary",
    "generate_tournament_report",
]
