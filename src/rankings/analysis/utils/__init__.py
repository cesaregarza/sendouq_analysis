"""
Utility functions for tournament ranking analysis.

This module is organized into submodules:
- summaries: Functions for preparing summary DataFrames
- formatting: Functions for formatting data for display
- comparisons: Functions for comparing rankings and entities
- matches: Functions for match analysis
- names: Functions for adding human-readable names
- analysis: Higher-level analysis functions
"""

# Higher-level analysis functions
from rankings.analysis.utils.analysis import (
    analyze_player_performance,
    compare_player_performances,
    generate_tournament_report,
)

# Comparison functions
from rankings.analysis.utils.comparisons import (
    compare_rankings,
    get_head_to_head_record,
)

# Formatting functions
from rankings.analysis.utils.formatting import (
    display_player_rankings,
    format_influential_matches,
    format_top_rankings,
    format_tournament_influence_summary,
)

# Match analysis functions
from rankings.analysis.utils.matches import (
    get_most_influential_matches,
    get_player_match_history,
)

# Name resolution functions
from rankings.analysis.utils.names import (
    add_match_timestamps,
    add_player_names,
    add_team_names,
    add_tournament_names,
    create_match_summary_with_names,
    get_tournament_name_lookup,
)

# Summary functions
from rankings.analysis.utils.summaries import (
    derive_team_ratings_from_players,
    prepare_player_summary,
    prepare_team_summary,
    prepare_tournament_summary,
)

__all__ = [
    # Higher-level analysis
    "analyze_player_performance",
    "compare_player_performances",
    "generate_tournament_report",
    # Summaries
    "derive_team_ratings_from_players",
    "prepare_player_summary",
    "prepare_team_summary",
    "prepare_tournament_summary",
    # Formatting
    "format_top_rankings",
    "display_player_rankings",
    "format_influential_matches",
    "format_tournament_influence_summary",
    # Comparisons
    "compare_rankings",
    "get_head_to_head_record",
    # Matches
    "get_most_influential_matches",
    "get_player_match_history",
    # Names
    "add_player_names",
    "add_team_names",
    "add_tournament_names",
    "add_match_timestamps",
    "create_match_summary_with_names",
    "get_tournament_name_lookup",
]
