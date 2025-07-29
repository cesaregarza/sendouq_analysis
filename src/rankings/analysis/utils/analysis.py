"""
Higher-level analysis functions for tournament rankings.

This module provides convenience functions that combine multiple utility
functions for common analysis tasks.
"""

from __future__ import annotations

from typing import Optional

import polars as pl

from rankings.analysis.engine import RatingEngine
from rankings.analysis.utils.formatting import (
    format_influential_matches,
    format_tournament_influence_summary,
)
from rankings.analysis.utils.matches import get_most_influential_matches
from rankings.analysis.utils.names import get_tournament_name_lookup


def analyze_player_performance(
    player_id: int,
    player_name: str,
    engine: RatingEngine,
    matches_df: pl.DataFrame,
    players_df: pl.DataFrame,
    tournament_data: list[dict],
    top_n_matches: int = 10,
    print_output: bool = True,
) -> dict:
    """
    Comprehensive analysis of a player's performance and influential matches.

    Parameters
    ----------
    player_id : int
        The user_id of the player to analyze
    player_name : str
        The username of the player
    engine : RatingEngine
        The RatingEngine instance with computed rankings
    matches_df : pl.DataFrame
        Matches DataFrame
    players_df : pl.DataFrame
        Players DataFrame
    tournament_data : list[dict]
        Raw tournament data
    top_n_matches : int, optional
        Number of top influential matches to analyze
    print_output : bool, optional
        Whether to print the formatted output

    Returns
    -------
    dict
        Dictionary containing:
        - influential_matches: dict with wins/losses DataFrames
        - tournament_names: dict of tournament ID to name mapping
        - formatted_output: str with formatted analysis
    """
    # Get tournament names
    tournament_names = get_tournament_name_lookup(tournament_data)

    # Get influential matches
    influential_matches = get_most_influential_matches(
        player_id=player_id,
        matches_df=matches_df,
        players_df=players_df,
        engine=engine,
        top_n=top_n_matches,
    )

    # Format the output
    formatted_output = format_influential_matches(
        influential_matches,
        player_name=player_name,
        tournament_names=tournament_names,
    )

    if print_output:
        print(formatted_output)

    return {
        "influential_matches": influential_matches,
        "tournament_names": tournament_names,
        "formatted_output": formatted_output,
    }


def generate_tournament_report(
    engine: "RatingEngine",
    tournament_data: list[dict],
    top_n: int = 10,
    print_output: bool = True,
) -> dict:
    """
    Generate a comprehensive tournament strength report.

    Parameters
    ----------
    engine : RatingEngine
        The RatingEngine instance with computed rankings
    tournament_data : list[dict]
        Raw tournament data
    top_n : int, optional
        Number of top tournaments to show
    print_output : bool, optional
        Whether to print the formatted output

    Returns
    -------
    dict
        Dictionary containing:
        - tournament_influence: dict of tournament influences
        - tournament_strength: DataFrame of tournament strengths
        - tournament_names: dict of tournament ID to name mapping
        - formatted_output: str with formatted analysis
    """
    # Get tournament names
    tournament_names = get_tournament_name_lookup(tournament_data)

    # Format the output
    formatted_output = format_tournament_influence_summary(
        tournament_influence=engine.tournament_influence,
        tournament_strength=engine.tournament_strength,
        tournament_names=tournament_names,
        top_n=top_n,
    )

    if print_output:
        print(formatted_output)

    return {
        "tournament_influence": engine.tournament_influence,
        "tournament_strength": engine.tournament_strength,
        "tournament_names": tournament_names,
        "formatted_output": formatted_output,
    }


def compare_player_performances(
    player_ids: list[int],
    engine: RatingEngine,
    matches_df: pl.DataFrame,
    players_df: pl.DataFrame,
    player_summary: pl.DataFrame,
    tournament_data: list[dict],
    top_n_matches: int = 5,
) -> dict:
    """
    Compare performance metrics for multiple players.

    Parameters
    ----------
    player_ids : list[int]
        List of player IDs to compare
    engine : RatingEngine
        The RatingEngine instance
    matches_df : pl.DataFrame
        Matches DataFrame
    players_df : pl.DataFrame
        Players DataFrame
    player_summary : pl.DataFrame
        Player summary with rankings
    tournament_data : list[dict]
        Raw tournament data
    top_n_matches : int, optional
        Number of top matches to show per player

    Returns
    -------
    dict
        Dictionary with player_id as keys and analysis results as values
    """
    results = {}
    tournament_names = get_tournament_name_lookup(tournament_data)

    for player_id in player_ids:
        # Get player info
        player_info = player_summary.filter(
            pl.col("user_id") == player_id
        ).head(1)

        if player_info.is_empty():
            continue

        player_name = player_info["username"][0]
        player_rank = player_info["player_rank"][0]
        player_score = player_info["score_normalized"][0]

        # Get influential matches
        influential_matches = get_most_influential_matches(
            player_id=player_id,
            matches_df=matches_df,
            players_df=players_df,
            engine=engine,
            top_n=top_n_matches,
        )

        # Calculate win/loss statistics
        total_wins = influential_matches["wins"].height
        total_losses = influential_matches["losses"].height

        avg_win_weight = (
            influential_matches["wins"]["match_weight"].mean()
            if total_wins > 0
            else 0
        )
        avg_loss_weight = (
            influential_matches["losses"]["match_weight"].mean()
            if total_losses > 0
            else 0
        )

        results[player_id] = {
            "name": player_name,
            "rank": player_rank,
            "score": player_score,
            "influential_wins": total_wins,
            "influential_losses": total_losses,
            "avg_win_weight": avg_win_weight,
            "avg_loss_weight": avg_loss_weight,
            "win_loss_ratio": total_wins / max(total_losses, 1),
            "influential_matches": influential_matches,
        }

    return results
