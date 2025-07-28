"""
Summary preparation functions for tournament ranking analysis.

This module provides functions to create comprehensive summaries of players,
teams, and tournaments with additional statistics and rankings.
"""

from __future__ import annotations

from typing import Optional

import polars as pl

from rankings.core.constants import MIN_TOURNAMENTS_FOR_RANKING


def prepare_player_summary(
    players_df: pl.DataFrame,
    rankings_df: pl.DataFrame,
    min_tournaments: int = MIN_TOURNAMENTS_FOR_RANKING,
    rank_column: str = "player_rank",
    score_column: str = "player_rank",
) -> pl.DataFrame:
    """
    Create a summary of player rankings with additional statistics.

    Parameters
    ----------
    players_df : pl.DataFrame
        Players DataFrame from tournament parser
    rankings_df : pl.DataFrame
        Rankings DataFrame from ranking functions (with 'id' column for player IDs)
    min_tournaments : int, optional
        Minimum tournaments for inclusion in results
    rank_column : str, optional
        Name of the ranking column to create
    score_column : str, optional
        Name of the score column from rankings_df

    Returns
    -------
    pl.DataFrame
        Enhanced player summary with rankings and statistics
    """
    # Group players by user_id to get unique players and tournament counts
    players_unique = players_df.group_by("user_id").agg(
        [
            pl.all().sort_by("tournament_id", descending=True).first(),
            pl.len().alias("tournament_count"),
        ]
    )

    # Join with rankings - rankings_df has 'id' column, players has 'user_id'
    # Rename the score column to avoid conflicts when creating rank_column
    rankings_renamed = rankings_df.rename(
        {score_column: f"{score_column}_score"}
    )
    result = (
        players_unique.join(
            rankings_renamed, left_on="user_id", right_on="id", how="left"
        )
        .filter(pl.col(f"{score_column}_score").is_not_null())
        .filter(pl.col("tournament_count") >= min_tournaments)
    )

    # Add normalized score and ranking
    if result.height > 0:
        score_col_renamed = f"{score_column}_score"
        result = result.with_columns(
            [
                (
                    (
                        (
                            pl.col(score_col_renamed)
                            - pl.col(score_col_renamed).min()
                        )
                        * 100
                    )
                    / (
                        pl.col(score_col_renamed).max()
                        - pl.col(score_col_renamed).min()
                    )
                ).alias("score_normalized"),
                pl.col(score_col_renamed)
                .rank(method="min", descending=True)
                .alias(rank_column),
            ]
        )

    # Select and order columns nicely - use the renamed score column
    score_col_renamed = f"{score_column}_score"
    return result.select(
        [
            "username",
            "user_id",
            score_col_renamed,
            "score_normalized",
            rank_column,
            "tournament_count",
            "in_game_name",
            "country",
        ]
    ).sort("score_normalized", descending=True)


def prepare_team_summary(
    teams_df: pl.DataFrame,
    rankings_df: pl.DataFrame,
    rank_column: str = "team_rank",
    score_column: str = "team_rank",
) -> pl.DataFrame:
    """
    Create a summary of team rankings with additional statistics.

    Parameters
    ----------
    teams_df : pl.DataFrame
        Teams DataFrame from tournament parser
    rankings_df : pl.DataFrame
        Rankings DataFrame from ranking functions
    rank_column : str, optional
        Name of the ranking column to create
    score_column : str, optional
        Name of the score column from rankings_df

    Returns
    -------
    pl.DataFrame
        Enhanced team summary with rankings and statistics
    """
    # Group teams by team_id to get unique teams and tournament counts
    teams_unique = teams_df.group_by("team_id").agg(
        [
            pl.all().sort_by("tournament_id", descending=True).first(),
            pl.len().alias("tournament_count"),
        ]
    )

    # Join with rankings
    result = teams_unique.join(rankings_df, on="team_id", how="left").filter(
        pl.col(score_column).is_not_null()
    )

    # Add normalized score and ranking
    if result.height > 0:
        result = result.with_columns(
            [
                (
                    ((pl.col(score_column) - pl.col(score_column).min()) * 100)
                    / (pl.col(score_column).max() - pl.col(score_column).min())
                ).alias("score_normalized"),
                pl.col(score_column)
                .rank(method="min", descending=True)
                .alias(rank_column),
            ]
        )

    # Select and order columns
    return result.select(
        [
            "team_name",
            "team_id",
            score_column,
            "score_normalized",
            rank_column,
            "tournament_count",
            "seed",
            "dropped_out",
        ]
    ).sort("score_normalized", descending=True)


def prepare_tournament_summary(
    tournament_data: list[dict],
    tournament_influence: Optional[dict[int, float]] = None,
    tournament_strength: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """
    Create a summary of tournaments with names and strength metrics.

    Parameters
    ----------
    tournament_data : list[dict]
        Raw tournament data from JSON
    tournament_influence : dict[int, float], optional
        Tournament influence values from RatingEngine
    tournament_strength : pl.DataFrame, optional
        Tournament strength DataFrame from RatingEngine

    Returns
    -------
    pl.DataFrame
        Tournament summary with names and strength metrics
    """
    # Extract tournament metadata
    tournament_meta = []
    for entry in tournament_data:
        ctx = entry.get("tournament", {}).get("ctx", {})
        if "id" in ctx:
            tournament_meta.append(
                {
                    "tournament_id": ctx["id"],
                    "name": ctx.get("name", "Unknown"),
                    "created_at": ctx.get("createdAt"),
                    "bracket_url": ctx.get("bracketUrl"),
                    "is_ranked": ctx.get("isRanked", False),
                }
            )

    result = pl.DataFrame(tournament_meta)

    # Add influence if provided
    if tournament_influence:
        influence_df = pl.DataFrame(
            {
                "tournament_id": list(tournament_influence.keys()),
                "influence": list(tournament_influence.values()),
            }
        )
        result = result.join(influence_df, on="tournament_id", how="left")

    # Add strength if provided
    if tournament_strength is not None:
        result = result.join(
            tournament_strength, on="tournament_id", how="left"
        )

    return result.sort("influence", descending=True, nulls_last=True)
