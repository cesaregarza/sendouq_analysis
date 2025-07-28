"""
Utility functions for tournament ranking analysis.

This module provides helper functions for data preparation, result formatting,
and common analysis tasks.
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


def format_top_rankings(
    rankings_df: pl.DataFrame,
    n: int = 20,
    score_column: str = "score_normalized",
    name_column: str = "username",
) -> str:
    """
    Format top rankings as a nice string for display.

    Parameters
    ----------
    rankings_df : pl.DataFrame
        Rankings DataFrame with score and name columns
    n : int, optional
        Number of top entries to show
    score_column : str, optional
        Column name containing scores
    name_column : str, optional
        Column name containing names

    Returns
    -------
    str
        Formatted ranking string
    """
    if rankings_df.is_empty():
        return "No rankings available"

    top_n = rankings_df.head(n)

    lines = [f"Top {n} Rankings:"]
    lines.append("=" * 50)

    for i, row in enumerate(top_n.iter_rows(named=True), 1):
        score = row.get(score_column, 0)
        name = row.get(name_column, "Unknown")
        lines.append(f"{i:2d}. {name:<20} {score:6.1f}")

    return "\n".join(lines)


def compare_rankings(
    rankings1: pl.DataFrame,
    rankings2: pl.DataFrame,
    id_column: str = "user_id",
    score_column1: str = "player_rank_1",
    score_column2: str = "player_rank_2",
    name_column: str = "username",
    n: int = 15,
) -> pl.DataFrame:
    """
    Compare two ranking systems and show differences.

    Parameters
    ----------
    rankings1 : pl.DataFrame
        First ranking system results
    rankings2 : pl.DataFrame
        Second ranking system results
    id_column : str, optional
        Column name for entity IDs
    score_column1 : str, optional
        Score column name for first ranking
    score_column2 : str, optional
        Score column name for second ranking
    name_column : str, optional
        Column name for entity names
    n : int, optional
        Number of top entries to compare

    Returns
    -------
    pl.DataFrame
        Comparison DataFrame showing rank changes
    """
    # Add rank columns to each dataset
    r1 = rankings1.with_columns(
        pl.col(score_column1.replace("_1", ""))
        .rank(method="min", descending=True)
        .alias("rank_1")
    ).select([id_column, "rank_1", name_column])

    r2 = rankings2.with_columns(
        pl.col(score_column2.replace("_2", ""))
        .rank(method="min", descending=True)
        .alias("rank_2")
    ).select([id_column, "rank_2"])

    # Join and compute differences
    comparison = (
        r1.join(r2, on=id_column, how="inner")
        .with_columns(
            (pl.col("rank_1") - pl.col("rank_2")).alias("rank_change")
        )
        .sort("rank_2")  # Sort by second ranking
    )

    return comparison.head(n)


def get_head_to_head_record(
    matches_df: pl.DataFrame,
    entity1_id: int,
    entity2_id: int,
    id_type: str = "team",
) -> dict:
    """
    Get head-to-head record between two entities.

    Parameters
    ----------
    matches_df : pl.DataFrame
        Matches DataFrame
    entity1_id : int
        First entity ID
    entity2_id : int
        Second entity ID
    id_type : str, optional
        Type of entity ("team" or "user")

    Returns
    -------
    dict
        Head-to-head statistics
    """
    if id_type == "team":
        winner_col = "winner_team_id"
        loser_col = "loser_team_id"
    else:
        winner_col = "winner_user_id"
        loser_col = "loser_user_id"

    # Find matches between the two entities
    direct_matches = matches_df.filter(
        ((pl.col(winner_col) == entity1_id) & (pl.col(loser_col) == entity2_id))
        | (
            (pl.col(winner_col) == entity2_id)
            & (pl.col(loser_col) == entity1_id)
        )
    )

    if direct_matches.is_empty():
        return {
            "total_matches": 0,
            "entity1_wins": 0,
            "entity2_wins": 0,
            "win_rate_entity1": 0.0,
        }

    entity1_wins = direct_matches.filter(
        pl.col(winner_col) == entity1_id
    ).height
    entity2_wins = direct_matches.filter(
        pl.col(winner_col) == entity2_id
    ).height
    total = direct_matches.height

    return {
        "total_matches": total,
        "entity1_wins": entity1_wins,
        "entity2_wins": entity2_wins,
        "win_rate_entity1": entity1_wins / total if total > 0 else 0.0,
    }
