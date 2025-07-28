"""
Comparison functions for tournament ranking analysis.

This module provides functions to compare rankings between different
systems and analyze head-to-head records.
"""

from __future__ import annotations

import polars as pl


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
