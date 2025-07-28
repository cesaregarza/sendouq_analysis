"""
Match analysis functions for tournament rankings.

This module provides functions for analyzing individual matches and their
impact on player rankings.
"""

from __future__ import annotations

from typing import Optional

import polars as pl


def get_most_influential_matches(
    player_id: int,
    matches_df: pl.DataFrame,
    players_df: pl.DataFrame,
    engine: "RatingEngine",
    top_n: int = 10,
) -> dict[str, pl.DataFrame]:
    """
    Get the most influential wins and losses for a specific player.

    This function recreates the match weights using tournament influences
    to identify which specific matches had the most impact on a player's ranking.

    Parameters
    ----------
    player_id : int
        The user_id of the player to analyze
    matches_df : pl.DataFrame
        Matches DataFrame with tournament_id, winner_team_id, loser_team_id, etc.
    players_df : pl.DataFrame
        Players DataFrame with user_id, team_id, tournament_id
    engine : RatingEngine
        The RatingEngine instance that has already computed tournament influences
    top_n : int, optional
        Number of top matches to return for wins and losses

    Returns
    -------
    dict[str, pl.DataFrame]
        Dictionary with 'wins' and 'losses' DataFrames containing:
        - match details (tournament_id, opponent info, scores)
        - match_weight (time decay * tournament strength)
        - influence_rank (1 = most influential)
    """
    import math
    from datetime import timezone

    # Get tournament influences
    if (
        not hasattr(engine, "tournament_influence_")
        or engine.tournament_influence_ is None
    ):
        raise ValueError(
            "Engine must have computed tournament influences. Run rank_players() first."
        )

    tournament_influence = engine.tournament_influence_

    # Convert tournament influences to DataFrame for joining
    influence_df = pl.DataFrame(
        {
            "tournament_id": list(tournament_influence.keys()),
            "tournament_influence": list(tournament_influence.values()),
        }
    )

    # Get all matches with tournament influences
    matches_with_influence = matches_df.join(
        influence_df, on="tournament_id", how="left"
    ).fill_null(
        1.0
    )  # Default influence for unseen tournaments

    # Add time decay weight
    # First add event timestamp
    matches_with_influence = matches_with_influence.with_columns(
        pl.when(pl.col("last_game_finished_at").is_not_null())
        .then(pl.col("last_game_finished_at"))
        .otherwise(pl.col("match_created_at"))
        .fill_null(int(engine.now.timestamp()))
        .alias("event_ts")
    )

    # Calculate time decay
    matches_with_influence = matches_with_influence.with_columns(
        (
            (int(engine.now.timestamp()) - pl.col("event_ts").cast(pl.Float64))
            / 86400.0
        )
        .mul(-engine.decay_rate)
        .exp()
        .alias("time_decay")
    )

    # Calculate match weight (time decay * tournament strength^beta)
    matches_with_influence = matches_with_influence.with_columns(
        [
            (
                pl.col("time_decay")
                * (pl.col("tournament_influence") ** engine.beta)
            ).alias("match_weight"),
            (pl.col("tournament_id").cast(pl.Int64)),
            (pl.col("winner_team_id").cast(pl.Int64)),
            (pl.col("loser_team_id").cast(pl.Int64)),
        ]
    )

    # Get player's teams across all tournaments
    player_teams = players_df.filter(pl.col("user_id") == player_id).select(
        ["tournament_id", "team_id"]
    )

    # Find matches where player's team won
    wins = matches_with_influence.join(
        player_teams,
        left_on=["tournament_id", "winner_team_id"],
        right_on=["tournament_id", "team_id"],
        how="inner",
    ).filter(
        pl.col("loser_team_id").is_not_null()
    )  # Valid matches only

    # Find matches where player's team lost
    losses = matches_with_influence.join(
        player_teams,
        left_on=["tournament_id", "loser_team_id"],
        right_on=["tournament_id", "team_id"],
        how="inner",
    ).filter(
        pl.col("winner_team_id").is_not_null()
    )  # Valid matches only

    # Get opponent player info for wins
    if not wins.is_empty():
        # Get losing team players
        opponent_players_wins = players_df.group_by(
            ["tournament_id", "team_id"]
        ).agg(pl.col("username").str.concat(", ").alias("opponent_players"))

        wins = wins.join(
            opponent_players_wins,
            left_on=["tournament_id", "loser_team_id"],
            right_on=["tournament_id", "team_id"],
            how="left",
        )

        # Sort by match weight and add rank
        wins = (
            wins.sort("match_weight", descending=True)
            .with_columns(pl.arange(1, wins.height + 1).alias("influence_rank"))
            .select(
                [
                    "influence_rank",
                    "tournament_id",
                    "match_id",
                    "loser_team_id",
                    "opponent_players",
                    "team1_score",
                    "team2_score",
                    "total_games",
                    "match_weight",
                    "tournament_influence",
                    "time_decay",
                    "event_ts",
                ]
            )
            .head(top_n)
        )
    else:
        wins = pl.DataFrame()

    # Get opponent player info for losses
    if not losses.is_empty():
        # Get winning team players
        opponent_players_losses = players_df.group_by(
            ["tournament_id", "team_id"]
        ).agg(pl.col("username").str.concat(", ").alias("opponent_players"))

        losses = losses.join(
            opponent_players_losses,
            left_on=["tournament_id", "winner_team_id"],
            right_on=["tournament_id", "team_id"],
            how="left",
        )

        # Sort by match weight and add rank
        losses = (
            losses.sort("match_weight", descending=True)
            .with_columns(
                pl.arange(1, losses.height + 1).alias("influence_rank")
            )
            .select(
                [
                    "influence_rank",
                    "tournament_id",
                    "match_id",
                    "winner_team_id",
                    "opponent_players",
                    "team1_score",
                    "team2_score",
                    "total_games",
                    "match_weight",
                    "tournament_influence",
                    "time_decay",
                    "event_ts",
                ]
            )
            .head(top_n)
        )
    else:
        losses = pl.DataFrame()

    return {"wins": wins, "losses": losses}


def get_player_match_history(
    player_id: int,
    matches_df: pl.DataFrame,
    players_df: pl.DataFrame,
    teams_df: pl.DataFrame,
    tournament_data: list[dict],
    limit: Optional[int] = None,
) -> pl.DataFrame:
    """
    Get a player's complete match history with names and context.

    Parameters
    ----------
    player_id : int
        The user_id of the player
    matches_df : pl.DataFrame
        Matches DataFrame
    players_df : pl.DataFrame
        Players DataFrame
    teams_df : pl.DataFrame
        Teams DataFrame
    tournament_data : list[dict]
        Raw tournament data
    limit : int, optional
        Maximum number of matches to return

    Returns
    -------
    pl.DataFrame
        Player's match history with full context
    """
    from .names import create_match_summary_with_names

    # Get player's teams
    player_teams = players_df.filter(pl.col("user_id") == player_id).select(
        ["tournament_id", "team_id"]
    )

    # Get all matches for those teams
    wins = matches_df.join(
        player_teams,
        left_on=["tournament_id", "winner_team_id"],
        right_on=["tournament_id", "team_id"],
        how="inner",
    ).with_columns(pl.lit("win").alias("result"))

    losses = matches_df.join(
        player_teams,
        left_on=["tournament_id", "loser_team_id"],
        right_on=["tournament_id", "team_id"],
        how="inner",
    ).with_columns(pl.lit("loss").alias("result"))

    # Combine wins and losses
    all_matches = pl.concat([wins, losses])

    # Add names and context
    result = create_match_summary_with_names(
        all_matches, players_df, teams_df, tournament_data
    )

    # Sort by date (most recent first)
    result = result.sort("event_ts", descending=True)

    if limit:
        result = result.head(limit)

    return result
