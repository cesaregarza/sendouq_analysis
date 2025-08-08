"""
Name resolution functions for tournament ranking analysis.

This module provides functions to add human-readable names to DataFrames
containing various entity IDs (players, teams, tournaments).
"""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl


def add_player_names(
    df: pl.DataFrame,
    players_df: pl.DataFrame,
    user_id_col: str = "user_id",
    name_col: str = "player_name",
) -> pl.DataFrame:
    """
    Add player names to a DataFrame containing user IDs.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with user IDs to add names to
    players_df : pl.DataFrame
        Players DataFrame with user_id and username columns
    user_id_col : str, optional
        Name of the user ID column in df
    name_col : str, optional
        Name for the new player name column

    Returns
    -------
    pl.DataFrame
        DataFrame with player names added
    """
    # Get unique player names (latest occurrence)
    unique_players = players_df.group_by("user_id").agg(
        pl.col("username").last().alias("_player_name")
    )

    return df.join(
        unique_players, left_on=user_id_col, right_on="user_id", how="left"
    ).rename({"_player_name": name_col})


def add_team_names(
    df: pl.DataFrame,
    teams_df: pl.DataFrame,
    team_id_col: str = "team_id",
    name_col: str = "team_name",
) -> pl.DataFrame:
    """
    Add team names to a DataFrame containing team IDs.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with team IDs to add names to
    teams_df : pl.DataFrame
        Teams DataFrame with team_id and team_name columns
    team_id_col : str, optional
        Name of the team ID column in df
    name_col : str, optional
        Name for the new team name column

    Returns
    -------
    pl.DataFrame
        DataFrame with team names added
    """
    # Get unique team names (latest occurrence)
    unique_teams = teams_df.group_by("team_id").agg(
        pl.col("team_name").last().alias("_team_name")
    )

    return df.join(
        unique_teams, left_on=team_id_col, right_on="team_id", how="left"
    ).rename({"_team_name": name_col})


def add_tournament_names(
    df: pl.DataFrame,
    tournament_data: list[dict],
    tournament_id_col: str = "tournament_id",
    name_col: str = "tournament_name",
) -> pl.DataFrame:
    """
    Add tournament names to a DataFrame containing tournament IDs.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with tournament IDs to add names to
    tournament_data : list[dict]
        Raw tournament data from JSON
    tournament_id_col : str, optional
        Name of the tournament ID column in df
    name_col : str, optional
        Name for the new tournament name column

    Returns
    -------
    pl.DataFrame
        DataFrame with tournament names added
    """
    # Extract tournament names
    tournament_names = {}
    for entry in tournament_data:
        ctx = entry.get("tournament", {}).get("ctx", {})
        if "id" in ctx and "name" in ctx:
            tournament_names[ctx["id"]] = ctx["name"]

    # Create lookup DataFrame
    names_df = pl.DataFrame(
        {
            "tournament_id": list(tournament_names.keys()),
            "_tournament_name": list(tournament_names.values()),
        }
    )

    return df.join(
        names_df,
        left_on=tournament_id_col,
        right_on="tournament_id",
        how="left",
    ).rename({"_tournament_name": name_col})


def add_match_timestamps(
    df: pl.DataFrame,
    timestamp_col: str = "event_ts",
    datetime_col: str = "match_datetime",
    date_col: str = "match_date",
    include_relative: bool = True,
) -> pl.DataFrame:
    """
    Add human-readable timestamps to a DataFrame with Unix timestamps.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with Unix timestamp column
    timestamp_col : str, optional
        Name of the Unix timestamp column
    datetime_col : str, optional
        Name for the new datetime column
    date_col : str, optional
        Name for the new date column
    include_relative : bool, optional
        Whether to include a relative time column (e.g., "3 days ago")

    Returns
    -------
    pl.DataFrame
        DataFrame with human-readable timestamps added
    """
    result = df.with_columns(
        [
            # Convert Unix timestamp (seconds) to datetime
            pl.from_epoch(pl.col(timestamp_col), unit="s").alias(datetime_col),
            # Extract just the date
            pl.from_epoch(pl.col(timestamp_col), unit="s")
            .dt.date()
            .alias(date_col),
        ]
    )

    if include_relative:
        # Add relative time (days ago)
        now = datetime.now(timezone.utc)
        result = result.with_columns(
            ((pl.lit(int(now.timestamp())) - pl.col(timestamp_col)) / 86400.0)
            .round(0)
            .cast(pl.Int32)
            .alias("days_ago")
        )

    return result


def create_match_summary_with_names(
    matches_df: pl.DataFrame,
    players_df: pl.DataFrame,
    teams_df: pl.DataFrame,
    tournament_data: list[dict],
    include_timestamps: bool = True,
) -> pl.DataFrame:
    """
    Create a comprehensive match summary with all names included.

    Parameters
    ----------
    matches_df : pl.DataFrame
        Matches DataFrame
    players_df : pl.DataFrame
        Players DataFrame
    teams_df : pl.DataFrame
        Teams DataFrame
    tournament_data : list[dict]
        Raw tournament data
    include_timestamps : bool, optional
        Whether to include human-readable timestamps

    Returns
    -------
    pl.DataFrame
        Match summary with tournament, team, and player names
    """
    # Start with matches
    result = matches_df.select(
        [
            "match_id",
            "tournament_id",
            "winner_team_id",
            "loser_team_id",
            "team1_score",
            "team2_score",
            "total_games",
            "last_game_finished_at",
            "match_created_at",
        ]
    )

    # Add tournament names
    result = add_tournament_names(result, tournament_data)

    # Add team names
    result = add_team_names(
        result, teams_df, "winner_team_id", "winner_team_name"
    )
    result = add_team_names(
        result, teams_df, "loser_team_id", "loser_team_name"
    )

    # Add timestamps if requested
    if include_timestamps:
        # Create event timestamp
        result = result.with_columns(
            pl.when(pl.col("last_game_finished_at").is_not_null())
            .then(pl.col("last_game_finished_at"))
            .otherwise(pl.col("match_created_at"))
            .alias("event_ts")
        )
        result = add_match_timestamps(result)

    return result


def get_tournament_name_lookup(tournament_data: list[dict]) -> dict[int, str]:
    """
    Extract tournament ID to name mapping from tournament data.

    Parameters
    ----------
    tournament_data : list[dict]
        Raw tournament data from JSON

    Returns
    -------
    dict[int, str]
        Dictionary mapping tournament ID to tournament name
    """
    tournament_names = {}
    for entry in tournament_data:
        ctx = entry.get("tournament", {}).get("ctx", {})
        if "id" in ctx and "name" in ctx:
            tournament_names[ctx["id"]] = ctx["name"]

    return tournament_names
