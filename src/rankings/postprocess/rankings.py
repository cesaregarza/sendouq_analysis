"""
Post-processing utilities for ranking results.

This module provides engine-agnostic post-processing functions for
formatting and enhancing ranking DataFrames.
"""

from typing import List, Optional

import polars as pl


def post_process_rankings(
    rankings: pl.DataFrame,
    players_df: pl.DataFrame,
    min_tournaments: int = 3,
    rank_cutoffs: Optional[List[float]] = None,
    rank_labels: Optional[List[str]] = None,
    score_multiplier: float = 25.0,
    score_offset: float = 0.0,
    tournaments_df: Optional[pl.DataFrame] = None,
    id_column: str = "id",
    score_column: str = "score",
) -> pl.DataFrame:
    """
    Post-process rankings with tournament filtering and grade assignment.

    This is engine-agnostic and works with any ranking DataFrame that has
    player IDs and scores.

    Parameters
    ----------
    rankings : pl.DataFrame
        Raw rankings with at least 'id' (or specified id_column) and 'score' columns
    players_df : pl.DataFrame
        Players DataFrame with tournament_id, user_id, username
    min_tournaments : int, default=3
        Minimum tournaments required for inclusion
    rank_cutoffs : Optional[List[float]], default=None
        Score cutoffs for rank labels
    rank_labels : Optional[List[str]], default=None
        Labels for rank grades
    score_multiplier : float, default=25.0
        Multiplier for final score display
    score_offset : float, default=0.0
        Offset to add to scores before applying multiplier
    tournaments_df : Optional[pl.DataFrame], default=None
        Tournaments DataFrame with tournament_id and start_time columns
        If provided, adds last_active date for each player
    id_column : str, default="id"
        Name of the player ID column in rankings
    score_column : str, default="score"
        Name of the score column in rankings

    Returns
    -------
    pl.DataFrame
        Processed rankings with grades and filtered players
    """
    # Default rank cutoffs and labels if not provided
    if rank_cutoffs is None:
        rank_cutoffs = [-3, -1, 0, 1, 2, 3, 4, 5]
    if rank_labels is None:
        rank_labels = ["A-", "A", "A+", "S-", "S", "S+", "X", "X+", "X★"]

    # Normalize column names
    if id_column != "id":
        rankings = rankings.rename({id_column: "id"})

    # Handle both 'score' and 'player_rank' columns
    if score_column != "score":
        rankings = rankings.rename({score_column: "score"})
    elif "player_rank" in rankings.columns and "score" not in rankings.columns:
        # If we have player_rank but no score, use player_rank as score
        rankings = rankings.rename({"player_rank": "score"})

    # Prepare player statistics
    if tournaments_df is not None:
        # Join players with tournaments to get tournament dates
        players_with_dates = players_df.join(
            tournaments_df.select(["tournament_id", "start_time"]),
            on="tournament_id",
            how="left",
        )

        # Group by user and get the latest tournament date
        player_stats = (
            players_with_dates.group_by("user_id")
            .agg(
                [
                    pl.col("username").last(),
                    pl.col("tournament_id")
                    .n_unique()
                    .alias("tournament_count"),
                    pl.col("start_time").max().alias("last_active"),
                ]
            )
            .select(["user_id", "username", "tournament_count", "last_active"])
        )
    else:
        # Aggregation without last_active
        player_stats = (
            players_df.group_by("user_id")
            .agg(
                [
                    pl.col("username").last(),
                    pl.col("tournament_id")
                    .n_unique()
                    .alias("tournament_count"),
                ]
            )
            .select(["user_id", "username", "tournament_count"])
        )

    # Sort rankings by score and add raw rank
    rankings = rankings.sort("score", descending=True)
    rankings = rankings.with_columns(
        pl.arange(1, len(rankings) + 1).alias("raw_rank")
    )

    # Join with player statistics
    final_rankings = rankings.join(
        player_stats,
        left_on="id",
        right_on="user_id",
        how="left",
    )

    # Filter by minimum tournaments
    final_rankings = final_rankings.filter(
        pl.col("tournament_count") >= min_tournaments
    )

    # Re-rank after filtering
    final_rankings = final_rankings.sort("score", descending=True)
    final_rankings = final_rankings.with_columns(
        pl.arange(1, len(final_rankings) + 1).alias("rank")
    )

    # Calculate win/loss ratio if columns are available
    if "win_pr" in rankings.columns and "loss_pr" in rankings.columns:
        final_rankings = final_rankings.with_columns(
            [
                (
                    pl.col("win_pr") / pl.col("loss_pr").clip(lower_bound=1e-10)
                ).alias("win_loss_ratio"),
                (pl.col("win_pr") - pl.col("loss_pr")).alias("win_loss_diff"),
            ]
        )
    else:
        # Use score as proxy for win/loss ratio
        final_rankings = final_rankings.with_columns(
            [
                pl.col("score").exp().alias("win_loss_ratio"),
                pl.col("score").alias("win_loss_diff"),
            ]
        )

    # Add grade labels
    final_rankings = final_rankings.with_columns(
        pl.col("score")
        .cut(breaks=rank_cutoffs, labels=rank_labels)
        .alias("rank_label")
    )

    # Calculate display score
    final_rankings = final_rankings.with_columns(
        ((pl.col("score") + score_offset) * score_multiplier).alias(
            "display_score"
        )
    )

    # Select and order final columns
    output_cols = [
        "rank",
        "rank_label",
        "username",
        pl.col("id").alias("player_id"),
        "display_score",
        "score",
        "win_loss_ratio",
        "tournament_count",
    ]

    # Add optional columns if they exist
    if "last_active" in final_rankings.columns:
        output_cols.append("last_active")
    if "exposure" in final_rankings.columns:
        output_cols.insert(5, "exposure")
    if "win_pr" in final_rankings.columns:
        output_cols.insert(6, "win_pr")
    if "loss_pr" in final_rankings.columns:
        output_cols.insert(7, "loss_pr")

    final_rankings = final_rankings.select(output_cols)

    return final_rankings


def assign_percentile_grades(
    rankings: pl.DataFrame,
    percentile_cutoffs: Optional[List[float]] = None,
    grade_labels: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    Assign grades based on percentile rankings.

    Parameters
    ----------
    rankings : pl.DataFrame
        Rankings DataFrame with 'rank' column
    percentile_cutoffs : Optional[List[float]], default=None
        Percentile cutoffs for grades (0.0 to 1.0)
    grade_labels : Optional[List[str]], default=None
        Grade labels corresponding to percentile ranges

    Returns
    -------
    pl.DataFrame
        Rankings with percentile-based grade labels
    """
    if percentile_cutoffs is None:
        percentile_cutoffs = [0.003, 0.03, 0.10, 0.25, 0.50, 0.75, 0.90, 0.97]
    if grade_labels is None:
        grade_labels = ["X★", "X", "S+", "S", "S-", "A+", "A", "A-", "B"]

    total_players = len(rankings)

    # Calculate percentile for each player
    rankings = rankings.with_columns(
        (pl.col("rank") / total_players).alias("percentile")
    )

    # Assign grades based on percentiles
    rankings = rankings.with_columns(
        pl.col("percentile")
        .cut(breaks=percentile_cutoffs, labels=grade_labels)
        .alias("percentile_grade")
    )

    return rankings


def add_activity_decay(
    rankings: pl.DataFrame,
    last_active_col: str = "last_active",
    decay_delay_days: float = 30.0,
    decay_rate: float = 0.01,
    current_timestamp: Optional[float] = None,
) -> pl.DataFrame:
    """
    Apply inactivity decay to display scores.

    Parameters
    ----------
    rankings : pl.DataFrame
        Rankings with last_active column
    last_active_col : str, default="last_active"
        Column containing last activity timestamps
    decay_delay_days : float, default=30.0
        Days before decay starts
    decay_rate : float, default=0.01
        Daily decay rate after delay
    current_timestamp : Optional[float], default=None
        Current time (defaults to now)

    Returns
    -------
    pl.DataFrame
        Rankings with decayed display scores
    """
    if last_active_col not in rankings.columns:
        return rankings

    if current_timestamp is None:
        import time

        current_timestamp = time.time()

    # Calculate days inactive
    rankings = rankings.with_columns(
        ((current_timestamp - pl.col(last_active_col)) / 86400.0).alias(
            "days_inactive"
        )
    )

    # Apply decay only after delay period
    rankings = rankings.with_columns(
        pl.when(pl.col("days_inactive") > decay_delay_days)
        .then(
            pl.col("display_score")
            * (-(pl.col("days_inactive") - decay_delay_days) * decay_rate).exp()
        )
        .otherwise(pl.col("display_score"))
        .alias("display_score_decayed")
    )

    return rankings


def standardize_usernames(rankings: pl.DataFrame) -> pl.DataFrame:
    """
    Standardize usernames for display.

    Parameters
    ----------
    rankings : pl.DataFrame
        Rankings with username column

    Returns
    -------
    pl.DataFrame
        Rankings with standardized usernames
    """
    if "username" not in rankings.columns:
        return rankings

    # Basic standardization
    rankings = rankings.with_columns(
        pl.col("username")
        .str.strip_chars()  # Remove leading/trailing whitespace
        .str.replace_all(r"\s+", " ")  # Normalize internal whitespace
        .alias("username_display")
    )

    # Keep original username for reference
    rankings = rankings.rename({"username": "username_original"})
    rankings = rankings.rename({"username_display": "username"})

    return rankings
