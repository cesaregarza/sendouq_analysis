"""Data conversion utilities for ranking algorithms."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from typing import Any


def convert_matches_dataframe(
    matches: pl.DataFrame,
    players: pl.DataFrame,
    tournament_influence: dict[int, float],
    now_timestamp: float,
    decay_rate: float,
    beta: float = 0.0,
    *,
    rosters: pl.DataFrame | None = None,
    appearances: pl.DataFrame | None = None,
    include_share: bool = True,
    streaming: bool = False,
) -> pl.DataFrame:
    """Build a compact matches table with winners/losers roster lists and weights.

    This is the optimized Polars-based conversion path.

    Args:
        matches: Match data with tournament_id, winner_team_id, loser_team_id.
        players: Player/roster data mapping users to teams.
        tournament_influence: Tournament ID to influence score mapping.
        now_timestamp: Current timestamp for decay calculations.
        decay_rate: Time decay rate.
        beta: Tournament influence exponent.
        rosters: Optional pre-filtered roster data. Defaults to None.
        appearances: Optional per-match player appearances. Supports two schemas:
            - With team context: columns [tournament_id, match_id, team_id, user_id]
            - Player-only: columns [tournament_id, match_id, user_id] (team inferred from rosters)
        include_share: Whether to include share calculations. Defaults to True.
        streaming: Whether to use streaming mode. Defaults to False.

    Returns:
        DataFrame with columns: match_id, tournament_id, winners (list of user_ids),
        losers (list of user_ids), weight (float), ts (timestamp). If include_share=True,
        also includes winner_count, loser_count, share.
    """
    needed_columns = [
        "match_id",
        "tournament_id",
        "winner_team_id",
        "loser_team_id",
    ]

    for column_name in ["last_game_finished_at", "match_created_at"]:
        if column_name in matches.columns:
            needed_columns.append(column_name)

    if "is_bye" in matches.columns:
        needed_columns.append("is_bye")

    match_data = matches.select(
        [c for c in needed_columns if c in matches.columns]
    )

    filter_condition = (
        pl.col("winner_team_id").is_not_null()
        & pl.col("loser_team_id").is_not_null()
    )

    if "is_bye" in match_data.columns:
        filter_condition = filter_condition & ~pl.col("is_bye").fill_null(False)

    match_data = match_data.filter(filter_condition)

    timestamp_expressions = []
    if "last_game_finished_at" in match_data.columns:
        timestamp_expressions.append(pl.col("last_game_finished_at"))
    if "match_created_at" in match_data.columns:
        timestamp_expressions.append(pl.col("match_created_at"))
    timestamp_expressions.append(pl.lit(now_timestamp))
    timestamp_expr = pl.coalesce(timestamp_expressions).cast(pl.Int64)

    match_data = match_data.with_columns(timestamp_expr.alias("ts"))

    if tournament_influence:
        strength_dataframe = pl.DataFrame(
            {
                "tournament_id": list(tournament_influence.keys()),
                "tournament_strength": list(tournament_influence.values()),
            }
        )
        match_data = match_data.join(
            strength_dataframe, on="tournament_id", how="left"
        ).with_columns(pl.col("tournament_strength").fill_null(1.0))
    else:
        match_data = match_data.with_columns(
            pl.lit(1.0).alias("tournament_strength")
        )

    time_decay_factor = (
        ((pl.lit(now_timestamp) - pl.col("ts").cast(pl.Float64)) / 86400.0)
        .mul(-decay_rate)
        .exp()
    )

    if beta == 0.0:
        weight_expression = time_decay_factor
    else:
        weight_expression = time_decay_factor * (
            pl.col("tournament_strength") ** beta
        )

    match_data = match_data.with_columns(weight_expression.alias("weight"))

    # Build baseline roster lists (team-level) for fallback
    if rosters is None:
        used_teams = pl.concat(
            [
                match_data.select(pl.col("winner_team_id").alias("team_id")),
                match_data.select(pl.col("loser_team_id").alias("team_id")),
            ]
        ).unique()

        rosters = players.join(
            used_teams, left_on="team_id", right_on="team_id", how="inner"
        ).select(["tournament_id", "team_id", "user_id"])

    roster_lists = rosters.group_by(["tournament_id", "team_id"]).agg(
        pl.col("user_id").alias("roster")
    )

    # Optional: override winners/losers with actual per-match appearances when available
    winners_col = "winners"
    losers_col = "losers"
    if appearances is not None and not appearances.is_empty():
        # Ensure dtypes and group to per-match team lists
        # Accept appearances missing team_id (derive from rosters)
        base_cols = [
            pl.col("tournament_id").cast(pl.Int64).alias("tournament_id"),
            pl.col("match_id").cast(pl.Int64).alias("match_id"),
            pl.col("user_id").cast(pl.Int64).alias("user_id"),
        ]
        has_team_id = "team_id" in appearances.columns
        if has_team_id:
            base_cols.append(pl.col("team_id").cast(pl.Int64).alias("team_id"))
        appearances_norm = appearances.select(base_cols)

        # Derive team_id from rosters if missing (or null)
        if (not has_team_id) or appearances_norm.select(
            pl.col("team_id").is_null().any()
        ).item():
            # Join on (tournament_id, user_id) to fetch team_id from rosters
            appearances_norm = appearances_norm.join(
                rosters, on=["tournament_id", "user_id"], how="left"
            )
            # After join, ensure column name is 'team_id' (normalize suffixes)
            if "team_id_right" in appearances_norm.columns:
                appearances_norm = appearances_norm.with_columns(
                    pl.coalesce(
                        [pl.col("team_id"), pl.col("team_id_right")]
                    ).alias("team_id")
                ).drop(
                    [
                        c
                        for c in ["team_id_right"]
                        if c in appearances_norm.columns
                    ]
                )

        # Now group by (tournament_id, match_id, team_id)
        appearance_lists_df = (
            appearances_norm.drop_nulls(["team_id"])
            .group_by(["tournament_id", "match_id", "team_id"])
            .agg(pl.col("user_id").alias("played"))
        )

        # Join baseline rosters to compute fallbacks if appearance missing
        match_data = match_data.join(
            roster_lists,
            left_on=["tournament_id", "winner_team_id"],
            right_on=["tournament_id", "team_id"],
            how="left",
        ).rename({"roster": "_w_roster"})

        match_data = match_data.join(
            roster_lists,
            left_on=["tournament_id", "loser_team_id"],
            right_on=["tournament_id", "team_id"],
            how="left",
        ).rename({"roster": "_l_roster"})

        # Attach appearances for winners and losers
        match_data = match_data.join(
            appearance_lists_df,
            left_on=["tournament_id", "match_id", "winner_team_id"],
            right_on=["tournament_id", "match_id", "team_id"],
            how="left",
        ).rename({"played": "_w_played"})
        match_data = match_data.join(
            appearance_lists_df,
            left_on=["tournament_id", "match_id", "loser_team_id"],
            right_on=["tournament_id", "match_id", "team_id"],
            how="left",
        ).rename({"played": "_l_played"})

        # Coalesce: prefer played-over-roster
        match_data = match_data.with_columns(
            [
                pl.coalesce([pl.col("_w_played"), pl.col("_w_roster")]).alias(
                    winners_col
                ),
                pl.coalesce([pl.col("_l_played"), pl.col("_l_roster")]).alias(
                    losers_col
                ),
            ]
        )
        # Drop temp columns
        match_data = match_data.drop(
            [
                c
                for c in ["_w_roster", "_l_roster", "_w_played", "_l_played"]
                if c in match_data.columns
            ]
        )
    else:
        # No appearances provided; use team rosters
        match_data = match_data.join(
            roster_lists,
            left_on=["tournament_id", "winner_team_id"],
            right_on=["tournament_id", "team_id"],
            how="left",
        ).rename({"roster": winners_col})

        match_data = match_data.join(
            roster_lists,
            left_on=["tournament_id", "loser_team_id"],
            right_on=["tournament_id", "team_id"],
            how="left",
        ).rename({"roster": losers_col})

    match_data = match_data.filter(
        pl.col(winners_col).is_not_null() & pl.col(losers_col).is_not_null()
    )

    if include_share:
        match_data = match_data.with_columns(
            [
                pl.col(winners_col).list.len().alias("winner_count"),
                pl.col(losers_col).list.len().alias("loser_count"),
            ]
        )
        match_data = match_data.with_columns(
            (
                pl.col("weight")
                / (pl.col("winner_count") * pl.col("loser_count"))
            ).alias("share")
        )

    final_columns = [
        "match_id",
        "tournament_id",
        winners_col,
        losers_col,
        "weight",
        "ts",
    ]
    if include_share:
        final_columns.extend(["winner_count", "loser_count", "share"])

    return match_data.select(final_columns)


def convert_matches_format(
    matches: pl.DataFrame,
    players: pl.DataFrame,
    tournament_influence: dict[int, float],
    now_timestamp: float,
    decay_rate: float,
    beta: float = 0.0,
) -> list[dict[str, Any]]:
    """Convert polars DataFrame matches to list of dicts with winners/losers lists.

    This is the fallback/legacy conversion path.

    Args:
        matches: Match data.
        players: Player/roster data.
        tournament_influence: Tournament influence scores.
        now_timestamp: Current timestamp.
        decay_rate: Time decay rate.
        beta: Tournament influence exponent.

    Returns:
        List of match dictionaries.
    """
    converted = []

    for row in matches.iter_rows(named=True):
        if row.get("is_bye", False):
            continue

        winner_team_id = row.get("winner_team_id")
        loser_team_id = row.get("loser_team_id")

        if not winner_team_id or not loser_team_id:
            continue

        tournament_id = row["tournament_id"]

        winner_player_ids = players.filter(
            (pl.col("tournament_id") == tournament_id)
            & (pl.col("team_id") == winner_team_id)
        )["user_id"].to_list()

        loser_player_ids = players.filter(
            (pl.col("tournament_id") == tournament_id)
            & (pl.col("team_id") == loser_team_id)
        )["user_id"].to_list()

        if not winner_player_ids or not loser_player_ids:
            continue

        tournament_strength = tournament_influence.get(tournament_id, 1.0)

        if "last_game_finished_at" in row and row["last_game_finished_at"]:
            timestamp = row["last_game_finished_at"]
        elif "match_created_at" in row and row["match_created_at"]:
            timestamp = row["match_created_at"]
        else:
            timestamp = now_timestamp

        days_ago = (now_timestamp - timestamp) / 86400.0
        time_decay_factor = math.exp(-decay_rate * days_ago)

        weight = time_decay_factor * (tournament_strength**beta)

        converted.append(
            {
                "winners": winner_player_ids,
                "losers": loser_player_ids,
                "weight": weight,
                "tournament_id": tournament_id,
                "match_id": row.get("match_id"),
                "timestamp": timestamp,
            }
        )

    return converted


def convert_team_matches(
    matches: pl.DataFrame,
    tournament_influence: dict[int, float],
    now_timestamp: float,
    decay_rate: float,
    beta: float = 0.0,
) -> list[dict[str, Any]]:
    """Convert team matches to required format.

    Args:
        matches: Match data with team IDs.
        tournament_influence: Tournament influence scores.
        now_timestamp: Current timestamp.
        decay_rate: Time decay rate.
        beta: Tournament influence exponent.

    Returns:
        List of match dictionaries with team IDs as single-element lists.
    """
    converted = []

    for row in matches.iter_rows(named=True):
        if row.get("is_bye", False):
            continue

        winner_team_id = row.get("winner_team_id")
        loser_team_id = row.get("loser_team_id")

        if not winner_team_id or not loser_team_id:
            continue

        tournament_id = row["tournament_id"]
        tournament_strength = tournament_influence.get(tournament_id, 1.0)

        if "last_game_finished_at" in row and row["last_game_finished_at"]:
            timestamp = row["last_game_finished_at"]
        elif "match_created_at" in row and row["match_created_at"]:
            timestamp = row["match_created_at"]
        else:
            timestamp = now_timestamp

        days_ago = (now_timestamp - timestamp) / 86400.0
        time_decay_factor = math.exp(-decay_rate * days_ago)

        weight = time_decay_factor * (tournament_strength**beta)

        converted.append(
            {
                "winners": [winner_team_id],
                "losers": [loser_team_id],
                "weight": weight,
                "tournament_id": tournament_id,
                "match_id": row.get("match_id"),
                "timestamp": timestamp,
            }
        )

    return converted


def factorize_ids(
    node_ids: list[Any],
) -> tuple[list[Any], dict[Any, int]]:
    """Convert list of IDs to indices.

    Args:
        node_ids: List of unique IDs.

    Returns:
        Tuple of (unique_ids, id_to_index_mapping).
    """
    unique_ids = list(dict.fromkeys(node_ids))
    id_to_index = {node_id: index for index, node_id in enumerate(unique_ids)}
    return unique_ids, id_to_index


def build_node_mapping(
    matches_dataframe: pl.DataFrame,
    winner_column: str = "winners",
    loser_column: str = "losers",
) -> tuple[list[Any], dict[Any, int]]:
    """Build node ID to index mapping from matches DataFrame.

    Args:
        matches_dataframe: DataFrame with winner/loser columns.
        winner_column: Name of winner column. Defaults to "winners".
        loser_column: Name of loser column. Defaults to "losers".

    Returns:
        Tuple of (node_list, node_to_index_map).
    """
    if (
        winner_column in matches_dataframe.columns
        and loser_column in matches_dataframe.columns
    ):
        if matches_dataframe[winner_column].dtype == pl.List:
            winner_ids = (
                matches_dataframe.select(pl.col(winner_column).list.explode())[
                    winner_column
                ]
                .unique()
                .to_list()
            )
            loser_ids = (
                matches_dataframe.select(pl.col(loser_column).list.explode())[
                    loser_column
                ]
                .unique()
                .to_list()
            )
        else:
            winner_ids = matches_dataframe[winner_column].unique().to_list()
            loser_ids = matches_dataframe[loser_column].unique().to_list()

        all_node_ids = list(set(winner_ids) | set(loser_ids))
    else:
        raise ValueError(f"Columns {winner_column} or {loser_column} not found")

    return factorize_ids(all_node_ids)
