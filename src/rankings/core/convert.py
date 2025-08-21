"""Data conversion utilities for ranking algorithms."""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl


def convert_matches_dataframe(
    matches: pl.DataFrame,
    players: pl.DataFrame,
    tournament_influence: Dict[int, float],
    now_ts: float,
    decay_rate: float,
    beta: float = 0.0,
    *,
    rosters: Optional[pl.DataFrame] = None,
    include_share: bool = True,
    streaming: bool = False,
) -> pl.DataFrame:
    """
    Build a compact matches table with winners/losers roster lists and weights.

    This is the optimized Polars-based conversion path.

    Args:
        matches: Match data with tournament_id, winner_team_id, loser_team_id
        players: Player/roster data mapping users to teams
        tournament_influence: Tournament ID to influence score mapping
        now_ts: Current timestamp for decay calculations
        decay_rate: Time decay rate
        beta: Tournament influence exponent
        rosters: Optional pre-filtered roster data
        include_share: Whether to include share calculations
        streaming: Whether to use streaming mode

    Returns:
        DataFrame with columns:
        - match_id, tournament_id
        - winners (list of user_ids)
        - losers (list of user_ids)
        - weight (float)
        - ts (timestamp)
        - If include_share=True: wlen, llen, share
    """
    # Select only needed columns
    needed = [
        "match_id",
        "tournament_id",
        "winner_team_id",
        "loser_team_id",
    ]

    # Add timestamp columns if available
    for col in ["last_game_finished_at", "match_created_at"]:
        if col in matches.columns:
            needed.append(col)

    if "is_bye" in matches.columns:
        needed.append("is_bye")

    m = matches.select([c for c in needed if c in matches.columns])

    # Filter out byes and null teams
    filt = (
        pl.col("winner_team_id").is_not_null()
        & pl.col("loser_team_id").is_not_null()
    )

    if "is_bye" in m.columns:
        filt = filt & ~pl.col("is_bye").fill_null(False)

    m = m.filter(filt)

    # Create timestamp expression with fallbacks
    ts_exprs = []
    if "last_game_finished_at" in m.columns:
        ts_exprs.append(pl.col("last_game_finished_at"))
    if "match_created_at" in m.columns:
        ts_exprs.append(pl.col("match_created_at"))
    ts_exprs.append(pl.lit(now_ts))
    ts_expr = pl.coalesce(ts_exprs).cast(pl.Int64)

    m = m.with_columns(ts_expr.alias("ts"))

    # Add tournament influence
    if tournament_influence:
        s_df = pl.DataFrame(
            {
                "tournament_id": list(tournament_influence.keys()),
                "S": list(tournament_influence.values()),
            }
        )
        m = m.join(s_df, on="tournament_id", how="left").with_columns(
            pl.col("S").fill_null(1.0)
        )
    else:
        m = m.with_columns(pl.lit(1.0).alias("S"))

    # Compute weight: exp(-decay_rate * age_days) * (S ** beta)
    time_decay = (
        ((pl.lit(now_ts) - pl.col("ts").cast(pl.Float64)) / 86400.0)
        .mul(-decay_rate)
        .exp()
    )

    if beta == 0.0:
        weight_expr = time_decay
    else:
        weight_expr = time_decay * (pl.col("S") ** beta)

    m = m.with_columns(weight_expr.alias("weight"))

    # Handle rosters
    if rosters is None:
        # Use all teams from matches
        used_teams = pl.concat(
            [
                m.select(pl.col("winner_team_id").alias("team_id")),
                m.select(pl.col("loser_team_id").alias("team_id")),
            ]
        ).unique()

        rosters = players.join(
            used_teams, left_on="team_id", right_on="team_id", how="inner"
        ).select(["tournament_id", "team_id", "user_id"])

    # Group rosters by team
    roster_lists = rosters.group_by(["tournament_id", "team_id"]).agg(
        pl.col("user_id").alias("roster")
    )

    # Join winner rosters
    m = m.join(
        roster_lists,
        left_on=["tournament_id", "winner_team_id"],
        right_on=["tournament_id", "team_id"],
        how="left",
    ).rename({"roster": "winners"})

    # Join loser rosters
    m = m.join(
        roster_lists,
        left_on=["tournament_id", "loser_team_id"],
        right_on=["tournament_id", "team_id"],
        how="left",
    ).rename({"roster": "losers"})

    # Filter out matches with null rosters
    m = m.filter(
        pl.col("winners").is_not_null() & pl.col("losers").is_not_null()
    )

    # Add share calculations if requested
    if include_share:
        m = m.with_columns(
            [
                pl.col("winners").list.len().alias("wlen"),
                pl.col("losers").list.len().alias("llen"),
            ]
        )
        m = m.with_columns(
            (pl.col("weight") / (pl.col("wlen") * pl.col("llen"))).alias(
                "share"
            )
        )

    # Select final columns
    cols = ["match_id", "tournament_id", "winners", "losers", "weight", "ts"]
    if include_share:
        cols.extend(["wlen", "llen", "share"])

    return m.select(cols)


def convert_matches_format(
    matches: pl.DataFrame,
    players: pl.DataFrame,
    tournament_influence: Dict[int, float],
    now_ts: float,
    decay_rate: float,
    beta: float = 0.0,
) -> List[Dict]:
    """
    Convert polars DataFrame matches to list of dicts with winners/losers lists.

    This is the fallback/legacy conversion path.

    Args:
        matches: Match data
        players: Player/roster data
        tournament_influence: Tournament influence scores
        now_ts: Current timestamp
        decay_rate: Time decay rate
        beta: Tournament influence exponent

    Returns:
        List of match dictionaries
    """
    converted = []

    for row in matches.iter_rows(named=True):
        if row.get("is_bye", False):
            continue

        winner_team = row.get("winner_team_id")
        loser_team = row.get("loser_team_id")

        if not winner_team or not loser_team:
            continue

        tid = row["tournament_id"]

        # Get players from teams
        winner_players = players.filter(
            (pl.col("tournament_id") == tid)
            & (pl.col("team_id") == winner_team)
        )["user_id"].to_list()

        loser_players = players.filter(
            (pl.col("tournament_id") == tid) & (pl.col("team_id") == loser_team)
        )["user_id"].to_list()

        if not winner_players or not loser_players:
            continue

        # Compute match weight
        t_influence = tournament_influence.get(tid, 1.0)

        # Time decay
        if "last_game_finished_at" in row and row["last_game_finished_at"]:
            ts = row["last_game_finished_at"]
        elif "match_created_at" in row and row["match_created_at"]:
            ts = row["match_created_at"]
        else:
            ts = now_ts

        days_ago = (now_ts - ts) / 86400.0
        time_decay = math.exp(-decay_rate * days_ago)

        weight = time_decay * (t_influence**beta)

        converted.append(
            {
                "winners": winner_players,
                "losers": loser_players,
                "weight": weight,
                "tournament_id": tid,
                "match_id": row.get("match_id"),
                "timestamp": ts,
            }
        )

    return converted


def convert_team_matches(
    matches: pl.DataFrame,
    tournament_influence: Dict[int, float],
    now_ts: float,
    decay_rate: float,
    beta: float = 0.0,
) -> List[Dict]:
    """
    Convert team matches to required format.

    Args:
        matches: Match data with team IDs
        tournament_influence: Tournament influence scores
        now_ts: Current timestamp
        decay_rate: Time decay rate
        beta: Tournament influence exponent

    Returns:
        List of match dictionaries with team IDs as single-element lists
    """
    converted = []

    for row in matches.iter_rows(named=True):
        if row.get("is_bye", False):
            continue

        winner_team = row.get("winner_team_id")
        loser_team = row.get("loser_team_id")

        if not winner_team or not loser_team:
            continue

        tid = row["tournament_id"]
        t_influence = tournament_influence.get(tid, 1.0)

        # Time decay
        if "last_game_finished_at" in row and row["last_game_finished_at"]:
            ts = row["last_game_finished_at"]
        elif "match_created_at" in row and row["match_created_at"]:
            ts = row["match_created_at"]
        else:
            ts = now_ts

        days_ago = (now_ts - ts) / 86400.0
        time_decay = math.exp(-decay_rate * days_ago)

        weight = time_decay * (t_influence**beta)

        converted.append(
            {
                "winners": [winner_team],
                "losers": [loser_team],
                "weight": weight,
                "tournament_id": tid,
                "match_id": row.get("match_id"),
                "timestamp": ts,
            }
        )

    return converted


def factorize_ids(
    ids: List,
) -> Tuple[List, Dict[object, int]]:
    """
    Convert list of IDs to indices.

    Args:
        ids: List of unique IDs

    Returns:
        Tuple of (unique_ids, id_to_index_mapping)
    """
    unique_ids = list(dict.fromkeys(ids))  # Preserve order, remove duplicates
    id_to_idx = {id_val: idx for idx, id_val in enumerate(unique_ids)}
    return unique_ids, id_to_idx


def build_node_mapping(
    matches_df: pl.DataFrame,
    winner_col: str = "winners",
    loser_col: str = "losers",
) -> Tuple[List, Dict]:
    """
    Build node ID to index mapping from matches DataFrame.

    Args:
        matches_df: DataFrame with winner/loser columns
        winner_col: Name of winner column
        loser_col: Name of loser column

    Returns:
        Tuple of (node_list, node_to_index_map)
    """
    # Extract all unique IDs
    if winner_col in matches_df.columns and loser_col in matches_df.columns:
        # Handle list columns
        if matches_df[winner_col].dtype == pl.List:
            winners = (
                matches_df.select(pl.col(winner_col).list.explode())[winner_col]
                .unique()
                .to_list()
            )
            losers = (
                matches_df.select(pl.col(loser_col).list.explode())[loser_col]
                .unique()
                .to_list()
            )
        else:
            winners = matches_df[winner_col].unique().to_list()
            losers = matches_df[loser_col].unique().to_list()

        all_ids = list(set(winners) | set(losers))
    else:
        raise ValueError(f"Columns {winner_col} or {loser_col} not found")

    return factorize_ids(all_ids)
