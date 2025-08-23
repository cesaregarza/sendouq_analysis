"""Edge building utilities for ranking algorithms."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from typing import Any, Dict, Tuple


def build_player_edges(
    matches: pl.DataFrame,
    players: pl.DataFrame,
    tournament_influence: dict[int, float],
    now_timestamp: float,
    decay_rate: float,
    beta: float = 0.0,
    timestamp_column: str | None = None,
) -> pl.DataFrame:
    """Build player-level edges with tournament strength weighting.

    Args:
        matches: Match data.
        players: Player/roster data.
        tournament_influence: Tournament ID to influence mapping.
        now_timestamp: Current timestamp.
        decay_rate: Time decay rate.
        beta: Tournament influence exponent.
        timestamp_column: Optional timestamp column name. Defaults to None.

    Returns:
        Edge DataFrame with columns: loser_user_id, winner_user_id, weight_sum.
    """
    if matches.is_empty() or players.is_empty():
        return pl.DataFrame([])

    if tournament_influence:
        strength_dataframe = pl.DataFrame(
            {
                "tournament_id": list(tournament_influence.keys()),
                "tournament_strength": list(tournament_influence.values()),
            }
        )
    else:
        strength_dataframe = None

    filter_expression = (
        pl.col("winner_team_id").is_not_null()
        & pl.col("loser_team_id").is_not_null()
    )

    if "is_bye" in matches.columns:
        filter_expression = filter_expression & ~pl.col("is_bye").fill_null(
            False
        )

    match_data = matches.filter(filter_expression)

    if strength_dataframe is not None:
        match_data = match_data.join(
            strength_dataframe, on="tournament_id", how="left"
        ).fill_null(1.0)
    else:
        match_data = match_data.with_columns(
            pl.lit(1.0).alias("tournament_strength")
        )

    if timestamp_column and timestamp_column in match_data.columns:
        match_data = match_data.with_columns(
            pl.col(timestamp_column).alias("ts")
        )
    else:
        timestamp_expressions = []
        if "last_game_finished_at" in match_data.columns:
            timestamp_expressions.append(pl.col("last_game_finished_at"))
        if "match_created_at" in match_data.columns:
            timestamp_expressions.append(pl.col("match_created_at"))
        timestamp_expressions.append(pl.lit(now_timestamp))
        match_data = match_data.with_columns(
            pl.coalesce(timestamp_expressions).alias("ts")
        )

    time_decay_factor = (
        ((pl.lit(now_timestamp) - pl.col("ts").cast(pl.Float64)) / 86400.0)
        .mul(-decay_rate)
        .exp()
    )

    if beta == 0.0:
        match_weight = time_decay_factor
    else:
        match_weight = time_decay_factor * (
            pl.col("tournament_strength") ** beta
        )

    match_data = match_data.with_columns(match_weight.alias("match_weight"))

    # Player selection for joins
    player_selection = players.select(
        ["tournament_id", "team_id", "user_id"]
    ).with_columns(
        [
            pl.col("tournament_id").cast(pl.Int64),
            pl.col("team_id").cast(pl.Int64),
        ]
    )

    # Expand winning teams to winning players
    winners = (
        match_data.select(
            ["match_id", "tournament_id", "winner_team_id", "match_weight"]
        )
        .with_columns(
            [
                pl.col("tournament_id").cast(pl.Int64),
                pl.col("winner_team_id").cast(pl.Int64),
            ]
        )
        .join(
            player_selection,
            left_on=["tournament_id", "winner_team_id"],
            right_on=["tournament_id", "team_id"],
            how="inner",
        )
        .rename({"user_id": "winner_user_id"})
        .select(["match_id", "winner_user_id", "match_weight"])
    )

    # Expand losing teams to losing players
    losers = (
        match_data.select(
            ["match_id", "tournament_id", "loser_team_id", "match_weight"]
        )
        .with_columns(
            [
                pl.col("tournament_id").cast(pl.Int64),
                pl.col("loser_team_id").cast(pl.Int64),
            ]
        )
        .join(
            player_selection,
            left_on=["tournament_id", "loser_team_id"],
            right_on=["tournament_id", "team_id"],
            how="inner",
        )
        .rename({"user_id": "loser_user_id"})
        .select(["match_id", "loser_user_id", "match_weight"])
    )

    # Create player-to-player pairs
    pairs = losers.join(winners, on="match_id", how="inner").select(
        ["loser_user_id", "winner_user_id", "match_weight"]
    )

    if pairs.is_empty():
        return pl.DataFrame([])

    # Aggregate raw pair weights
    edges = pairs.group_by(["loser_user_id", "winner_user_id"]).agg(
        pl.col("match_weight").sum().alias("weight_sum")
    )

    return edges


def build_team_edges(
    matches: pl.DataFrame,
    tournament_influence: dict[int, float],
    now_timestamp: float,
    decay_rate: float,
    beta: float = 0.0,
) -> pl.DataFrame:
    """Build team-level edges from matches.

    Args:
        matches: Match data with team IDs.
        tournament_influence: Tournament ID to influence mapping.
        now_timestamp: Current timestamp.
        decay_rate: Time decay rate.
        beta: Tournament influence exponent.

    Returns:
        Edge DataFrame with columns: loser_team_id, winner_team_id, weight_sum.
    """
    if matches.is_empty():
        return pl.DataFrame([])

    # Filter out byes and null teams
    filter_expression = (
        pl.col("winner_team_id").is_not_null()
        & pl.col("loser_team_id").is_not_null()
    )

    if "is_bye" in matches.columns:
        filter_expression = filter_expression & ~pl.col("is_bye").fill_null(
            False
        )

    match_data = matches.filter(filter_expression)

    # Add tournament influence
    if tournament_influence:
        strength_dataframe = pl.DataFrame(
            {
                "tournament_id": list(tournament_influence.keys()),
                "tournament_strength": list(tournament_influence.values()),
            }
        )
        match_data = match_data.join(
            strength_dataframe, on="tournament_id", how="left"
        ).fill_null(1.0)
    else:
        match_data = match_data.with_columns(
            pl.lit(1.0).alias("tournament_strength")
        )

    # Add timestamp
    timestamp_expressions = []
    if "last_game_finished_at" in match_data.columns:
        timestamp_expressions.append(pl.col("last_game_finished_at"))
    if "match_created_at" in match_data.columns:
        timestamp_expressions.append(pl.col("match_created_at"))
    timestamp_expressions.append(pl.lit(now_timestamp))
    match_data = match_data.with_columns(
        pl.coalesce(timestamp_expressions).alias("ts")
    )

    # Compute match weight
    time_decay_factor = (
        ((pl.lit(now_timestamp) - pl.col("ts").cast(pl.Float64)) / 86400.0)
        .mul(-decay_rate)
        .exp()
    )

    if beta == 0.0:
        match_weight = time_decay_factor
    else:
        match_weight = time_decay_factor * (
            pl.col("tournament_strength") ** beta
        )

    match_data = match_data.with_columns(match_weight.alias("match_weight"))

    # Aggregate team-to-team edges
    edges = match_data.group_by(["loser_team_id", "winner_team_id"]).agg(
        pl.col("match_weight").sum().alias("weight_sum")
    )

    return edges


def build_exposure_triplets(
    matches_dataframe: pl.DataFrame,
    node_to_index: dict[Any, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Produce COO triplets (row=winner_idx, col=loser_idx, data=share).

    Uses list-explode to form the cartesian product winner×loser per match.

    Args:
        matches_dataframe: DataFrame with winners, losers, share columns.
        node_to_index: Mapping from node IDs to indices.

    Returns:
        Tuple of (row_indices, col_indices, weights).
    """
    # Explode both lists to pairs
    pairs = (
        matches_dataframe.explode("winners")
        .explode("losers")
        .select(
            [
                pl.col("winners").alias("winner_id"),
                pl.col("losers").alias("loser_id"),
                "share",
            ]
        )
    )

    # Map IDs to indices - ensure proper dtypes
    valid_items = [
        (key, value) for key, value in node_to_index.items() if key is not None
    ]
    if not valid_items:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
        )

    node_ids, node_indices = zip(*valid_items)
    index_mapping = pl.DataFrame(
        {"id": list(node_ids), "idx": list(node_indices)}
    )

    # Ensure winner_id and loser_id have matching dtypes with id column
    id_dtype = index_mapping["id"].dtype
    pairs = pairs.with_columns(
        [pl.col("winner_id").cast(id_dtype), pl.col("loser_id").cast(id_dtype)]
    )

    pairs = (
        pairs.join(
            index_mapping, left_on="winner_id", right_on="id", how="inner"
        )
        .rename({"idx": "winner_idx"})
        .join(index_mapping, left_on="loser_id", right_on="id", how="inner")
        .rename({"idx": "loser_idx"})
        .select(["winner_idx", "loser_idx", "share"])
    )

    # Aggregate duplicate pairs
    pairs = pairs.group_by(["winner_idx", "loser_idx"]).agg(
        pl.col("share").sum().alias("weight_sum")
    )

    # Convert to numpy arrays
    row_indices = pairs["winner_idx"].to_numpy()
    col_indices = pairs["loser_idx"].to_numpy()
    weights = pairs["weight_sum"].to_numpy()

    return row_indices, col_indices, weights


def compute_denominators(
    edges: pl.DataFrame,
    smoothing_strategy,
    loser_column: str = "loser_user_id",
    winner_column: str = "winner_user_id",
    weight_column: str = "weight_sum",
) -> pl.DataFrame:
    """Compute denominators with smoothing for edge normalization.

    Args:
        edges: Edge DataFrame.
        smoothing_strategy: Smoothing strategy object with denom() method.
        loser_column: Column name for loser/source nodes.
        winner_column: Column name for winner/target nodes.
        weight_column: Column name for edge weights.

    Returns:
        DataFrame with columns: node_id, loss_weights, win_weights, denom, lambda.
    """
    # Compute loss mass (outgoing edges)
    loss_totals = edges.group_by(loser_column).agg(
        pl.col(weight_column).sum().alias("loss_weights")
    )

    # Compute win mass (incoming edges)
    win_totals = (
        edges.group_by(winner_column)
        .agg(pl.col(weight_column).sum().alias("win_weights"))
        .rename({winner_column: loser_column})
    )

    # Join and fill nulls
    denominators = loss_totals.join(
        win_totals, on=loser_column, how="left"
    ).with_columns(pl.col("win_weights").fill_null(0.0))

    # Apply smoothing strategy
    if smoothing_strategy:
        loss_weights_array = denominators["loss_weights"].to_numpy()
        win_weights_array = denominators["win_weights"].to_numpy()
        denominator_values = smoothing_strategy.denom(
            loss_weights_array, win_weights_array
        )

        denominators = denominators.with_columns(
            pl.Series("denom", denominator_values)
        )
    else:
        denominators = denominators.with_columns(
            pl.col("loss_weights").alias("denom")
        )

    # Compute lambda (smoothing term)
    denominators = denominators.with_columns(
        (pl.col("denom") - pl.col("loss_weights")).alias("lambda")
    )

    return denominators


def normalize_edges(
    edges: pl.DataFrame,
    denominators: pl.DataFrame,
    loser_column: str = "loser_user_id",
    weight_column: str = "weight_sum",
) -> pl.DataFrame:
    """Normalize edge weights by denominators.

    Args:
        edges: Edge DataFrame.
        denominators: Denominator DataFrame.
        loser_column: Column name for source nodes.
        weight_column: Column name for weights.

    Returns:
        Normalized edge DataFrame.
    """
    # Join denominators
    edges = edges.join(
        denominators.select([loser_column, "denom"]),
        on=loser_column,
        how="left",
    )

    # Normalize weights
    edges = edges.with_columns(
        (pl.col(weight_column) / pl.col("denom")).alias("normalized_weight")
    )

    return edges


def edges_to_triplets(
    edges: pl.DataFrame,
    node_to_index: dict[Any, int],
    source_column: str,
    target_column: str,
    weight_column: str = "weight_sum",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert edge DataFrame to COO triplets.

    Args:
        edges: Edge DataFrame.
        node_to_index: Node ID to index mapping.
        source_column: Source node column.
        target_column: Target node column.
        weight_column: Weight column.

    Returns:
        Tuple of (row_indices, col_indices, weights).
    """
    # Map node IDs to indices
    index_mapping = pl.DataFrame(
        {"id": list(node_to_index.keys()), "idx": list(node_to_index.values())}
    )

    # Join to get indices
    edges_with_indices = (
        edges.join(
            index_mapping, left_on=source_column, right_on="id", how="inner"
        )
        .rename({"idx": "source_idx"})
        .join(index_mapping, left_on=target_column, right_on="id", how="inner")
        .rename({"idx": "target_idx"})
    )

    # Extract arrays
    row_indices = edges_with_indices["source_idx"].to_numpy()
    col_indices = edges_with_indices["target_idx"].to_numpy()
    weights = edges_with_indices[weight_column].to_numpy()

    return row_indices, col_indices, weights
