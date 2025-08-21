"""Edge building utilities for ranking algorithms."""

import math
from typing import Dict, Optional, Tuple

import numpy as np
import polars as pl


def build_player_edges(
    matches: pl.DataFrame,
    players: pl.DataFrame,
    tournament_influence: Dict[int, float],
    now_ts: float,
    decay_rate: float,
    beta: float = 0.0,
    ts_col: Optional[str] = None,
) -> pl.DataFrame:
    """
    Build player-level edges with tournament strength weighting.

    Args:
        matches: Match data
        players: Player/roster data
        tournament_influence: Tournament ID to influence mapping
        now_ts: Current timestamp
        decay_rate: Time decay rate
        beta: Tournament influence exponent
        ts_col: Optional timestamp column name

    Returns:
        Edge DataFrame with columns: loser_user_id, winner_user_id, w_sum
    """
    if matches.is_empty() or players.is_empty():
        return pl.DataFrame([])

    # Tournament strength lookup
    if tournament_influence:
        s_df = pl.DataFrame(
            {
                "tournament_id": list(tournament_influence.keys()),
                "S": list(tournament_influence.values()),
            }
        )
    else:
        s_df = None

    # Filter out byes and null teams
    filter_expr = (
        pl.col("winner_team_id").is_not_null()
        & pl.col("loser_team_id").is_not_null()
    )

    if "is_bye" in matches.columns:
        filter_expr = filter_expr & ~pl.col("is_bye").fill_null(False)

    m = matches.filter(filter_expr)

    # Join tournament influence
    if s_df is not None:
        m = m.join(s_df, on="tournament_id", how="left").fill_null(1.0)
    else:
        m = m.with_columns(pl.lit(1.0).alias("S"))

    # Add timestamp
    if ts_col and ts_col in m.columns:
        m = m.with_columns(pl.col(ts_col).alias("ts"))
    else:
        # Use fallback timestamp logic
        ts_exprs = []
        if "last_game_finished_at" in m.columns:
            ts_exprs.append(pl.col("last_game_finished_at"))
        if "match_created_at" in m.columns:
            ts_exprs.append(pl.col("match_created_at"))
        ts_exprs.append(pl.lit(now_ts))
        m = m.with_columns(pl.coalesce(ts_exprs).alias("ts"))

    # Compute match weight with decay and tournament influence
    time_decay = (
        ((pl.lit(now_ts) - pl.col("ts").cast(pl.Float64)) / 86400.0)
        .mul(-decay_rate)
        .exp()
    )

    if beta == 0.0:
        match_weight = time_decay
    else:
        match_weight = time_decay * (pl.col("S") ** beta)

    m = m.with_columns(match_weight.alias("match_w"))

    # Player selection for joins
    pl_sel = players.select(
        ["tournament_id", "team_id", "user_id"]
    ).with_columns(
        [
            pl.col("tournament_id").cast(pl.Int64),
            pl.col("team_id").cast(pl.Int64),
        ]
    )

    # Expand winning teams to winning players
    winners = (
        m.select(["match_id", "tournament_id", "winner_team_id", "match_w"])
        .with_columns(
            [
                pl.col("tournament_id").cast(pl.Int64),
                pl.col("winner_team_id").cast(pl.Int64),
            ]
        )
        .join(
            pl_sel,
            left_on=["tournament_id", "winner_team_id"],
            right_on=["tournament_id", "team_id"],
            how="inner",
        )
        .rename({"user_id": "winner_user_id"})
        .select(["match_id", "winner_user_id", "match_w"])
    )

    # Expand losing teams to losing players
    losers = (
        m.select(["match_id", "tournament_id", "loser_team_id", "match_w"])
        .with_columns(
            [
                pl.col("tournament_id").cast(pl.Int64),
                pl.col("loser_team_id").cast(pl.Int64),
            ]
        )
        .join(
            pl_sel,
            left_on=["tournament_id", "loser_team_id"],
            right_on=["tournament_id", "team_id"],
            how="inner",
        )
        .rename({"user_id": "loser_user_id"})
        .select(["match_id", "loser_user_id", "match_w"])
    )

    # Create player-to-player pairs
    pairs = losers.join(winners, on="match_id", how="inner").select(
        ["loser_user_id", "winner_user_id", "match_w"]
    )

    if pairs.is_empty():
        return pl.DataFrame([])

    # Aggregate raw pair weights
    edges = pairs.group_by(["loser_user_id", "winner_user_id"]).agg(
        pl.col("match_w").sum().alias("w_sum")
    )

    return edges


def build_team_edges(
    matches: pl.DataFrame,
    tournament_influence: Dict[int, float],
    now_ts: float,
    decay_rate: float,
    beta: float = 0.0,
) -> pl.DataFrame:
    """
    Build team-level edges from matches.

    Args:
        matches: Match data with team IDs
        tournament_influence: Tournament ID to influence mapping
        now_ts: Current timestamp
        decay_rate: Time decay rate
        beta: Tournament influence exponent

    Returns:
        Edge DataFrame with columns: loser_team_id, winner_team_id, w_sum
    """
    if matches.is_empty():
        return pl.DataFrame([])

    # Filter out byes and null teams
    filter_expr = (
        pl.col("winner_team_id").is_not_null()
        & pl.col("loser_team_id").is_not_null()
    )

    if "is_bye" in matches.columns:
        filter_expr = filter_expr & ~pl.col("is_bye").fill_null(False)

    m = matches.filter(filter_expr)

    # Add tournament influence
    if tournament_influence:
        s_df = pl.DataFrame(
            {
                "tournament_id": list(tournament_influence.keys()),
                "S": list(tournament_influence.values()),
            }
        )
        m = m.join(s_df, on="tournament_id", how="left").fill_null(1.0)
    else:
        m = m.with_columns(pl.lit(1.0).alias("S"))

    # Add timestamp
    ts_exprs = []
    if "last_game_finished_at" in m.columns:
        ts_exprs.append(pl.col("last_game_finished_at"))
    if "match_created_at" in m.columns:
        ts_exprs.append(pl.col("match_created_at"))
    ts_exprs.append(pl.lit(now_ts))
    m = m.with_columns(pl.coalesce(ts_exprs).alias("ts"))

    # Compute match weight
    time_decay = (
        ((pl.lit(now_ts) - pl.col("ts").cast(pl.Float64)) / 86400.0)
        .mul(-decay_rate)
        .exp()
    )

    if beta == 0.0:
        match_weight = time_decay
    else:
        match_weight = time_decay * (pl.col("S") ** beta)

    m = m.with_columns(match_weight.alias("match_w"))

    # Aggregate team-to-team edges
    edges = m.group_by(["loser_team_id", "winner_team_id"]).agg(
        pl.col("match_w").sum().alias("w_sum")
    )

    return edges


def build_exposure_triplets(
    matches_df: pl.DataFrame,
    node_to_idx: Dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Produce COO triplets (row=winner_idx, col=loser_idx, data=share).

    Uses list-explode to form the cartesian product winner×loser per match.

    Args:
        matches_df: DataFrame with winners, losers, share columns
        node_to_idx: Mapping from node IDs to indices

    Returns:
        Tuple of (row_indices, col_indices, weights)
    """
    # Explode both lists to pairs
    pairs = (
        matches_df.explode("winners")
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
    # Filter out any None keys and ensure consistent types
    valid_items = [(k, v) for k, v in node_to_idx.items() if k is not None]
    if not valid_items:
        # Return empty arrays if no valid items
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
        )

    ids, indices = zip(*valid_items)
    idx_map = pl.DataFrame({"id": list(ids), "idx": list(indices)})

    # Ensure winner_id and loser_id have matching dtypes with id column
    # Cast both to the same type (use the type of the id column)
    id_dtype = idx_map["id"].dtype
    pairs = pairs.with_columns(
        [pl.col("winner_id").cast(id_dtype), pl.col("loser_id").cast(id_dtype)]
    )

    pairs = (
        pairs.join(idx_map, left_on="winner_id", right_on="id", how="inner")
        .rename({"idx": "w_idx"})
        .join(idx_map, left_on="loser_id", right_on="id", how="inner")
        .rename({"idx": "l_idx"})
        .select(["w_idx", "l_idx", "share"])
    )

    # Aggregate duplicate pairs
    pairs = pairs.group_by(["w_idx", "l_idx"]).agg(
        pl.col("share").sum().alias("w_sum")
    )

    # Convert to numpy arrays
    rows = pairs["w_idx"].to_numpy()
    cols = pairs["l_idx"].to_numpy()
    data = pairs["w_sum"].to_numpy()

    return rows, cols, data


def compute_denominators(
    edges: pl.DataFrame,
    smoothing_strategy,
    loser_col: str = "loser_user_id",
    winner_col: str = "winner_user_id",
    weight_col: str = "w_sum",
) -> pl.DataFrame:
    """
    Compute denominators with smoothing for edge normalization.

    Args:
        edges: Edge DataFrame
        smoothing_strategy: Smoothing strategy object with denom() method
        loser_col: Column name for loser/source nodes
        winner_col: Column name for winner/target nodes
        weight_col: Column name for edge weights

    Returns:
        DataFrame with columns: node_id, W_loss, W_win, denom, lambda
    """
    # Compute loss mass (outgoing edges)
    loss_tot = edges.group_by(loser_col).agg(
        pl.col(weight_col).sum().alias("W_loss")
    )

    # Compute win mass (incoming edges)
    wins_tot = (
        edges.group_by(winner_col)
        .agg(pl.col(weight_col).sum().alias("W_win"))
        .rename({winner_col: loser_col})
    )

    # Join and fill nulls
    denoms = loss_tot.join(wins_tot, on=loser_col, how="left").with_columns(
        pl.col("W_win").fill_null(0.0)
    )

    # Apply smoothing strategy
    if smoothing_strategy:
        # Convert to numpy for smoothing calculation
        W_loss = denoms["W_loss"].to_numpy()
        W_win = denoms["W_win"].to_numpy()
        denom_values = smoothing_strategy.denom(W_loss, W_win)

        denoms = denoms.with_columns(pl.Series("denom", denom_values))
    else:
        # No smoothing
        denoms = denoms.with_columns(pl.col("W_loss").alias("denom"))

    # Compute lambda (smoothing term)
    denoms = denoms.with_columns(
        (pl.col("denom") - pl.col("W_loss")).alias("lambda")
    )

    return denoms


def normalize_edges(
    edges: pl.DataFrame,
    denoms: pl.DataFrame,
    loser_col: str = "loser_user_id",
    weight_col: str = "w_sum",
) -> pl.DataFrame:
    """
    Normalize edge weights by denominators.

    Args:
        edges: Edge DataFrame
        denoms: Denominator DataFrame
        loser_col: Column name for source nodes
        weight_col: Column name for weights

    Returns:
        Normalized edge DataFrame
    """
    # Join denominators
    edges = edges.join(
        denoms.select([loser_col, "denom"]), on=loser_col, how="left"
    )

    # Normalize weights
    edges = edges.with_columns(
        (pl.col(weight_col) / pl.col("denom")).alias("normalized_weight")
    )

    return edges


def edges_to_triplets(
    edges: pl.DataFrame,
    node_to_idx: Dict,
    source_col: str,
    target_col: str,
    weight_col: str = "w_sum",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert edge DataFrame to COO triplets.

    Args:
        edges: Edge DataFrame
        node_to_idx: Node ID to index mapping
        source_col: Source node column
        target_col: Target node column
        weight_col: Weight column

    Returns:
        Tuple of (row_indices, col_indices, weights)
    """
    # Map node IDs to indices
    idx_map = pl.DataFrame(
        {"id": list(node_to_idx.keys()), "idx": list(node_to_idx.values())}
    )

    # Join to get indices
    edges_idx = (
        edges.join(idx_map, left_on=source_col, right_on="id", how="inner")
        .rename({"idx": "src_idx"})
        .join(idx_map, left_on=target_col, right_on="id", how="inner")
        .rename({"idx": "tgt_idx"})
    )

    # Extract arrays
    rows = edges_idx["src_idx"].to_numpy()
    cols = edges_idx["tgt_idx"].to_numpy()
    weights = edges_idx[weight_col].to_numpy()

    return rows, cols, weights
