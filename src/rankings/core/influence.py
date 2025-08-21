"""Tournament influence and retrospective strength calculations."""

from typing import Dict, List, Literal, Optional

import numpy as np
import polars as pl


def compute_tournament_influence(
    pagerank: np.ndarray,
    participants: Dict[int, List[int]],
    method: Literal[
        "arithmetic", "geometric", "top_20_sum", "top_20_geom"
    ] = "geometric",
) -> Dict[int, float]:
    """
    Compute tournament influence scores from PageRank values.

    Args:
        pagerank: PageRank scores for all nodes
        participants: Mapping from tournament ID to list of participant indices
        method: Aggregation method for combining participant scores

    Returns:
        Dictionary mapping tournament ID to influence score
    """
    influence = {}

    for tid, player_indices in participants.items():
        if not player_indices:
            influence[tid] = 1.0
            continue

        # Get PageRank values for tournament participants
        pr_values = pagerank[player_indices]

        # Sort for top-k methods
        if "top_20" in method:
            pr_values = np.sort(pr_values)[::-1][:20]

        # Compute aggregate
        if method == "arithmetic" or method == "top_20_sum":
            influence[tid] = pr_values.mean() if len(pr_values) > 0 else 1.0
        elif method == "geometric" or method == "top_20_geom":
            # Geometric mean with small epsilon for stability
            eps = 1e-10
            influence[tid] = np.exp(np.log(pr_values + eps).mean())
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    return influence


def compute_retrospective_strength(
    edges: pl.DataFrame,
    pagerank: np.ndarray,
    node_to_idx: Dict,
    decay_rate: float = 0.005,
    weight_col: str = "w_sum",
) -> np.ndarray:
    """
    Compute retrospective strength scores.

    This measures the strength of opponents a player has beaten,
    weighted by match importance and time decay.

    Args:
        edges: Edge DataFrame with winner/loser columns
        pagerank: Current PageRank scores
        node_to_idx: Node ID to index mapping
        decay_rate: Additional decay for retrospective calculation
        weight_col: Column containing edge weights

    Returns:
        Array of retrospective strength scores
    """
    n = len(pagerank)
    retro_strength = np.zeros(n)

    # Convert edges to numpy for faster computation
    if "winner_user_id" in edges.columns:
        winner_col = "winner_user_id"
        loser_col = "loser_user_id"
    else:
        winner_col = "winner_team_id"
        loser_col = "loser_team_id"

    # Get unique winners
    winners = edges[winner_col].unique().to_list()

    for winner_id in winners:
        if winner_id not in node_to_idx:
            continue

        winner_idx = node_to_idx[winner_id]

        # Get all opponents this player/team beat
        opponent_edges = edges.filter(pl.col(winner_col) == winner_id)

        # Sum weighted PageRank of beaten opponents
        total_strength = 0.0
        for row in opponent_edges.iter_rows(named=True):
            loser_id = row[loser_col]
            if loser_id in node_to_idx:
                loser_idx = node_to_idx[loser_id]
                weight = row[weight_col]
                # Apply additional decay for retrospective calculation
                decayed_weight = weight * np.exp(-decay_rate * weight)
                total_strength += pagerank[loser_idx] * decayed_weight

        retro_strength[winner_idx] = total_strength

    return retro_strength


def normalize_influence(
    influence: Dict[int, float],
    method: Literal["minmax", "zscore", "log", "none"] = "minmax",
) -> Dict[int, float]:
    """
    Normalize tournament influence scores.

    Args:
        influence: Raw influence scores
        method: Normalization method

    Returns:
        Normalized influence scores
    """
    if not influence or method == "none":
        return influence

    values = np.array(list(influence.values()))

    if method == "minmax":
        # Min-max normalization to [0.5, 2.0] range
        min_val = values.min()
        max_val = values.max()
        if max_val > min_val:
            normalized = 0.5 + 1.5 * (values - min_val) / (max_val - min_val)
        else:
            normalized = np.ones_like(values)

    elif method == "zscore":
        # Z-score normalization, clipped to reasonable range
        mean = values.mean()
        std = values.std()
        if std > 0:
            normalized = (values - mean) / std
            normalized = np.clip(normalized, -2, 2)  # Clip to ±2 std
            # Transform to [0.5, 2.0] range
            normalized = 1.0 + 0.5 * normalized
        else:
            normalized = np.ones_like(values)

    elif method == "log":
        # Log normalization for heavy-tailed distributions
        log_values = np.log(values + 1e-10)
        min_log = log_values.min()
        max_log = log_values.max()
        if max_log > min_log:
            normalized = np.exp(
                min_log
                + (log_values - min_log) * np.log(2) / (max_log - min_log)
            )
        else:
            normalized = np.ones_like(values)

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Reconstruct dictionary
    return dict(zip(influence.keys(), normalized))


def aggregate_multi_round_influence(
    influences: List[Dict[int, float]],
    weights: Optional[List[float]] = None,
) -> Dict[int, float]:
    """
    Aggregate influence scores across multiple tick-tock rounds.

    Args:
        influences: List of influence dictionaries from each round
        weights: Optional weights for each round (defaults to uniform)

    Returns:
        Aggregated influence scores
    """
    if not influences:
        return {}

    if weights is None:
        weights = [1.0] * len(influences)

    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()

    # Collect all tournament IDs
    all_tids = set()
    for inf in influences:
        all_tids.update(inf.keys())

    # Weighted average
    aggregated = {}
    for tid in all_tids:
        values = []
        round_weights = []
        for i, inf in enumerate(influences):
            if tid in inf:
                values.append(inf[tid])
                round_weights.append(weights[i])

        if values:
            round_weights = np.array(round_weights)
            round_weights = round_weights / round_weights.sum()
            aggregated[tid] = np.average(values, weights=round_weights)
        else:
            aggregated[tid] = 1.0

    return aggregated
