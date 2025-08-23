"""Tournament influence and retrospective strength calculations."""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl


def compute_tournament_influence(
    pagerank: np.ndarray,
    participants: dict[int, list[int]],
    method: Literal[
        "arithmetic", "geometric", "top_20_sum", "top_20_geom"
    ] = "geometric",
) -> dict[int, float]:
    """Compute tournament influence scores from PageRank values.

    Args:
        pagerank: PageRank scores for all nodes.
        participants: Mapping from tournament ID to list of participant indices.
        method: Aggregation method for combining participant scores.

    Returns:
        Dictionary mapping tournament ID to influence score.
    """
    influence = {}

    for tournament_id, player_indices in participants.items():
        if not player_indices:
            influence[tournament_id] = 1.0
            continue

        # Get PageRank values for tournament participants
        pagerank_values = pagerank[player_indices]

        # Sort for top-k methods
        if "top_20" in method:
            pagerank_values = np.sort(pagerank_values)[::-1][:20]

        # Compute aggregate
        if method == "arithmetic" or method == "top_20_sum":
            influence[tournament_id] = (
                pagerank_values.mean() if len(pagerank_values) > 0 else 1.0
            )
        elif method == "geometric" or method == "top_20_geom":
            epsilon = 1e-10
            influence[tournament_id] = np.exp(
                np.log(pagerank_values + epsilon).mean()
            )
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    return influence


def compute_retrospective_strength(
    edges: pl.DataFrame,
    pagerank: np.ndarray,
    node_to_index: dict,
    decay_rate: float = 0.005,
    weight_column: str = "weight_sum",
) -> np.ndarray:
    """Compute retrospective strength scores.

    This measures the strength of opponents a player has beaten,
    weighted by match importance and time decay.

    Args:
        edges: Edge DataFrame with winner/loser columns.
        pagerank: Current PageRank scores.
        node_to_index: Node ID to index mapping.
        decay_rate: Additional decay for retrospective calculation.
        weight_column: Column containing edge weights.

    Returns:
        Array of retrospective strength scores.
    """
    num_nodes = len(pagerank)
    retrospective_strength = np.zeros(num_nodes)

    if "winner_user_id" in edges.columns:
        winner_column = "winner_user_id"
        loser_column = "loser_user_id"
    else:
        winner_column = "winner_team_id"
        loser_column = "loser_team_id"

    # Get unique winners
    winners = edges[winner_column].unique().to_list()

    for winner_id in winners:
        if winner_id not in node_to_index:
            continue

        winner_index = node_to_index[winner_id]

        # Get all opponents this player/team beat
        opponent_edges = edges.filter(pl.col(winner_column) == winner_id)

        # Sum weighted PageRank of beaten opponents
        total_strength = 0.0
        for row in opponent_edges.iter_rows(named=True):
            loser_id = row[loser_column]
            if loser_id in node_to_index:
                loser_index = node_to_index[loser_id]
                weight = row[weight_column]
                decayed_weight = weight * np.exp(-decay_rate * weight)
                total_strength += pagerank[loser_index] * decayed_weight

        retrospective_strength[winner_index] = total_strength

    return retrospective_strength


def normalize_influence(
    influence: dict[int, float],
    method: Literal["minmax", "zscore", "log", "none"] = "minmax",
) -> dict[int, float]:
    """Normalize tournament influence scores.

    Args:
        influence: Raw influence scores.
        method: Normalization method.

    Returns:
        Normalized influence scores.
    """
    if not influence or method == "none":
        return influence

    values = np.array(list(influence.values()))

    if method == "minmax":
        min_value = values.min()
        max_value = values.max()
        if max_value > min_value:
            normalized = 0.5 + 1.5 * (values - min_value) / (
                max_value - min_value
            )
        else:
            normalized = np.ones_like(values)

    elif method == "zscore":
        mean_value = values.mean()
        std_deviation = values.std()
        if std_deviation > 0:
            normalized = (values - mean_value) / std_deviation
            normalized = np.clip(normalized, -2, 2)
            normalized = 1.0 + 0.5 * normalized
        else:
            normalized = np.ones_like(values)

    elif method == "log":
        log_values = np.log(values + 1e-10)
        min_log_value = log_values.min()
        max_log_value = log_values.max()
        if max_log_value > min_log_value:
            normalized = np.exp(
                min_log_value
                + (log_values - min_log_value)
                * np.log(2)
                / (max_log_value - min_log_value)
            )
        else:
            normalized = np.ones_like(values)

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Reconstruct dictionary
    return dict(zip(influence.keys(), normalized))


def aggregate_multi_round_influence(
    influences: list[dict[int, float]],
    weights: list[float] | None = None,
) -> dict[int, float]:
    """Aggregate influence scores across multiple tick-tock rounds.

    Args:
        influences: List of influence dictionaries from each round.
        weights: Optional weights for each round (defaults to uniform).

    Returns:
        Aggregated influence scores.
    """
    if not influences:
        return {}

    if weights is None:
        weights = [1.0] * len(influences)

    weights = np.array(weights)
    weights = weights / weights.sum()

    # Collect all tournament IDs
    all_tournament_ids = set()
    for influence_dict in influences:
        all_tournament_ids.update(influence_dict.keys())

    # Weighted average
    aggregated = {}
    for tournament_id in all_tournament_ids:
        values = []
        round_weights = []
        for round_index, influence_dict in enumerate(influences):
            if tournament_id in influence_dict:
                values.append(influence_dict[tournament_id])
                round_weights.append(weights[round_index])

        if values:
            round_weights = np.array(round_weights)
            round_weights = round_weights / round_weights.sum()
            aggregated[tournament_id] = np.average(
                values, weights=round_weights
            )
        else:
            aggregated[tournament_id] = 1.0

    return aggregated
