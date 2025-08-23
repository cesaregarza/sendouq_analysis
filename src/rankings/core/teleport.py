"""Teleport vector strategies for PageRank algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Sequence, runtime_checkable

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from typing import Any


@runtime_checkable
class TeleportStrategy(Protocol):
    """Protocol for teleport vector computation strategies."""

    def __call__(
        self,
        nodes: Sequence[Any],
        edges: pl.DataFrame,
        from_column: str,
    ) -> np.ndarray:
        """Compute teleport probabilities for nodes.

        Args:
            nodes: List of node identifiers.
            edges: Edge DataFrame.
            from_column: Column name for source nodes.

        Returns:
            Teleport probability vector (sums to 1).
        """
        ...


class UniformTeleport:
    """Uniform teleport probabilities for all nodes."""

    def __call__(
        self,
        nodes: Sequence[Any],
        edges: pl.DataFrame,
        from_column: str,
    ) -> np.ndarray:
        """Return uniform probabilities.

        Args:
            nodes: List of node identifiers.
            edges: Edge DataFrame (unused).
            from_column: Column name for source nodes (unused).

        Returns:
            Uniform probability vector.
        """
        node_count = len(nodes)
        return np.ones(node_count) / max(node_count, 1)


class VolumeInverseTeleport:
    """Teleport probabilities inversely proportional to node volume."""

    def __call__(
        self,
        nodes: Sequence[Any],
        edges: pl.DataFrame,
        from_column: str,
    ) -> np.ndarray:
        """Compute teleport probabilities inversely proportional to sqrt of degree.

        This gives higher teleport probability to nodes with fewer connections.

        Args:
            nodes: List of node identifiers.
            edges: Edge DataFrame.
            from_column: Column name for source nodes.

        Returns:
            Volume-inverse probability vector.
        """
        edge_counts = edges[from_column].value_counts()
        count_mapping = dict(
            zip(edge_counts[from_column], edge_counts["count"])
        )

        teleport_vector = np.array(
            [1.0 / np.sqrt(count_mapping.get(node, 1)) for node in nodes],
            dtype=float,
        )

        teleport_vector /= teleport_vector.sum() or 1.0
        return teleport_vector


class ActivePlayersTeleport:
    """Teleport only to active players based on recent match activity.

    This strategy assigns zero teleport probability to inactive players
    and uniform probability to active players.
    """

    def __init__(self, active_threshold_days: float = 90.0) -> None:
        """Initialize active players teleport strategy.

        Args:
            active_threshold_days: Days since last match to consider player active.
                Defaults to 90.0.
        """
        self.active_threshold_days = active_threshold_days

    def __call__(
        self,
        nodes: Sequence[Any],
        edges: pl.DataFrame,
        from_column: str,
    ) -> np.ndarray:
        """Compute teleport probabilities for active players only.

        Note: This assumes edges DataFrame has a 'timestamp' or similar column
        to determine activity. If not available, falls back to uniform.

        Args:
            nodes: List of node identifiers.
            edges: Edge DataFrame.
            from_column: Column name for source nodes.

        Returns:
            Teleport probabilities for active players.
        """
        # For now, return uniform as a baseline
        # This would need match timestamps to properly implement
        node_count = len(nodes)
        return np.ones(node_count) / max(node_count, 1)


class CustomTeleport:
    """Custom teleport probabilities from provided weights."""

    def __init__(self, weights: dict[Any, float]) -> None:
        """Initialize custom teleport strategy.

        Args:
            weights: Dictionary mapping node IDs to teleport weights.
        """
        self.weights = weights

    def __call__(
        self,
        nodes: Sequence[Any],
        edges: pl.DataFrame,
        from_column: str,
    ) -> np.ndarray:
        """Use custom weights for teleport probabilities.

        Args:
            nodes: List of node identifiers.
            edges: Edge DataFrame (unused).
            from_column: Column name for source nodes (unused).

        Returns:
            Custom weighted teleport probabilities.
        """
        teleport_vector = np.array(
            [self.weights.get(node, 1.0) for node in nodes], dtype=float
        )
        teleport_vector /= teleport_vector.sum() or 1.0
        return teleport_vector


def uniform(nodes: Sequence[Any], *args: Any) -> np.ndarray:
    """Uniform teleport probabilities.

    Args:
        nodes: List of node identifiers.
        *args: Additional arguments (unused).

    Returns:
        Uniform teleport probability vector.
    """
    return UniformTeleport()(nodes, pl.DataFrame(), "")


def volume_inverse(
    nodes: Sequence[Any],
    edges: pl.DataFrame,
    from_column: str,
) -> np.ndarray:
    """Volume-inverse teleport probabilities.

    Args:
        nodes: List of node identifiers.
        edges: Edge DataFrame.
        from_column: Column name for source nodes.

    Returns:
        Volume-inverse teleport probability vector.
    """
    return VolumeInverseTeleport()(nodes, edges, from_column)
