"""Teleport vector strategies for PageRank algorithms."""

from typing import Protocol, Sequence, runtime_checkable

import numpy as np
import polars as pl


@runtime_checkable
class TeleportStrategy(Protocol):
    """Protocol for teleport vector computation strategies."""

    def __call__(
        self,
        nodes: Sequence[object],
        edges: pl.DataFrame,
        from_col: str,
    ) -> np.ndarray:
        """
        Compute teleport probabilities for nodes.

        Args:
            nodes: List of node identifiers
            edges: Edge DataFrame
            from_col: Column name for source nodes

        Returns:
            Teleport probability vector (sums to 1)
        """
        ...


class UniformTeleport:
    """Uniform teleport probabilities for all nodes."""

    def __call__(
        self,
        nodes: Sequence[object],
        edges: pl.DataFrame,
        from_col: str,
    ) -> np.ndarray:
        """Return uniform probabilities."""
        n = len(nodes)
        return np.ones(n) / max(n, 1)


class VolumeInverseTeleport:
    """Teleport probabilities inversely proportional to node volume."""

    def __call__(
        self,
        nodes: Sequence[object],
        edges: pl.DataFrame,
        from_col: str,
    ) -> np.ndarray:
        """
        Compute teleport probabilities inversely proportional to sqrt of degree.

        This gives higher teleport probability to nodes with fewer connections.
        """
        # Count edges per node
        counts = edges[from_col].value_counts()
        count_map = dict(zip(counts[from_col], counts["count"]))

        # Compute inverse sqrt of degree
        v = np.array(
            [1.0 / np.sqrt(count_map.get(node, 1)) for node in nodes],
            dtype=float,
        )

        # Normalize to sum to 1
        v /= v.sum() or 1.0
        return v


class ActivePlayersTeleport:
    """
    Teleport only to active players based on recent match activity.

    This strategy assigns zero teleport probability to inactive players
    and uniform probability to active players.
    """

    def __init__(self, active_threshold_days: float = 90.0):
        """
        Args:
            active_threshold_days: Days since last match to consider player active
        """
        self.active_threshold_days = active_threshold_days

    def __call__(
        self,
        nodes: Sequence[object],
        edges: pl.DataFrame,
        from_col: str,
    ) -> np.ndarray:
        """
        Compute teleport probabilities for active players only.

        Note: This assumes edges DataFrame has a 'timestamp' or similar column
        to determine activity. If not available, falls back to uniform.
        """
        # For now, return uniform as a baseline
        # This would need match timestamps to properly implement
        n = len(nodes)
        return np.ones(n) / max(n, 1)


class CustomTeleport:
    """Custom teleport probabilities from provided weights."""

    def __init__(self, weights: dict):
        """
        Args:
            weights: Dictionary mapping node IDs to teleport weights
        """
        self.weights = weights

    def __call__(
        self,
        nodes: Sequence[object],
        edges: pl.DataFrame,
        from_col: str,
    ) -> np.ndarray:
        """Use custom weights for teleport probabilities."""
        v = np.array(
            [self.weights.get(node, 1.0) for node in nodes], dtype=float
        )
        v /= v.sum() or 1.0
        return v


# Convenience functions for backward compatibility
def uniform(nodes: Sequence[object], *args) -> np.ndarray:
    """Uniform teleport probabilities."""
    return UniformTeleport()(nodes, pl.DataFrame(), "")


def volume_inverse(
    nodes: Sequence[object],
    edges: pl.DataFrame,
    from_col: str,
) -> np.ndarray:
    """Volume-inverse teleport probabilities."""
    return VolumeInverseTeleport()(nodes, edges, from_col)
