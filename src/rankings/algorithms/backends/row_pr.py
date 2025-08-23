"""Classic row-stochastic PageRank backend for tick-tock orchestration."""

import logging
from typing import Dict, List, Optional

import numpy as np
import polars as pl

from rankings.core import (
    Clock,
    PageRankConfig,
    VolumeInverseTeleport,
    WinsProportional,
    build_player_edges,
    compute_denominators,
    edges_to_triplets,
    normalize_edges,
    pagerank_sparse,
)
from rankings.core.logging import get_logger


class RowPRBackend:
    """
    Classic row-stochastic PageRank rating backend.

    Implements the original tick-tock rating approach using
    row-stochastic PageRank with teleport-proportional smoothing.
    """

    def __init__(
        self,
        decay_rate: float = 0.00385,
        beta: float = 1.0,
        alpha: float = 0.85,
        teleport_mode: str = "volume_inverse",
        smoothing_gamma: float = 0.02,
        smoothing_cap_ratio: float = 1.0,
        pagerank_tol: float = 1e-8,
        pagerank_max_iter: int = 200,
    ):
        """
        Initialize the row PageRank backend.

        Args:
            decay_rate: Time decay rate
            beta: Tournament influence exponent
            alpha: PageRank damping factor
            teleport_mode: Teleport vector mode
            smoothing_gamma: Wins-proportional smoothing parameter
            smoothing_cap_ratio: Cap ratio for smoothing
            pagerank_tol: PageRank convergence tolerance
            pagerank_max_iter: Maximum PageRank iterations
        """
        self.decay_rate = decay_rate
        self.beta = beta
        self.alpha = alpha
        self.teleport_mode = teleport_mode
        self.smoothing_gamma = smoothing_gamma
        self.smoothing_cap_ratio = smoothing_cap_ratio
        self.pagerank_tol = pagerank_tol
        self.pagerank_max_iter = pagerank_max_iter

        self.logger = get_logger(self.__class__.__name__)
        self.clock = Clock()

        # Initialize teleport strategy
        if teleport_mode == "volume_inverse":
            self.teleport_strategy = VolumeInverseTeleport()
        else:
            from rankings.core import UniformTeleport

            self.teleport_strategy = UniformTeleport()

        # Initialize smoothing strategy
        self.smoothing_strategy = WinsProportional(
            gamma=smoothing_gamma,
            cap_ratio=smoothing_cap_ratio,
        )

    def compute(
        self,
        matches: pl.DataFrame,
        players: Optional[pl.DataFrame],
        active_ids: List,
        tournament_influence: Dict[int, float],
        **kwargs,
    ) -> pl.DataFrame:
        """
        Compute classic PageRank ratings.

        Args:
            matches: Match data
            players: Player/roster data
            active_ids: Active player IDs (not used)
            tournament_influence: Tournament influence scores (S)

        Returns:
            DataFrame with id, score, quality_mass
        """
        if players is None:
            self.logger.warning(
                "No player data provided, cannot compute ratings"
            )
            return pl.DataFrame()

        # Build edges with tournament influence
        edges = build_player_edges(
            matches,
            players,
            tournament_influence,
            self.clock.now,
            self.decay_rate,
            self.beta,
        )

        if edges.is_empty():
            self.logger.warning("No edges built")
            return pl.DataFrame()

        # Get unique nodes
        all_nodes = set()
        all_nodes.update(edges["loser_user_id"].unique().to_list())
        all_nodes.update(edges["winner_user_id"].unique().to_list())
        node_ids = sorted(list(all_nodes))
        node_to_idx = {node: idx for idx, node in enumerate(node_ids)}
        n = len(node_ids)

        # Compute denominators with smoothing
        denominators = compute_denominators(
            edges,
            self.smoothing_strategy,
            loser_column="loser_user_id",
            winner_column="winner_user_id",
        )

        # Normalize edges
        edges = normalize_edges(
            edges,
            denominators,
            loser_column="loser_user_id",
        )

        # Convert to triplets
        rows, cols, weights = edges_to_triplets(
            edges,
            node_to_idx,
            source_column="loser_user_id",
            target_column="winner_user_id",
            weight_column="normalized_weight",
        )

        # Compute teleport vector
        teleport = self.teleport_strategy(
            node_ids,
            edges,
            "loser_user_id",
        )

        # Run PageRank
        pr_config = PageRankConfig(
            alpha=self.alpha,
            tol=self.pagerank_tol,
            max_iter=self.pagerank_max_iter,
            orientation="row",
            redistribute_dangling=True,
        )

        pagerank = pagerank_sparse(rows, cols, weights, n, teleport, pr_config)

        # For classic PageRank, quality_mass = pagerank (normalized)
        # This maintains compatibility with existing influence aggregation
        quality_mass = pagerank / pagerank.sum()

        # Create result DataFrame
        result = pl.DataFrame(
            {
                "id": node_ids,
                "score": pagerank.tolist(),
                "quality_mass": quality_mass.tolist(),
                "pagerank": pagerank.tolist(),
            }
        )

        return result
