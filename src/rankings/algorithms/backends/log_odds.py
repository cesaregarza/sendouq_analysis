"""Log-odds rating computation backend for tick-tock orchestration."""

from __future__ import annotations

import numpy as np
import polars as pl

from rankings.core import (
    Clock,
    PageRankConfig,
    build_exposure_triplets,
    convert_matches_dataframe,
    pagerank_sparse,
)
from rankings.core.logging import get_logger


class LogOddsBackend:
    """Log-odds rating backend.

    Implements log-odds ratings using PageRank computations with
    exposure-based teleport vectors and lambda smoothing.
    """

    def __init__(
        self,
        decay_rate: float = 0.00385,
        beta: float = 1.0,
        alpha: float = 0.85,
        lambda_mode: str = "auto",
        fixed_lambda: float | None = None,
        pagerank_tol: float = 1e-8,
        pagerank_max_iter: int = 200,
        epsilon: float = 1e-9,
    ) -> None:
        """Initialize the log-odds backend.

        Args:
            decay_rate: Time decay rate for match weights.
            beta: Tournament influence exponent.
            alpha: PageRank damping factor.
            lambda_mode: Lambda computation mode ('auto' or 'fixed').
            fixed_lambda: Fixed lambda value when mode is 'fixed'.
            pagerank_tol: PageRank convergence tolerance.
            pagerank_max_iter: Maximum PageRank iterations.
            epsilon: Small value for numerical stability.
        """
        self.decay_rate = decay_rate
        self.beta = beta
        self.alpha = alpha
        self.lambda_mode = lambda_mode
        self.fixed_lambda = fixed_lambda
        self.pagerank_tol = pagerank_tol
        self.pagerank_max_iter = pagerank_max_iter
        self.epsilon = epsilon

        self.logger = get_logger(self.__class__.__name__)
        self.clock = Clock()
        self._last_rho = None

    def compute(
        self,
        matches: pl.DataFrame,
        players: pl.DataFrame | None,
        active_ids: list[int],
        tournament_influence: dict[int, float],
        **kwargs,
    ) -> pl.DataFrame:
        """Compute log-odds ratings for players.

        Args:
            matches: Match data.
            players: Player/roster data.
            active_ids: Active player IDs (not used in this backend).
            tournament_influence: Tournament influence scores.
            **kwargs: Additional keyword arguments.

        Returns:
            DataFrame with player ratings and metrics.
        """
        if players is None:
            self.logger.warning(
                "No player data provided, cannot compute ratings"
            )
            return pl.DataFrame()

        matches_df = convert_matches_dataframe(
            matches,
            players,
            tournament_influence or {},
            self.clock.now,
            self.decay_rate,
            self.beta,
            include_share=True,
            streaming=False,
        )
        if matches_df.is_empty():
            self.logger.warning("No valid matches after conversion")
            return pl.DataFrame()

        all_player_ids = set()
        for row in matches_df.iter_rows(named=True):
            all_player_ids.update(row["winners"])
            all_player_ids.update(row["losers"])
        node_ids = sorted(list(all_player_ids))
        node_to_idx = {player_id: idx for idx, player_id in enumerate(node_ids)}
        num_nodes = len(node_ids)

        winners_share = (
            matches_df.select(["winners", "share"])
            .explode("winners")
            .rename({"winners": "id"})
        )
        losers_share = (
            matches_df.select(["losers", "share"])
            .explode("losers")
            .rename({"losers": "id"})
        )
        exposure_share_df = (
            pl.concat([winners_share, losers_share])
            .group_by("id")
            .agg(pl.col("share").sum().alias("e_share"))
        )

        teleport_vector = np.full(num_nodes, self.epsilon, dtype=float)
        for row in exposure_share_df.iter_rows(named=True):
            idx = node_to_idx.get(row["id"])
            if idx is not None:
                teleport_vector[idx] += float(row["e_share"])
        teleport_vector /= (
            teleport_vector.sum() if teleport_vector.sum() > 0 else 1.0
        )
        self._last_rho = teleport_vector

        rows, cols, data = build_exposure_triplets(matches_df, node_to_idx)

        pagerank_config = PageRankConfig(
            alpha=self.alpha,
            tol=self.pagerank_tol,
            max_iter=self.pagerank_max_iter,
            orientation="col",
            redistribute_dangling=True,
        )

        win_pagerank = pagerank_sparse(
            rows, cols, data, num_nodes, teleport_vector, pagerank_config
        )
        loss_pagerank = pagerank_sparse(
            cols, rows, data, num_nodes, teleport_vector, pagerank_config
        )

        if self.lambda_mode == "fixed" and self.fixed_lambda is not None:
            lambda_smooth = float(self.fixed_lambda)
        elif self.lambda_mode == "auto":
            target = 0.025 * float(np.median(win_pagerank))
            median_teleport = float(np.median(teleport_vector))
            lambda_smooth = (
                0.0
                if median_teleport == 0.0
                else max(target / median_teleport, 0.0)
            )
        else:
            lambda_smooth = 1e-4

        smoothed_win_pagerank = win_pagerank + lambda_smooth * teleport_vector
        smoothed_loss_pagerank = loss_pagerank + lambda_smooth * teleport_vector
        scores = np.log(smoothed_win_pagerank / smoothed_loss_pagerank)
        quality_mass = smoothed_win_pagerank / (
            smoothed_win_pagerank + smoothed_loss_pagerank
        )

        winners_weight = (
            matches_df.select(["winners", "weight"])
            .explode("winners")
            .rename({"winners": "id"})
        )
        losers_weight = (
            matches_df.select(["losers", "weight"])
            .explode("losers")
            .rename({"losers": "id"})
        )
        exposure_weight_df = (
            pl.concat([winners_weight, losers_weight])
            .group_by("id")
            .agg(pl.col("weight").sum().alias("exposure"))
        )

        exposure = np.zeros(num_nodes, dtype=float)
        for row in exposure_weight_df.iter_rows(named=True):
            idx = node_to_idx.get(row["id"])
            if idx is not None:
                exposure[idx] = float(row["exposure"])

        return pl.DataFrame(
            {
                "id": node_ids,
                "score": scores.tolist(),
                "quality_mass": quality_mass.tolist(),
                "win_pagerank": win_pagerank.tolist(),
                "loss_pagerank": loss_pagerank.tolist(),
                "exposure": exposure.tolist(),
                "lambda_used": [lambda_smooth] * num_nodes,
            }
        )
