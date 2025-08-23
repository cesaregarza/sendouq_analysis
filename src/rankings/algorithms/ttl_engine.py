"""
TTL (Tick-Tock over Log-Odds) Engine.

This implements the unified approach where tick-tock provides the outer loop
for tournament influence updates, while exposure log-odds (or any other backend)
provides the inner rating computation.
"""

import logging
import time
from typing import Dict, List, Optional

import numpy as np
import polars as pl

from rankings.algorithms.backends import LogOddsBackend
from rankings.core import (
    Clock,
    ExposureLogOddsConfig,
    ExposureLogOddsResult,
    apply_inactivity_decay,
    compute_tournament_influence,
    normalize_influence,
)
from rankings.core.logging import get_logger, log_timing
from rankings.core.protocols import RatingBackend


class TTLEngine:
    """
    Tick-Tock over Log-Odds (TTL) ranking engine.

    This engine combines:
    - Tick-tock outer loop for anti-gaming via tournament influence
    - Exposure log-odds inner loop for volume-neutral ratings

    The result is rankings that are both volume-neutral AND resistant to gaming.
    """

    def __init__(
        self,
        config: Optional[ExposureLogOddsConfig] = None,
        backend: Optional[RatingBackend] = None,
    ):
        """
        Initialize the TTL engine.

        Args:
            config: Configuration object
            backend: Rating backend (defaults to LogOddsBackend)
        """
        self.config = config or ExposureLogOddsConfig()
        self.logger = get_logger(self.__class__.__name__)

        # Initialize backend if not provided
        if backend is None:
            backend = LogOddsBackend(
                decay_rate=self.config.decay.decay_rate,
                beta=self.config.engine.beta,
                alpha=self.config.pagerank.alpha,
                lambda_mode=self.config.lambda_mode,
                fixed_lambda=self.config.fixed_lambda,
                pagerank_tol=self.config.pagerank.tol,
                pagerank_max_iter=self.config.pagerank.max_iter,
            )
        self.backend = backend

        # Initialize clock
        self.clock = Clock()

        # Storage
        self.tournament_influence: Dict[int, float] = {}
        self.last_result: Optional[ExposureLogOddsResult] = None

        # Damping factor for stability
        self.damping_mu = 0.5  # Can be configured

    def rank_players(
        self,
        matches: pl.DataFrame,
        players: pl.DataFrame,
        initial_influence: Optional[Dict[int, float]] = None,
    ) -> pl.DataFrame:
        """
        Rank players using TTL algorithm.

        Args:
            matches: Match data
            players: Player/roster data
            initial_influence: Optional initial tournament influences

        Returns:
            DataFrame with player rankings
        """
        start_time = time.time()

        # Initialize tournament influence
        if initial_influence:
            self.tournament_influence = initial_influence.copy()
        else:
            # Start with uniform influence
            tournament_ids = matches["tournament_id"].unique().to_list()
            self.tournament_influence = {tid: 1.0 for tid in tournament_ids}

        # Get participants by tournament for influence aggregation
        participants = self._get_participants_by_tournament(matches, players)

        # TTL main loop
        converged = False
        iteration = 0
        max_iterations = self.config.tick_tock.max_ticks
        convergence_tol = self.config.tick_tock.convergence_tol

        influence_history = []
        rating_result = None

        while iteration < max_iterations:
            self.logger.info(f"TTL iteration {iteration + 1}")

            # TICK: Compute ratings with current tournament influences
            self.logger.debug("Computing ratings with current influences...")
            rating_result = self.backend.compute(
                matches,
                players,
                [],  # active_ids not used by log-odds backend
                self.tournament_influence,
            )

            if rating_result.is_empty():
                self.logger.warning("Empty rating result, stopping")
                break

            # TOCK: Update tournament influences using quality_mass
            self.logger.debug("Updating tournament influences...")

            # Map player IDs to indices for influence computation
            player_to_idx = {
                row["id"]: idx
                for idx, row in enumerate(rating_result.iter_rows(named=True))
            }

            # Convert participants to indices
            participants_indexed = {}
            for tid, player_ids in participants.items():
                indices = []
                for pid in player_ids:
                    if pid in player_to_idx:
                        indices.append(player_to_idx[pid])
                if indices:
                    participants_indexed[tid] = indices

            # Get quality mass array
            quality_mass = rating_result["quality_mass"].to_numpy()

            # Compute new tournament influences
            new_influence = compute_tournament_influence(
                quality_mass,
                participants_indexed,
                method=self.config.tick_tock.influence_method,
            )

            # Normalize influences
            new_influence = normalize_influence(new_influence, method="minmax")

            # Check convergence
            if iteration > 0:
                old_values = np.array(list(self.tournament_influence.values()))
                new_values = np.array(
                    [
                        new_influence.get(k, 1.0)
                        for k in self.tournament_influence.keys()
                    ]
                )
                convergence_delta = np.mean(np.abs(old_values - new_values))

                self.logger.info(
                    f"Influence convergence delta: {convergence_delta:.6f}"
                )

                if convergence_delta < convergence_tol:
                    converged = True
                    self.logger.info(
                        f"TTL converged at iteration {iteration + 1}"
                    )
                    break

            # Apply damping for stability
            if iteration > 0 and self.damping_mu < 1.0:
                damped_influence = {}
                for tid in self.tournament_influence:
                    old_val = self.tournament_influence[tid]
                    new_val = new_influence.get(tid, 1.0)
                    damped_val = (
                        1 - self.damping_mu
                    ) * old_val + self.damping_mu * new_val
                    damped_influence[tid] = damped_val
                self.tournament_influence = damped_influence
            else:
                self.tournament_influence = new_influence

            influence_history.append(self.tournament_influence.copy())
            iteration += 1

        if rating_result is None or rating_result.is_empty():
            self.logger.warning("No valid ratings computed")
            return pl.DataFrame()

        # Extract final scores and apply post-processing
        scores = rating_result["score"].to_numpy()
        player_ids = rating_result["id"].to_list()

        # Apply inactivity decay if configured
        if self.config.engine.score_decay_rate > 0:
            last_activity = self._get_last_activity_times(
                matches, players, player_ids
            )
            scores = apply_inactivity_decay(
                scores,
                last_activity,
                self.clock.now,
                delay_days=self.config.engine.score_decay_delay_days,
                decay_rate=self.config.engine.score_decay_rate,
            )

        # Filter by minimum exposure if configured
        if self.config.engine.min_exposure is not None:
            # Compute match counts
            match_counts = self._compute_match_counts(
                matches, players, player_ids
            )
            mask = match_counts >= self.config.engine.min_exposure
        else:
            mask = np.ones(len(scores), dtype=bool)

        # Store result
        self.last_result = ExposureLogOddsResult(
            scores=scores,
            ids=player_ids,
            win_pagerank=rating_result["win_pr"].to_numpy()
            if "win_pr" in rating_result.columns
            else None,
            loss_pagerank=rating_result["loss_pr"].to_numpy()
            if "loss_pr" in rating_result.columns
            else None,
            exposure=rating_result["exposure"].to_numpy()
            if "exposure" in rating_result.columns
            else None,
            lambda_used=rating_result["lambda_used"][0]
            if "lambda_used" in rating_result.columns
            else None,
            iterations=iteration,
            converged=converged,
            active_mask=mask,
        )

        # Create final DataFrame
        result_df = pl.DataFrame(
            {
                "player_id": player_ids,
                "score": scores.tolist(),
                "active": mask.tolist(),
            }
        )

        # Add diagnostic columns if available
        for col in ["win_pr", "loss_pr", "exposure", "quality_mass"]:
            if col in rating_result.columns:
                result_df = result_df.with_columns(rating_result[col])

        # Filter inactive players
        result_df = result_df.filter(pl.col("active"))

        elapsed = time.time() - start_time
        self.logger.info(f"TTL ranking completed in {elapsed:.2f}s")
        log_timing("ttl_total", elapsed)

        return result_df.sort("score", descending=True)

    def _get_participants_by_tournament(
        self,
        matches: pl.DataFrame,
        players: pl.DataFrame,
    ) -> Dict[int, List]:
        """Get mapping of tournament ID to participant IDs."""
        participants = (
            players.select(["tournament_id", "user_id"])
            .unique()
            .group_by("tournament_id")
            .agg(pl.col("user_id").alias("participants"))
        )

        result = {}
        for row in participants.iter_rows(named=True):
            result[row["tournament_id"]] = row["participants"]

        return result

    def _get_last_activity_times(
        self,
        matches: pl.DataFrame,
        players: pl.DataFrame,
        player_ids: List,
    ) -> np.ndarray:
        """Get last activity timestamp for each player."""
        # Simplified - would need proper implementation
        return np.full(len(player_ids), self.clock.now)

    def _compute_match_counts(
        self,
        matches: pl.DataFrame,
        players: pl.DataFrame,
        player_ids: List,
    ) -> np.ndarray:
        """Compute number of matches for each player."""
        # Simplified - would need proper implementation
        return np.ones(len(player_ids))
