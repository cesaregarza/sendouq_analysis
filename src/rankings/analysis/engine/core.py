"""Core rating engine implementation.

This module contains the main RatingEngine class that implements a sophisticated
rating system using a tick-tock algorithm for iterative refinement of ratings
and tournament strengths.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict, List, Literal, Optional
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

try:
    import scipy.sparse as sp
except Exception:
    sp = None

from rankings.analysis.asymmetric_confidence import (
    AsymmetricConfidenceCalculator,
    AsymmetricConfidenceMetrics,
)
from rankings.core.constants import (
    DEFAULT_BETA,
    DEFAULT_DAMPING_FACTOR,
    DEFAULT_DECAY_HALF_LIFE_DAYS,
    DEFAULT_INFLUENCE_AGG_METHOD,
    DEFAULT_MAX_PAGERANK_ITER,
    DEFAULT_MAX_TICK_TOCK,
    DEFAULT_PAGERANK_TOLERANCE,
    DEFAULT_REFERENCE_DATE,
    DEFAULT_STRENGTH_AGG,
    DEFAULT_STRENGTH_K,
    DEFAULT_TICK_TOCK_TOLERANCE,
    DEFAULT_VOLUME_EPSILON,
    DEFAULT_VOLUME_MIX_ETA,
    DEFAULT_VOLUME_MIX_GAMMA,
    TELEPORT_UNIFORM,
    TELEPORT_VOLUME_INVERSE,
    TELEPORT_VOLUME_MIX,
)
from rankings.core.logging import (
    get_logger,
    log_algorithm_convergence,
    log_dataframe_stats,
    log_timing,
)


class RatingEngine:
    """
    Advanced rating engine with tournament strength calculation.

    This engine implements a tick-tock algorithm that alternates between:
    1. **Tick**: Computing ratings given current tournament strengths
    2. **Tock**: Recomputing tournament strengths based on updated ratings

    The process iterates until convergence, resulting in ratings and tournament
    strengths that are mutually consistent.

    Key Features:
    - Time decay with configurable half-life
    - Tournament strength weighting with configurable exponent (beta)
    - Flexible teleport vectors for PageRank
    - Multiple aggregation methods for tournament influence calculation
    - Retroactive tournament strength metrics
    - Teleport-proportional smoothing for stability

    Parameters
    ----------
    decay_half_life_days : float, default=30.0
        Half-life for exponential time decay in days
    damping_factor : float, default=0.85
        PageRank damping factor (probability of following edges vs teleporting)
    beta : float, default=0.0
        Tournament strength exponent (0.0=no weighting, 1.0=full weighting)
    tick_tock_stabilize_tol : float, default=1e-4
        Convergence tolerance for tick-tock iterations
    max_tick_tock : int, default=5
        Maximum number of tick-tock iterations
    max_pagerank_iter : int, default=100
        Maximum PageRank iterations per tick step
    pagerank_tol : float, default=1e-8
        PageRank convergence tolerance
    now : datetime, optional
        Reference time for decay calculation
    teleport : str or dict, default="volume_inverse"
        Teleport vector specification ("uniform", "volume_inverse", "volume_mix", or dict)
    volume_mix_eta : float, default=0.3
        Weight for volume component in volume_mix teleport (0=uniform, 1=pure volume)
    volume_mix_gamma : float, default=0.5
        Exponent for volume scaling in volume_mix teleport (0.5=sqrt, 1=linear)
    volume_epsilon : float, default=1.0
        Epsilon for smoothing zero volumes in volume_mix teleport
    smoothing_mode : str, default="wins"
        Smoothing mode for handling rare losses ("none", "wins", "constant")
    smoothing_gamma : float, default=0.02
        Coefficient for wins-proportional smoothing
    smoothing_const : float, default=0.0
        Constant value for constant smoothing mode
    smoothing_cap_ratio : float, default=1.0
        Optional cap on lambda values (ratio to W_loss)
    influence_agg_method : str, default="mean"
        Method for aggregating participant ratings into tournament influence.
        Options: "mean", "sum", "median", "top_20_sum", "log_top_20_sum", "sqrt_top_20_sum"
    strength_agg : str, default="mean"
        Method for computing retrospective tournament strength
    strength_k : int, default=5
        Parameter for trimmed_mean or topN_sum strength aggregation
    normalize_min_influence : float, optional
        If set, normalizes tournament influences so minimum equals this value.
        Default None (no normalization). Set to 1.0 to prevent ultra-low influence exploits.
    """

    def __init__(
        self,
        *,
        decay_half_life_days: float = DEFAULT_DECAY_HALF_LIFE_DAYS,
        damping_factor: float = DEFAULT_DAMPING_FACTOR,
        beta: float = DEFAULT_BETA,
        tick_tock_stabilize_tol: float = DEFAULT_TICK_TOCK_TOLERANCE,
        max_tick_tock: int = DEFAULT_MAX_TICK_TOCK,
        max_pagerank_iter: int = DEFAULT_MAX_PAGERANK_ITER,
        pagerank_tol: float = DEFAULT_PAGERANK_TOLERANCE,
        now: datetime | None = None,
        teleport: str | dict[int, float] = TELEPORT_VOLUME_INVERSE,
        volume_mix_eta: float = DEFAULT_VOLUME_MIX_ETA,
        volume_mix_gamma: float = DEFAULT_VOLUME_MIX_GAMMA,
        volume_epsilon: float = DEFAULT_VOLUME_EPSILON,
        # Smoothing parameters (teleport-proportional smoothing)
        smoothing_mode: Literal["none", "wins", "constant"] = "wins",
        smoothing_gamma: float = 0.02,
        smoothing_const: float = 0.0,
        smoothing_cap_ratio: float = 1.0,
        influence_agg_method: Literal[
            "mean",
            "sum",
            "median",
            "top_20_sum",
            "log_top_20_sum",
            "sqrt_top_20_sum",
            "top_10_sum",
            "top_20_mean",
        ] = DEFAULT_INFLUENCE_AGG_METHOD,
        strength_agg: Literal[
            "mean", "median", "trimmed_mean", "topN_sum"
        ] = DEFAULT_STRENGTH_AGG,
        strength_k: int = DEFAULT_STRENGTH_K,
        normalize_min_influence: float | None = None,
        compute_edge_flux: bool = False,
    ):
        self.logger = get_logger(__name__)
        self.logger.info("Initializing RatingEngine with parameters:")
        self.logger.info(f"  decay_half_life_days={decay_half_life_days}")
        self.logger.info(f"  damping_factor={damping_factor}")
        self.logger.info(f"  beta={beta}")
        self.logger.info(f"  tick_tock_stabilize_tol={tick_tock_stabilize_tol}")
        self.logger.info(f"  max_tick_tock={max_tick_tock}")
        self.logger.info(f"  influence_agg_method={influence_agg_method}")
        self.logger.info(f"  normalize_min_influence={normalize_min_influence}")
        self.decay_half_life_days = decay_half_life_days
        self.decay_rate = math.log(2.0) / decay_half_life_days
        self.damping_factor = damping_factor
        self.beta = beta
        self.tick_tock_stabilize_tol = tick_tock_stabilize_tol
        self.max_tick_tock = max_tick_tock
        self.max_pagerank_iter = max_pagerank_iter
        self.pagerank_tol = pagerank_tol
        self.now = (now or DEFAULT_REFERENCE_DATE).astimezone(timezone.utc)
        self.now_ts = int(self.now.timestamp())  # Precompute for performance
        self.teleport_spec = teleport
        self.volume_mix_eta = volume_mix_eta
        self.volume_mix_gamma = volume_mix_gamma
        self.volume_epsilon = volume_epsilon
        # Store smoothing configuration
        self.smoothing_mode = smoothing_mode
        self.smoothing_gamma = smoothing_gamma
        self.smoothing_const = smoothing_const
        self.smoothing_cap_ratio = smoothing_cap_ratio
        self.influence_agg_method = influence_agg_method
        self.strength_agg = strength_agg
        self.strength_k = strength_k
        self.normalize_min_influence = normalize_min_influence
        self.compute_edge_flux = compute_edge_flux

        # Results populated after a run
        self.edge_flux_: pl.DataFrame | None = None
        self.tournament_influence_: dict[int, float] | None = None
        self.tournament_strength: pl.DataFrame | None = None
        self.ratings_df: pl.DataFrame | None = None
        self.global_prior_: float | None = None
        # Expose final per-source denominators for analysis reuse
        self.denominators_df_: pl.DataFrame | None = None
        self.confidence_metrics_: dict[
            str, AsymmetricConfidenceMetrics
        ] | None = None

    @property
    def tournament_influence(self) -> dict[int, float] | None:
        """Return the tournament influence dictionary from the last run."""
        return self.tournament_influence_

    # =========================================================================
    # Public API
    # =========================================================================

    def rank_teams(self, matches: pl.DataFrame) -> pl.DataFrame:
        """
        Rank teams using the tick-tock algorithm.

        Parameters
        ----------
        matches : pl.DataFrame
            Matches DataFrame with winner_team_id, loser_team_id columns

        Returns
        -------
        pl.DataFrame
            Team rankings with columns: team_id, team_rank
        """
        self.logger.info("Starting team ranking calculation")
        if self.logger.isEnabledFor(logging.DEBUG):
            log_dataframe_stats(self.logger, matches, "team_matches")

        with log_timing(self.logger, "team ranking calculation"):
            result = self._tick_tock_loop(
                matches=matches,
                players=None,
                node_from="loser_team_id",
                node_to="winner_team_id",
            )

        if self.logger.isEnabledFor(logging.DEBUG):
            log_dataframe_stats(self.logger, result, "team_rankings_result")
        self.logger.info(
            f"Team ranking completed: {result.height} teams ranked"
        )
        return result

    def rank_players(
        self, matches: pl.DataFrame, players: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Rank players using the tick-tock algorithm.

        Parameters
        ----------
        matches : pl.DataFrame
            Matches DataFrame
        players : pl.DataFrame
            Players DataFrame with tournament_id, team_id, user_id columns

        Returns
        -------
        pl.DataFrame
            Player rankings with columns: user_id, player_rank
        """
        self.logger.info("Starting player ranking calculation")
        if self.logger.isEnabledFor(logging.DEBUG):
            log_dataframe_stats(self.logger, matches, "matches")
            log_dataframe_stats(self.logger, players, "players")

        with log_timing(self.logger, "player ranking calculation"):
            result = self._tick_tock_loop(
                matches=matches,
                players=players,
                node_from="loser_user_id",
                node_to="winner_user_id",
            )

        if self.logger.isEnabledFor(logging.DEBUG):
            log_dataframe_stats(self.logger, result, "player_rankings_result")
        self.logger.info(
            f"Player ranking completed: {result.height} players ranked"
        )
        return result

    # =========================================================================
    # Core tick-tock algorithm
    # =========================================================================

    def _tick_tock_loop(
        self,
        *,
        matches: pl.DataFrame,
        players: pl.DataFrame | None,
        node_from: str,
        node_to: str,
    ) -> pl.DataFrame:
        """
        Core tick-tock iteration loop.

        Alternates between computing ratings (tick) and updating tournament
        strengths (tock) until convergence.
        """
        # Record operating mode for downstream consumers (player vs team)
        self.mode = "team" if node_to == "winner_team_id" else "player"

        self.logger.debug(
            f"Starting tick-tock loop with max_iterations={self.max_tick_tock}"
        )
        if self.logger.isEnabledFor(logging.DEBUG):
            log_dataframe_stats(self.logger, matches, "tick_tock_matches")
            if players is not None:
                log_dataframe_stats(self.logger, players, "tick_tock_players")

        # Initialize tournament influences to 1.0
        tournament_influence = {
            tid: 1.0 for tid in matches["tournament_id"].unique()
        }
        self.logger.info(
            f"Initialized {len(tournament_influence)} tournaments with influence=1.0"
        )
        prev_influence = {}
        rating_df = pl.DataFrame([])

        # Precompute participants once (they don't change during tick-tock)
        participants_df = self._participants_by_tournament(
            matches, players, node_to
        )
        self.logger.debug(
            f"Precomputed {participants_df.height} participant-tournament pairs"
        )

        with log_timing(self.logger, "tick-tock iterations"):
            for i in range(self.max_tick_tock):
                self.logger.debug(
                    f"Starting tick-tock iteration {i + 1}/{self.max_tick_tock}"
                )

                # TICK: Build edges with current tournament strengths
                with log_timing(
                    self.logger,
                    f"tick step {i + 1} - edge building",
                    logging.DEBUG,
                ):
                    if node_from == "loser_team_id":
                        edges = self._build_team_edges(
                            matches, tournament_influence
                        )
                    else:  # players
                        assert players is not None
                        edges = self._build_player_edges(
                            matches, players, tournament_influence
                        )

                if self.logger.isEnabledFor(logging.DEBUG):
                    log_dataframe_stats(
                        self.logger,
                        edges,
                        f"edges_iteration_{i + 1}",
                        logging.DEBUG,
                    )

                with log_timing(
                    self.logger, f"tick step {i + 1} - PageRank", logging.DEBUG
                ):
                    rating_df = self._pagerank(
                        edges, node_col_from=node_from, node_col_to=node_to
                    )

                if self.logger.isEnabledFor(logging.DEBUG):
                    log_dataframe_stats(
                        self.logger,
                        rating_df,
                        f"ratings_iteration_{i + 1}",
                        logging.DEBUG,
                    )

                # TOCK: Recompute tournament strengths from fresh ratings
                with log_timing(
                    self.logger,
                    f"tock step {i + 1} - tournament influence",
                    logging.DEBUG,
                ):
                    tournament_influence = self._compute_tournament_influence(
                        rating_df, matches, players, node_to, participants_df
                    )

                # Check convergence
                delta = sum(
                    abs(tournament_influence[t] - prev_influence.get(t, 0.0))
                    for t in tournament_influence
                ) / len(tournament_influence)

                log_algorithm_convergence(
                    self.logger,
                    i + 1,
                    delta,
                    self.tick_tock_stabilize_tol,
                    "tick-tock",
                )

                prev_influence = tournament_influence
                if delta < self.tick_tock_stabilize_tol:
                    self.logger.info(
                        f"Tick-tock converged at iteration {i + 1} with delta {delta:.2e}"
                    )
                    break
            else:
                self.logger.warning(
                    f"Tick-tock reached max iterations ({self.max_tick_tock}) without convergence. Final delta: {delta:.2e}"
                )

        # Save final results
        self.tournament_influence_ = tournament_influence
        self.logger.debug("Computing retrospective tournament strengths")
        self._compute_retro_strength(
            rating_df,
            matches,
            players,
            node_to,
            tournament_influence,
            participants_df,
        )
        self.ratings_df = rating_df

        # Return with appropriate column name
        score_name = (
            "team_rank" if node_from.startswith("loser_team") else "player_rank"
        )
        result = rating_df.rename({"score": score_name}).sort(
            score_name, descending=True
        )

        if self.logger.isEnabledFor(logging.DEBUG):
            log_dataframe_stats(self.logger, result, "final_tick_tock_result")
        self.logger.info(
            f"Tick-tock loop completed, returning {result.height} ranked entities"
        )
        return result

    # =========================================================================
    # Edge building functions
    # =========================================================================

    def _build_team_edges(
        self, matches: pl.DataFrame, tournament_strengths: dict[int, float]
    ) -> pl.DataFrame:
        """Build team-level edges with tournament strength weighting."""
        if matches.is_empty():
            return pl.DataFrame([])

        strength_df = pl.DataFrame(
            {
                "tournament_id": list(tournament_strengths.keys()),
                "S": list(tournament_strengths.values()),
            }
        )

        # Filter valid matches (exclude byes/forfeits) and join strengths
        filter_expr = (
            pl.col("winner_team_id").is_not_null()
            & pl.col("loser_team_id").is_not_null()
        )
        # Only filter by is_bye if column exists
        if "is_bye" in matches.columns:
            filter_expr = filter_expr & ~pl.col("is_bye").fill_null(False)

        df = (
            matches.filter(filter_expr)
            .join(strength_df, on="tournament_id", how="left")
            .fill_null(1.0)  # Default strength for unseen tournaments
        )

        # Add event timestamps and compute weights with influence decay
        self._current_df_columns = df.columns
        df = df.with_columns(self._event_ts_expr().alias("ts"))

        # Single time decay (no double decay)
        df = df.with_columns(
            (self._decay_expr("ts") * (pl.col("S") ** self.beta)).alias("w")
        )

        # Aggregate raw pair weights
        grouped = df.group_by(["loser_team_id", "winner_team_id"]).agg(
            pl.col("w").sum().alias("w_sum")
        )

        # Totals by source (loss mass)
        loss_tot = grouped.group_by("loser_team_id").agg(
            pl.col("w_sum").sum().alias("W_loss")
        )

        # Wins mass by source (sum of weights when this team is winner)
        wins_tot = (
            df.group_by("winner_team_id")
            .agg(pl.col("w").sum().alias("W_win"))
            .rename({"winner_team_id": "loser_team_id"})
        )

        # Build denominators with smoothing
        denoms = (
            loss_tot.join(wins_tot, on="loser_team_id", how="left")
            .with_columns(pl.col("W_win").fill_null(0.0))
            .with_columns(
                pl.when(pl.lit(self.smoothing_mode) == "wins")
                .then(
                    pl.col("W_loss")
                    + (pl.lit(self.smoothing_gamma) * pl.col("W_win"))
                )
                .when(pl.lit(self.smoothing_mode) == "constant")
                .then(pl.col("W_loss") + pl.lit(self.smoothing_const))
                .otherwise(pl.col("W_loss"))
                .alias("denom_raw")
            )
        )

        # Optional cap: denom = W_loss + min(lambda_i, rho * W_loss)
        if self.smoothing_mode != "none" and math.isfinite(
            self.smoothing_cap_ratio
        ):
            denoms = denoms.with_columns(
                (
                    pl.col("W_loss")
                    + pl.min_horizontal(
                        (pl.col("denom_raw") - pl.col("W_loss")),
                        (pl.lit(self.smoothing_cap_ratio) * pl.col("W_loss")),
                    )
                ).alias("denom")
            )
        else:
            denoms = denoms.with_columns(pl.col("denom_raw").alias("denom"))

        # Also expose lambda_i for analysis: lambda = denom - W_loss
        denoms = denoms.with_columns(
            (pl.col("denom") - pl.col("W_loss")).alias("lambda")
        )

        # Join denominators and compute normalized weights WITHOUT forcing row sums to 1
        edges = (
            grouped.join(
                denoms.select(["loser_team_id", "denom"]),
                on="loser_team_id",
                how="left",
            )
            .with_columns((pl.col("w_sum") / pl.col("denom")).alias("norm_w"))
            .select(["loser_team_id", "winner_team_id", "norm_w"])
        )

        # Store denominators for downstream analysis
        self.denominators_df_ = (
            denoms.select(
                ["loser_team_id", "W_loss", "W_win", "lambda", "denom"]
            )
            .rename({"loser_team_id": "id"})
            .with_columns(pl.col("id").cast(pl.Int64))
        )

        return edges

    def _build_player_edges(
        self,
        matches: pl.DataFrame,
        players: pl.DataFrame,
        tournament_strengths: dict[int, float],
    ) -> pl.DataFrame:
        """Build player-level edges with tournament strength weighting."""
        if matches.is_empty() or players.is_empty():
            return pl.DataFrame([])

        # Tournament strength lookup
        s_df = pl.DataFrame(
            {"tournament_id": list(S.keys()), "S": list(S.values())}
        )

        # Prepare matches with strengths and weights (exclude byes/forfeits)
        filter_expr = (
            pl.col("winner_team_id").is_not_null()
            & pl.col("loser_team_id").is_not_null()
        )
        # Only filter by is_bye if column exists
        if "is_bye" in matches.columns:
            filter_expr = filter_expr & ~pl.col("is_bye").fill_null(False)

        m = (
            matches.filter(filter_expr)
            .join(strength_df, on="tournament_id", how="left")
            .fill_null(1.0)
        )
        self._current_df_columns = m.columns
        m = m.with_columns(self._event_ts_expr().alias("ts"))

        # Single time decay (no double decay)
        m = m.with_columns(
            (self._decay_expr("ts") * (pl.col("S") ** self.beta)).alias(
                "match_w"
            )
        )

        # Player selection for joins (cast both sides explicitly)
        pl_sel = players.select(
            ["tournament_id", "team_id", "user_id"]
        ).with_columns(
            [
                pl.col("tournament_id").cast(pl.Int64),
                pl.col("team_id").cast(pl.Int64),
                # Do NOT cast user_id; allow string IDs in synthetic data
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

        # Totals by source (loss mass)
        loss_tot = edges.group_by("loser_user_id").agg(
            pl.col("w_sum").sum().alias("W_loss")
        )

        # Wins mass by source (sum of weights when this user is a winner)
        wins_tot = (
            pairs.group_by("winner_user_id")
            .agg(pl.col("match_w").sum().alias("W_win"))
            .rename({"winner_user_id": "loser_user_id"})
        )

        # Build denominators with smoothing
        denoms = (
            loss_tot.join(wins_tot, on="loser_user_id", how="left")
            .with_columns(pl.col("W_win").fill_null(0.0))
            .with_columns(
                pl.when(pl.lit(self.smoothing_mode) == "wins")
                .then(
                    pl.col("W_loss")
                    + (pl.lit(self.smoothing_gamma) * pl.col("W_win"))
                )
                .when(pl.lit(self.smoothing_mode) == "constant")
                .then(pl.col("W_loss") + pl.lit(self.smoothing_const))
                .otherwise(pl.col("W_loss"))
                .alias("denom_raw")
            )
        )

        # Optional cap: denom = W_loss + min(lambda_i, rho * W_loss)
        if self.smoothing_mode != "none" and math.isfinite(
            self.smoothing_cap_ratio
        ):
            denoms = denoms.with_columns(
                (
                    pl.col("W_loss")
                    + pl.min_horizontal(
                        (pl.col("denom_raw") - pl.col("W_loss")),
                        (pl.lit(self.smoothing_cap_ratio) * pl.col("W_loss")),
                    )
                ).alias("denom")
            )
        else:
            denoms = denoms.with_columns(pl.col("denom_raw").alias("denom"))

        # Also expose lambda_i for analysis: lambda = denom - W_loss
        denoms = denoms.with_columns(
            (pl.col("denom") - pl.col("W_loss")).alias("lambda")
        )

        # Join denominators and compute normalized weights WITHOUT forcing row sums to 1
        edges = (
            edges.join(
                denoms.select(["loser_user_id", "denom"]),
                on="loser_user_id",
                how="left",
            )
            .with_columns((pl.col("w_sum") / pl.col("denom")).alias("norm_w"))
            .select(["loser_user_id", "winner_user_id", "norm_w"])
        )

        # Store denominators for downstream analysis
        self.denominators_df_ = denoms.select(
            ["loser_user_id", "W_loss", "W_win", "lambda", "denom"]
        ).rename({"loser_user_id": "id"})

        return edges

    def _factorize_nodes(
        self, edges: pl.DataFrame, node_col_from: str, node_col_to: str
    ) -> tuple[list, pl.DataFrame]:
        """
        Map arbitrary node ids (ints/strings) to contiguous [0..n-1] indices
        using a fast join in Polars. Returns (nodes_list, edges_with_idx).
        """
        nodes_df = (
            pl.concat(
                [
                    edges.select(pl.col(node_col_from).alias("id")),
                    edges.select(pl.col(node_col_to).alias("id")),
                ]
            )
            .unique()
            .with_row_index("idx")  # idx: UInt32
        )
        edges_idx = (
            edges.join(
                nodes_df, left_on=node_col_from, right_on="id", how="left"
            )
            .rename({"idx": "src_idx"})
            .join(nodes_df, left_on=node_col_to, right_on="id", how="left")
            .rename({"idx": "dst_idx"})
            .select(["src_idx", "dst_idx", "norm_w"])
            .with_columns(
                [
                    pl.col("src_idx").cast(pl.UInt32),
                    pl.col("dst_idx").cast(pl.UInt32),
                    pl.col("norm_w").cast(pl.Float64),
                ]
            )
        )
        nodes = nodes_df.sort("idx")["id"].to_list()
        return nodes, edges_idx

    # =========================================================================
    # PageRank implementation
    # =========================================================================

    def _pagerank(
        self,
        edges: pl.DataFrame,
        *,
        node_col_from: str,
        node_col_to: str,
    ) -> pl.DataFrame:
        """
        Compute PageRank using a sparse transition matrix (row-stochastic with deficits).
        """
        self.logger.debug(f"Computing PageRank with {edges.height} edges")
        if edges.is_empty():
            self.logger.warning("Empty edges DataFrame, returning empty result")
            return pl.DataFrame([])

        # Map ids -> contiguous indices once
        nodes, eidx = self._factorize_nodes(edges, node_col_from, node_col_to)
        n = len(nodes)
        alpha = self.damping_factor

        # Build sparse matrix in CSR: rows=src, cols=dst, data=norm_w
        src = eidx["src_idx"].to_numpy()
        dst = eidx["dst_idx"].to_numpy()
        dat = eidx["norm_w"].to_numpy()

        if sp is not None:
            M = sp.csr_matrix((dat, (src, dst)), shape=(n, n))
            row_sums = np.asarray(M.sum(axis=1)).ravel()
        else:
            # NumPy sparse-like accumulation without SciPy
            row_sums = np.zeros(n, dtype=np.float64)
            # We'll keep COO triplets and use segment sums where needed
            M_coo = (src, dst, dat)
            np.add.at(row_sums, src, dat)

        # Teleport vector
        teleport_vec = self._make_teleport_vec(nodes, edges, node_col_from)
        assert abs(teleport_vec.sum() - 1.0) < 1e-12

        r = np.full(n, 1.0 / n, dtype=np.float64)
        base = (1.0 - alpha) * teleport_vec

        self.logger.debug(
            f"Starting sparse PageRank (max={self.max_pagerank_iter}, tol={self.pagerank_tol})"
        )
        for it in range(self.max_pagerank_iter):
            # deficit_mass = sum_i r[i] * (1 - row_sum[i])
            deficit_mass = float((1.0 - row_sums) @ r)

            if sp is not None:
                Mr = M.T.dot(r)
            else:
                # multiply using COO triplets: (src, dst, dat)
                Mr = np.zeros(n, dtype=np.float64)
                np.add.at(Mr, dst, r[src] * dat)

            r_new = base + alpha * Mr + alpha * deficit_mass * teleport_vec
            delta = np.abs(r_new - r).sum()
            if it % 20 == 0 or delta < self.pagerank_tol:
                self.logger.debug(f"PR iter {it}: delta={delta:.2e}")
            r = r_new
            if delta < self.pagerank_tol:
                break
        else:
            self.logger.warning(
                f"PageRank hit max_iter ({self.max_pagerank_iter}); final delta={delta:.2e}"
            )

        # Edge flux: only compute if requested (expensive for large graphs)
        if self.compute_edge_flux:
            self.logger.debug(
                "Computing edge flux information (real edges only, with alpha)"
            )
            node_scores = pl.DataFrame(
                {
                    node_col_from: nodes,
                    "r_src": r.tolist(),
                }
            )
            self.edge_flux_ = (
                edges.join(node_scores, on=node_col_from, how="left")
                .with_columns(
                    (pl.lit(alpha) * pl.col("r_src") * pl.col("norm_w")).alias(
                        "flux"
                    )
                )
                .select([node_col_from, node_col_to, "flux"])
            )
        else:
            self.edge_flux_ = None

        result = pl.DataFrame({"id": nodes, "score": r.tolist()}).sort(
            "score", descending=True
        )

        # Compute global_prior as 5th percentile of scores
        self.global_prior_ = float(np.quantile(r, 0.05))
        self.logger.debug(f"Global prior set to: {self.global_prior_:.6f}")

        self.logger.debug(
            f"PageRank completed: top score={result['score'][0]:.6f}, bottom score={result['score'][-1]:.6f}"
        )
        return result

    # =========================================================================
    # Tournament influence calculation
    # =========================================================================

    def _participants_by_tournament(
        self, matches: pl.DataFrame, players: pl.DataFrame | None, node_to: str
    ) -> pl.DataFrame:
        """Compute participants per tournament. Returns: [tournament_id, id] unique."""
        if node_to == "winner_team_id":
            return (
                matches.select(["tournament_id", "winner_team_id"])
                .rename({"winner_team_id": "id"})
                .vstack(
                    matches.select(["tournament_id", "loser_team_id"]).rename(
                        {"loser_team_id": "id"}
                    )
                )
                .unique()
            )
        else:
            return (
                players.join(
                    matches.unique(subset=["tournament_id"]).select(
                        ["tournament_id"]
                    ),
                    on="tournament_id",
                    how="inner",
                )
                .select(["tournament_id", pl.col("user_id").alias("id")])
                .unique()
            )

    def _compute_tournament_influence(
        self,
        rating_df: pl.DataFrame,
        matches: pl.DataFrame,
        players: pl.DataFrame | None,
        node_to: str,
        participants_df: pl.DataFrame | None = None,
    ) -> dict[int, float]:
        """
        Compute tournament influence from current ratings.

        This is the "tock" step that updates tournament strengths based on
        the ratings computed in the "tick" step.
        """
        # Use cached participants if provided, otherwise compute
        if participants_df is not None:
            part_df = participants_df
        else:
            part_df = self._participants_by_tournament(
                matches, players, node_to
            )

        # Add ratings to participant data (use global_prior instead of 0.0)
        global_prior = self.global_prior_ or 0.05
        rating_lookup = rating_df.select(
            [pl.col("id"), pl.col("score").alias("r")]
        )
        part_df = part_df.join(rating_lookup, on="id", how="left").with_columns(
            pl.col("r").fill_null(global_prior)
        )

        # Aggregate ratings by tournament using specified method
        if self.influence_agg_method == "mean":
            agg_expr = pl.col("r").mean()
        elif self.influence_agg_method == "sum":
            agg_expr = pl.col("r").sum()
        elif self.influence_agg_method == "median":
            agg_expr = pl.col("r").median()
        elif self.influence_agg_method == "top_20_sum":
            agg_expr = pl.col("r").top_k(20).sum()
        elif self.influence_agg_method == "log_top_20_sum":
            # Sum top 20, then apply log compression after normalization
            agg_expr = pl.col("r").top_k(20).sum()
        elif self.influence_agg_method == "sqrt_top_20_sum":
            # Take square root of top_20_sum for moderate compression
            # This provides less extreme compression than log
            agg_expr = pl.col("r").top_k(20).sum().sqrt()
        elif self.influence_agg_method == "top_10_sum":
            # Sum of top 10 players (less extreme than top 20)
            agg_expr = pl.col("r").top_k(10).sum()
        elif self.influence_agg_method == "top_20_mean":
            # Mean of top 20 players (normalizes for tournament size)
            agg_expr = pl.col("r").top_k(20).mean()
        else:
            raise ValueError(
                f"Unknown influence_agg_method: {self.influence_agg_method}"
            )

        # Compute tournament influences (normalized to mean=1.0 for proper beta scaling)
        S = (
            part_df.group_by("tournament_id")
            .agg(agg_expr.alias("S_raw"))
            .with_columns(
                (pl.col("S_raw") / pl.col("S_raw").mean()).alias("S_normalized")
            )
        )

        # For log_top_20_sum, apply logarithmic compression
        if self.influence_agg_method == "log_top_20_sum":
            # Apply log1p for numerical stability and compression
            S = S.with_columns(
                (pl.col("S_normalized").log1p()).alias("S_compressed")
            ).with_columns(
                # Rescale to maintain mean at 1
                (pl.col("S_compressed") / pl.col("S_compressed").mean()).alias(
                    "S"
                )
            )
        else:
            S = S.with_columns(pl.col("S_normalized").alias("S"))

        # Convert to dictionary
        tournament_influence = dict(zip(S["tournament_id"], S["S"]))

        # Apply minimum influence normalization if requested
        if self.normalize_min_influence is not None and tournament_influence:
            influences = list(tournament_influence.values())
            min_influence = min(influences)

            if min_influence < self.normalize_min_influence:
                # Shift all influences up so minimum equals normalize_min_influence
                shift = self.normalize_min_influence - min_influence

                tournament_influence = {
                    tid: influence + shift
                    for tid, influence in tournament_influence.items()
                }

                self.logger.info(
                    f"Normalized tournament influences: min {min_influence:.6f} → "
                    f"{self.normalize_min_influence:.6f} (shifted by {shift:.6f})"
                )

        return tournament_influence

    # =========================================================================
    # Retrospective tournament strength
    # =========================================================================

    def _compute_retro_strength(
        self,
        rating_df: pl.DataFrame,
        matches: pl.DataFrame,
        players: pl.DataFrame | None,
        node_to: str,
        influence: dict[int, float],
        participants_df: pl.DataFrame | None = None,
    ) -> None:
        """
        Compute retrospective tournament strength metrics.

        This creates a final tournament strength table that can be used
        for analysis and comparison purposes.
        """
        # No longer needed with joins

        # Build participant table (same logic as influence calculation)
        if participants_df is not None:
            part_df = participants_df
        else:
            part_df = self._participants_by_tournament(
                matches, players, node_to
            )

        rating_lookup = rating_df.select(
            [pl.col("id"), pl.col("score").alias("r")]
        )
        part_df = part_df.join(rating_lookup, on="id", how="left").with_columns(
            pl.col("r").fill_null(self.global_prior_ or 0.05)
        )

        # Apply aggregation method for retrospective strength
        grp = part_df.group_by("tournament_id")
        if self.strength_agg == "mean":
            agg_df = grp.agg(pl.col("r").mean().alias("raw"))
        elif self.strength_agg == "median":
            agg_df = grp.agg(pl.col("r").median().alias("raw"))
        elif self.strength_agg == "trimmed_mean":
            k = self.strength_k
            agg_df = grp.agg(
                pl.col("r")
                .sort()
                .slice(k, pl.len() - 2 * k)
                .mean()
                .alias("raw")
            )
        elif self.strength_agg == "topN_sum":
            n = self.strength_k
            agg_df = grp.agg(pl.col("r").top_k(n).sum().alias("raw"))
        else:
            raise ValueError(f"Unknown strength_agg: {self.strength_agg}")

        # Use raw strength values (not normalized)
        strength_df = agg_df.rename({"raw": "strength"}).select(
            ["tournament_id", "strength"]
        )

        # Merge with influence values
        infl_df = pl.DataFrame(
            {
                "tournament_id": list(influence.keys()),
                "influence": list(influence.values()),
            }
        )
        self.tournament_strength = strength_df.join(
            infl_df, on="tournament_id", how="left"
        ).select(["tournament_id", "influence", "strength"])

    # =========================================================================
    # Confidence calculation
    # =========================================================================

    def calculate_player_confidence(
        self,
        matches: pl.DataFrame,
        players: pl.DataFrame,
        rankings: Optional[pl.DataFrame] = None,
        include_quality_metrics: bool = True,
    ) -> pl.DataFrame:
        """
        Calculate confidence metrics for all players.

        Parameters
        ----------
        matches : pl.DataFrame
            Matches DataFrame
        players : pl.DataFrame
            Players DataFrame
        rankings : Optional[pl.DataFrame]
            Player rankings (uses self.ratings_df if not provided)
        include_quality_metrics : bool
            Whether to include detailed quality metrics

        Returns
        -------
        pl.DataFrame
            DataFrame with player_id and confidence metrics
        """
        self.logger.info("Calculating player confidence metrics")

        # Use provided rankings or last computed rankings
        if rankings is None:
            if self.ratings_df is None:
                raise ValueError(
                    "No rankings available. Run rank_players first."
                )
            rankings = self.ratings_df

        # Convert rankings to dict for fast lookup
        player_ranks = dict(enumerate(rankings["id"].to_list(), 1))
        player_ranks = {v: k for k, v in player_ranks.items()}

        # Precompute rosters for all teams (massive speedup)
        team_rosters = players.group_by(["tournament_id", "team_id"]).agg(
            pl.col("user_id").alias("roster")
        )

        # Join rosters to matches once
        matches_with_rosters = (
            matches.filter(~pl.col("is_bye").fill_null(False))
            .join(
                team_rosters,
                left_on=["tournament_id", "winner_team_id"],
                right_on=["tournament_id", "team_id"],
                how="inner",
            )
            .rename({"roster": "winners"})
            .join(
                team_rosters,
                left_on=["tournament_id", "loser_team_id"],
                right_on=["tournament_id", "team_id"],
                how="inner",
            )
            .rename({"roster": "losers"})
            .select(["tournament_id", "winners", "losers"])
        )

        # Build match data structure using vectorized operations
        from collections import defaultdict

        player_matches = defaultdict(list)

        for row in matches_with_rosters.iter_rows(named=True):
            match_record = {
                "tournament_id": row["tournament_id"],
                "winners": row["winners"],
                "losers": row["losers"],
            }

            for w in row["winners"]:
                if w:
                    player_matches[w].append(match_record)

            for l in row["losers"]:
                if l:
                    player_matches[l].append(match_record)

        # Initialize confidence calculator
        total_players = len(player_ranks)
        calculator = AsymmetricConfidenceCalculator(
            total_players=total_players,
            player_ranks=player_ranks,
        )

        # Calculate confidence for all players
        self.logger.info(
            f"Calculating confidence for {len(player_matches)} players"
        )
        confidence_results = []
        self.confidence_metrics_ = {}

        for player_id, matches_list in player_matches.items():
            metrics = calculator.calculate_asymmetric_confidence(
                player_id=player_id,
                matches=matches_list,
            )
            self.confidence_metrics_[player_id] = metrics

            # Build result row
            result = {
                "player_id": player_id,
                "confidence": metrics.overall_confidence,
                "confidence_tier": metrics.confidence_tier.name,
                "rankability": metrics.rankability,
                "tournaments": metrics.tournament_count,
                "matches": metrics.match_count,
                "unique_opponents": metrics.unique_opponents,
            }

            if include_quality_metrics:
                result.update(
                    {
                        "top_opponents_faced": metrics.top_opponents_faced,
                        "top_players_beaten": metrics.top_players_beaten,
                        "beaten_by_top": metrics.beaten_by_top,
                        "upward_mobility": metrics.upward_mobility,
                        "downward_pressure": metrics.downward_pressure,
                        "incoming_edge_quality": metrics.incoming_edge_quality,
                        "outgoing_edge_quality": metrics.outgoing_edge_quality,
                    }
                )

            confidence_results.append(result)

        # Create DataFrame
        confidence_df = pl.DataFrame(confidence_results)

        # Add player rank if available
        if rankings is not None:
            # Get the actual column names from rankings
            id_col = "user_id" if "user_id" in rankings.columns else "id"
            score_col = (
                "player_rank" if "player_rank" in rankings.columns else "score"
            )

            if score_col in rankings.columns:
                confidence_df = confidence_df.join(
                    rankings.select([id_col, score_col]).rename(
                        {id_col: "player_id", score_col: "rating"}
                    ),
                    on="player_id",
                    how="left",
                )

        # Sort by confidence
        confidence_df = confidence_df.sort("confidence", descending=True)

        # Log summary statistics
        tier_counts = confidence_df.group_by("confidence_tier").len()
        self.logger.info("Confidence tier distribution:")
        for row in tier_counts.iter_rows(named=True):
            self.logger.info(
                f"  {row['confidence_tier']}: {row['len']} players"
            )

        avg_confidence = confidence_df["confidence"].mean()
        self.logger.info(f"Average confidence: {avg_confidence:.3f}")

        return confidence_df

    def rank_players_with_confidence(
        self,
        matches: pl.DataFrame,
        players: pl.DataFrame,
        return_confidence: bool = True,
    ) -> pl.DataFrame:
        """
        Rank players and optionally calculate confidence metrics.

        This is a convenience method that combines ranking and confidence calculation.

        Parameters
        ----------
        matches : pl.DataFrame
            Matches DataFrame
        players : pl.DataFrame
            Players DataFrame
        return_confidence : bool
            Whether to include confidence metrics in the result

        Returns
        -------
        pl.DataFrame
            Player rankings with optional confidence metrics
        """
        # First rank players
        rankings = self.rank_players(matches, players)

        if not return_confidence:
            return rankings

        # Calculate confidence
        confidence_df = self.calculate_player_confidence(
            matches, players, rankings, include_quality_metrics=False
        )

        # Merge rankings with confidence
        # Get the ID column name from rankings
        id_col = "user_id" if "user_id" in rankings.columns else "id"

        result = rankings.join(
            confidence_df.select(
                [
                    "player_id",
                    "confidence",
                    "confidence_tier",
                    "rankability",
                    "tournaments",
                    "matches",
                    "unique_opponents",
                ]
            ).rename({"player_id": id_col}),
            on=id_col,
            how="left",
        )

        # Fill any missing confidence values (players with no matches)
        result = result.with_columns(
            [
                pl.col("confidence").fill_null(0.0),
                pl.col("confidence_tier").fill_null("PROVISIONAL"),
                pl.col("rankability").fill_null("unrankable"),
                pl.col("tournaments").fill_null(0),
                pl.col("matches").fill_null(0),
                pl.col("unique_opponents").fill_null(0),
            ]
        )

        return result

    # =========================================================================
    # Helper functions
    # =========================================================================

    def _make_teleport_vec(
        self,
        nodes: List[int],
        edges: pl.DataFrame,
        node_col_from: str,
    ) -> np.ndarray:
        """Create teleport vector based on specification.

        Supports multiple teleport strategies:
        - uniform: Equal probability to all nodes
        - volume_inverse: Inverse sqrt of loss count (down-weight grinders)
        - volume_mix: Mixture of uniform and volume-weighted components
        - dict: Custom teleport probabilities
        """
        n = len(nodes)

        if isinstance(self.teleport_spec, dict):
            v = np.array([self.teleport_spec.get(node, 0.0) for node in nodes])

        elif self.teleport_spec == TELEPORT_VOLUME_INVERSE:
            counts = edges[node_col_from].value_counts()
            c_map = dict(zip(counts[node_col_from], counts["count"]))
            v = np.array(
                [1.0 / math.sqrt(c_map.get(node, 1)) for node in nodes]
            )

        elif self.teleport_spec == TELEPORT_VOLUME_MIX:
            # For volume mix, we need to count the number of losses (edges from node)
            # Since edges are already normalized, we'll use edge counts as proxy for volume
            edge_counts = edges.group_by(node_col_from).agg(
                pl.count().alias("edge_count")
            )
            count_map = dict(
                zip(edge_counts[node_col_from], edge_counts["edge_count"])
            )

            # Volume component: (epsilon + count)^gamma
            v_vol = np.array(
                [
                    (self.volume_epsilon + count_map.get(node, 0.0))
                    ** self.volume_mix_gamma
                    for node in nodes
                ]
            )
            v_vol = v_vol / v_vol.sum()  # Normalize

            # Uniform component
            v_unif = np.ones(n) / n

            # Mix components
            v = (1 - self.volume_mix_eta) * v_unif + self.volume_mix_eta * v_vol

        else:  # uniform
            v = np.ones(n)

        # Ensure normalization with epsilon smoothing for numerical stability
        total = v.sum()
        if total == 0.0 or not np.isfinite(total):
            v = np.ones(n) / n
        else:
            v = v / total
            # Add small epsilon to avoid exact zeros (helps with dangling nodes)
            epsilon = 1e-10
            v = v * (1 - epsilon * n) + epsilon

        return v

    def _event_ts_expr(self) -> pl.Expr:
        """Create expression for event timestamp."""
        # Check which timestamp columns are available and use them
        # This allows flexibility in the match data format
        if hasattr(self, "_current_df_columns"):
            cols = self._current_df_columns
        else:
            # Default behavior when we don't know the columns
            return pl.col("match_created_at").fill_null(self.now_ts)

        if "last_game_finished_at" in cols:
            return (
                pl.when(pl.col("last_game_finished_at").is_not_null())
                .then(pl.col("last_game_finished_at"))
                .otherwise(pl.col("match_created_at"))
                .fill_null(self.now_ts)
            )
        elif "match_created_at" in cols:
            return pl.col("match_created_at").fill_null(self.now_ts)
        else:
            # Fallback to current timestamp if no timestamp columns
            return pl.lit(self.now_ts)

    def _decay_expr(self, ts_col: str) -> pl.Expr:
        """Create expression for time decay."""
        return (
            ((self.now_ts - pl.col(ts_col).cast(pl.Float64)) / 86400.0)
            .mul(-self.decay_rate)
            .exp()
        )

    @staticmethod
    def _row_normalize(
        edges: pl.DataFrame, group_col: str, to_col: str | None = None
    ) -> pl.DataFrame:
        """Normalize edge weights so each source node sums to 1.0.

        Parameters
        ----------
        edges : pl.DataFrame
            Contains columns [group_col, to_col, 'w_sum']
        group_col : str
            Source node column (row to normalize)
        to_col : str | None
            Destination node column; if None, uses edges.columns[1]
        """
        dest = to_col or edges.columns[1]
        return (
            edges.with_columns(
                pl.col("w_sum").sum().over(group_col).alias("tot_w")
            )
            .with_columns((pl.col("w_sum") / pl.col("tot_w")).alias("norm_w"))
            .select([group_col, dest, "norm_w"])
        )
