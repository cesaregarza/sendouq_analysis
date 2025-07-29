"""
Advanced tournament ranking engine with iterative tournament strength calculation.

This module implements a sophisticated rating system based on the RatingEngine from test_tour_4.
It uses a tick-tock algorithm that iteratively refines both player/team ratings and tournament
strength values until convergence.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

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
    TELEPORT_UNIFORM,
    TELEPORT_VOLUME_INVERSE,
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
        Teleport vector specification ("uniform", "volume_inverse", or dict)
    influence_agg_method : str, default="mean"
        Method for aggregating participant ratings into tournament influence
    strength_agg : str, default="mean"
        Method for computing retrospective tournament strength
    strength_k : int, default=5
        Parameter for trimmed_mean or topN_sum strength aggregation
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
        now: Optional[datetime] = None,
        teleport: str | dict[int, float] = TELEPORT_VOLUME_INVERSE,
        influence_agg_method: Literal[
            "mean", "sum", "median", "top_20_sum"
        ] = DEFAULT_INFLUENCE_AGG_METHOD,
        strength_agg: Literal[
            "mean", "median", "trimmed_mean", "topN_sum"
        ] = DEFAULT_STRENGTH_AGG,
        strength_k: int = DEFAULT_STRENGTH_K,
    ):
        self.logger = get_logger(__name__)
        self.logger.info("Initializing RatingEngine with parameters:")
        self.logger.info(f"  decay_half_life_days={decay_half_life_days}")
        self.logger.info(f"  damping_factor={damping_factor}")
        self.logger.info(f"  beta={beta}")
        self.logger.info(f"  tick_tock_stabilize_tol={tick_tock_stabilize_tol}")
        self.logger.info(f"  max_tick_tock={max_tick_tock}")
        self.logger.info(f"  influence_agg_method={influence_agg_method}")
        self.decay_half_life_days = decay_half_life_days
        self.decay_rate = math.log(2.0) / decay_half_life_days
        self.damping_factor = damping_factor
        self.beta = beta
        self.tick_tock_stabilize_tol = tick_tock_stabilize_tol
        self.max_tick_tock = max_tick_tock
        self.max_pagerank_iter = max_pagerank_iter
        self.pagerank_tol = pagerank_tol
        self.now = (now or DEFAULT_REFERENCE_DATE).astimezone(timezone.utc)
        self.teleport_spec = teleport
        self.influence_agg_method = influence_agg_method
        self.strength_agg = strength_agg
        self.strength_k = strength_k

        # Results populated after a run
        self.edge_flux_: Optional[pl.DataFrame] = None
        self.tournament_influence_: Optional[Dict[int, float]] = None
        self.tournament_strength: Optional[pl.DataFrame] = None
        self.ratings_df: Optional[pl.DataFrame] = None
        self.global_prior_: Optional[float] = None

    @property
    def tournament_influence(self) -> Optional[Dict[int, float]]:
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
        log_dataframe_stats(self.logger, matches, "team_matches")

        with log_timing(self.logger, "team ranking calculation"):
            result = self._tick_tock_loop(
                matches=matches,
                players=None,
                node_from="loser_team_id",
                node_to="winner_team_id",
            )

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
        log_dataframe_stats(self.logger, matches, "matches")
        log_dataframe_stats(self.logger, players, "players")

        with log_timing(self.logger, "player ranking calculation"):
            result = self._tick_tock_loop(
                matches=matches,
                players=players,
                node_from="loser_user_id",
                node_to="winner_user_id",
            )

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
        players: Optional[pl.DataFrame],
        node_from: str,
        node_to: str,
    ) -> pl.DataFrame:
        """
        Core tick-tock iteration loop.

        Alternates between computing ratings (tick) and updating tournament
        strengths (tock) until convergence.
        """
        self.logger.debug(
            f"Starting tick-tock loop with max_iterations={self.max_tick_tock}"
        )
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
                        rating_df, matches, players, node_to
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
            rating_df, matches, players, node_to, tournament_influence
        )
        self.ratings_df = rating_df

        # Return with appropriate column name
        score_name = (
            "team_rank" if node_from.startswith("loser_team") else "player_rank"
        )
        result = rating_df.rename({"score": score_name}).sort(
            score_name, descending=True
        )

        log_dataframe_stats(self.logger, result, "final_tick_tock_result")
        self.logger.info(
            f"Tick-tock loop completed, returning {result.height} ranked entities"
        )
        return result

    # =========================================================================
    # Edge building functions
    # =========================================================================

    def _build_team_edges(
        self, matches: pl.DataFrame, S: Dict[int, float]
    ) -> pl.DataFrame:
        """Build team-level edges with tournament strength weighting."""
        if matches.is_empty():
            return pl.DataFrame([])

        # Create tournament strength lookup
        s_df = pl.DataFrame(
            {"tournament_id": list(S.keys()), "S": list(S.values())}
        )

        # Filter valid matches (exclude byes/forfeits) and join strengths
        df = (
            matches.filter(
                pl.col("winner_team_id").is_not_null()
                & pl.col("loser_team_id").is_not_null()
                & ~pl.col("is_bye", default=False)
            )
            .join(s_df, on="tournament_id", how="left")
            .fill_null(1.0)  # Default strength for unseen tournaments
        )

        # Add event timestamps and compute weights with influence decay
        df = df.with_columns(self._event_ts_expr().alias("ts"))

        # Compute age in days
        age_days = (
            int(self.now.timestamp()) - pl.col("ts").cast(pl.Float64)
        ) / 86400.0

        # Apply influence decay: λ = log(2)/365 (1-year half-life for influence)
        influence_decay_rate = math.log(2) / 365.0
        influence_decay = (-influence_decay_rate * age_days).exp()

        df = df.with_columns(
            (
                self._decay_expr("ts")
                * ((pl.col("S") * influence_decay) ** self.beta)
            ).alias("w")
        )

        # Aggregate and normalize edges
        grouped = df.group_by(["loser_team_id", "winner_team_id"]).agg(
            pl.col("w").sum().alias("w_sum")
        )
        return self._row_normalize(grouped, "loser_team_id")

    def _build_player_edges(
        self,
        matches: pl.DataFrame,
        players: pl.DataFrame,
        S: Dict[int, float],
    ) -> pl.DataFrame:
        """Build player-level edges with tournament strength weighting."""
        if matches.is_empty() or players.is_empty():
            return pl.DataFrame([])

        # Tournament strength lookup
        s_df = pl.DataFrame(
            {"tournament_id": list(S.keys()), "S": list(S.values())}
        )

        # Prepare matches with strengths and weights (exclude byes/forfeits)
        m = (
            matches.filter(
                pl.col("winner_team_id").is_not_null()
                & pl.col("loser_team_id").is_not_null()
                & ~pl.col("is_bye", default=False)
            )
            .join(s_df, on="tournament_id", how="left")
            .fill_null(1.0)
        )
        m = m.with_columns(self._event_ts_expr().alias("ts"))

        # Compute age in days and apply influence decay
        age_days = (
            int(self.now.timestamp()) - pl.col("ts").cast(pl.Float64)
        ) / 86400.0
        influence_decay_rate = math.log(2) / 365.0
        influence_decay = (-influence_decay_rate * age_days).exp()

        m = m.with_columns(
            (
                self._decay_expr("ts")
                * ((pl.col("S") * influence_decay) ** self.beta)
            ).alias("match_w")
        )

        # Player selection for joins
        pl_sel = players.select(["tournament_id", "team_id", "user_id"])

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

        # Aggregate and normalize
        edges = pairs.group_by(["loser_user_id", "winner_user_id"]).agg(
            pl.col("match_w").sum().alias("w_sum")
        )
        return self._row_normalize(edges, "loser_user_id")

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
        Compute PageRank from edge table with flexible teleport vectors.
        """
        self.logger.debug(f"Computing PageRank with {edges.height} edges")

        if edges.is_empty():
            self.logger.warning("Empty edges DataFrame, returning empty result")
            return pl.DataFrame([])

        # Extract unique nodes
        nodes = sorted(
            set(edges[node_col_from].unique())
            | set(edges[node_col_to].unique())
        )
        n = len(nodes)
        self.logger.debug(f"PageRank graph has {n} nodes")
        index: Dict[int, int] = {v: i for i, v in enumerate(nodes)}

        # Build transition matrix
        self.logger.debug("Building transition matrix")
        M = np.zeros((n, n), dtype=float)
        for row in edges.iter_rows(named=True):
            i = index[row[node_col_from]]
            j = index[row[node_col_to]]
            M[i, j] += row["norm_w"]

        # Handle dangling nodes (no outgoing edges)
        row_sums = M.sum(axis=1)
        dangling_nodes = np.sum(row_sums == 0.0)
        if dangling_nodes > 0:
            self.logger.debug(
                f"Found {dangling_nodes} dangling nodes, fixing with uniform distribution"
            )
            M[row_sums == 0.0] = 1.0 / n

        # Create teleport vector
        self.logger.debug(
            f"Creating teleport vector with spec: {self.teleport_spec}"
        )
        teleport_vec = self._make_teleport_vec(nodes, edges, node_col_from)
        assert abs(teleport_vec.sum() - 1.0) < 1e-12

        # PageRank power iteration
        alpha = self.damping_factor
        r = np.full(n, 1.0 / n)
        base = (1.0 - alpha) * teleport_vec

        self.logger.debug(
            f"Starting PageRank iterations (max={self.max_pagerank_iter}, tol={self.pagerank_tol})"
        )

        for iteration in range(self.max_pagerank_iter):
            r_new = base + alpha * M.T.dot(r)
            delta = np.abs(r_new - r).sum()

            if iteration % 20 == 0 or delta < self.pagerank_tol:
                self.logger.debug(
                    f"PageRank iteration {iteration}: delta={delta:.2e}"
                )

            if delta < self.pagerank_tol:
                self.logger.debug(
                    f"PageRank converged at iteration {iteration} with delta {delta:.2e}"
                )
                r = r_new
                break
            r = r_new
        else:
            self.logger.warning(
                f"PageRank reached max iterations ({self.max_pagerank_iter}) without convergence. Final delta: {delta:.2e}"
            )

        # Store edge flux information for analysis
        self.logger.debug("Computing edge flux information")
        flux = M * r[:, None]
        li, wi = np.nonzero(M)
        self.edge_flux_ = pl.DataFrame(
            {
                node_col_from: [nodes[i] for i in li],
                node_col_to: [nodes[j] for j in wi],
                "flux": flux[li, wi],
            }
        )

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

    def _compute_tournament_influence(
        self,
        rating_df: pl.DataFrame,
        matches: pl.DataFrame,
        players: Optional[pl.DataFrame],
        node_to: str,
    ) -> Dict[int, float]:
        """
        Compute tournament influence from current ratings.

        This is the "tock" step that updates tournament strengths based on
        the ratings computed in the "tick" step.
        """
        rating_map = dict(zip(rating_df["id"], rating_df["score"]))

        if node_to == "winner_team_id":  # Team mode
            # Get all team participants per tournament
            part_df = (
                matches.select(["tournament_id", "winner_team_id"])
                .rename({"winner_team_id": "id"})
                .vstack(
                    matches.select(["tournament_id", "loser_team_id"]).rename(
                        {"loser_team_id": "id"}
                    )
                )
                .unique()
            )
        else:  # Player mode
            # Get all player participants per tournament
            part_df = (
                players.join(
                    matches.unique(subset=["tournament_id"]).select(
                        ["tournament_id"]
                    ),
                    on="tournament_id",
                    how="inner",
                )
                .select(["tournament_id", "user_id"])
                .rename({"user_id": "id"})
            )

        # Add ratings to participant data (use global_prior instead of 0.0)
        global_prior = self.global_prior_ or 0.05
        part_df = part_df.with_columns(
            pl.col("id")
            .map_elements(
                lambda x: rating_map.get(x, global_prior),
                return_dtype=pl.Float64,
            )
            .alias("r")
        )

        # Aggregate ratings by tournament using specified method
        if self.influence_agg_method == "mean":
            agg_expr = pl.col("r").mean()
        elif self.influence_agg_method == "sum":
            agg_expr = pl.col("r").sum()
        elif self.influence_agg_method == "median":
            agg_expr = pl.col("r").median()
        elif self.influence_agg_method == "top_20_sum":
            agg_expr = pl.col("r").sort(descending=True).head(20).sum()
        else:
            raise ValueError(
                f"Unknown influence_agg_method: {self.influence_agg_method}"
            )

        # Compute tournament influences (normalized to mean 1.0)
        S = (
            part_df.group_by("tournament_id")
            .agg(agg_expr.alias("S_raw"))
            .with_columns((pl.col("S_raw") / pl.col("S_raw").mean()).alias("S"))
        )
        return dict(zip(S["tournament_id"], S["S"]))

    # =========================================================================
    # Retrospective tournament strength
    # =========================================================================

    def _compute_retro_strength(
        self,
        rating_df: pl.DataFrame,
        matches: pl.DataFrame,
        players: Optional[pl.DataFrame],
        node_to: str,
        influence: Dict[int, float],
    ) -> None:
        """
        Compute retrospective tournament strength metrics.

        This creates a final tournament strength table that can be used
        for analysis and comparison purposes.
        """
        rating_map = dict(zip(rating_df["id"], rating_df["score"]))

        # Build participant table (same logic as influence calculation)
        if node_to == "winner_team_id":  # Team mode
            part_df = (
                matches.select(["tournament_id", "winner_team_id"])
                .rename({"winner_team_id": "id"})
                .vstack(
                    matches.select(["tournament_id", "loser_team_id"]).rename(
                        {"loser_team_id": "id"}
                    )
                )
                .unique()
            )
        else:  # Player mode
            part_df = (
                players.join(
                    matches.unique(subset=["tournament_id"]).select(
                        ["tournament_id"]
                    ),
                    on="tournament_id",
                    how="inner",
                )
                .select(["tournament_id", "user_id"])
                .rename({"user_id": "id"})
            )

        part_df = part_df.with_columns(
            pl.col("id")
            .map_elements(
                lambda x: rating_map.get(x, self.global_prior_ or 0.05),
                return_dtype=pl.Float64,
            )
            .alias("r")
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
                pl.col("r").sort().tail(-k).head(-k).mean().alias("raw")
            )
        elif self.strength_agg == "topN_sum":
            n = self.strength_k
            agg_df = grp.agg(
                pl.col("r").sort(descending=True).head(n).sum().alias("raw")
            )
        else:
            raise ValueError(f"Unknown strength_agg: {self.strength_agg}")

        # Normalize to mean 1.0
        strength_df = agg_df.with_columns(
            (pl.col("raw") / pl.col("raw").mean()).alias("strength")
        ).select(["tournament_id", "strength"])

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
    # Helper functions
    # =========================================================================

    def _make_teleport_vec(
        self,
        nodes: List[int],
        edges: pl.DataFrame,
        node_col_from: str,
    ) -> np.ndarray:
        """Create teleport vector based on specification."""
        n = len(nodes)
        if isinstance(self.teleport_spec, dict):
            v = np.array([self.teleport_spec.get(node, 0.0) for node in nodes])
        elif self.teleport_spec == TELEPORT_VOLUME_INVERSE:
            counts = edges[node_col_from].value_counts()
            c_map = dict(zip(counts[node_col_from], counts["counts"]))
            v = np.array(
                [1.0 / math.sqrt(c_map.get(node, 1)) for node in nodes]
            )
        else:  # uniform
            v = np.ones(n)

        total = v.sum()
        if total == 0.0:
            v = np.ones(n) / n
        else:
            v /= total
        return v

    def _event_ts_expr(self) -> pl.Expr:
        """Create expression for event timestamp."""
        return (
            pl.when(pl.col("last_game_finished_at").is_not_null())
            .then(pl.col("last_game_finished_at"))
            .otherwise(pl.col("match_created_at"))
            .fill_null(int(self.now.timestamp()))
        )

    def _decay_expr(self, ts_col: str) -> pl.Expr:
        """Create expression for time decay."""
        return (
            (
                (int(self.now.timestamp()) - pl.col(ts_col).cast(pl.Float64))
                / 86400.0
            )
            .mul(-self.decay_rate)
            .exp()
        )

    @staticmethod
    def _row_normalize(edges: pl.DataFrame, group_col: str) -> pl.DataFrame:
        """Normalize edge weights so each source node sums to 1.0."""
        totals = edges.group_by(group_col).agg(
            pl.col("w_sum").sum().alias("tot_w")
        )
        return (
            edges.join(totals, on=group_col)
            .with_columns((pl.col("w_sum") / pl.col("tot_w")).alias("norm_w"))
            .select([group_col, edges.columns[1], "norm_w"])
        )
