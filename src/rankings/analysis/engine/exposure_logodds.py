"""
Exposure Log-Odds Ranking Implementation.

This module implements the exposure log-odds ranking system that removes volume bias
by using two PageRanks on mirrored graphs with the same exposure baseline, then
taking a log-ratio so volume cancels and only conversion quality remains.

Based on the corrected specification in plan.md with fixes for:
1. Matrix orientation (A[dst, src] with column-stochastic normalization)
2. Team weight division to preserve total match mass
3. Exact graph mirroring (loss graph = transpose of win graph)
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
import polars as pl

# Try to import scipy for sparse matrix operations
try:
    import scipy.sparse as sp
except Exception:
    sp = None

from rankings.analysis.engine.core import RatingEngine
from rankings.core.logging import get_logger, log_dataframe_stats, log_timing


class ExposureLogOddsEngine(RatingEngine):
    """
    Exposure log-odds rating engine that eliminates volume bias.

    This engine computes two PageRanks (win and loss) with the same exposure-based
    teleport vector, then uses the log-ratio to produce rankings that depend on
    conversion quality rather than play volume.

    Parameters
    ----------
    lambda_smooth : Optional[float], default=None
        Smoothing parameter for log-ratio. If None, auto-tuned to 2.5% of median PageRank.
    use_surprisal : bool, default=False
        Whether to use surprisal weighting (upset bonus)
    surprisal_T : float, default=1.0
        Temperature for surprisal calculation
    surprisal_iters : int, default=2
        Number of iterations for surprisal refinement
    epsilon : float, default=1e-9
        Small value for numerical stability
    min_exposure : Optional[float], default=None
        Minimum exposure threshold for ranking eligibility
    score_decay_delay_days : float, default=30.0
        Number of days of inactivity before decay starts
    score_decay_rate : float, default=0.01
        Daily decay rate after delay period (e.g., 0.01 = 1% per day)
    **kwargs
        Additional parameters passed to base RatingEngine
    """

    def __init__(
        self,
        *,
        lambda_smooth: float | None = None,
        use_surprisal: bool = False,
        surprisal_T: float = 1.0,
        surprisal_iters: int = 2,
        epsilon: float = 1e-9,
        min_exposure: float | None = None,
        score_decay_delay_days: float = 30.0,
        score_decay_rate: float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lambda_smooth = lambda_smooth
        self.use_surprisal = use_surprisal
        self.surprisal_T = surprisal_T
        self.surprisal_iters = surprisal_iters
        self.epsilon = epsilon
        self.min_exposure = min_exposure
        self.score_decay_delay_days = score_decay_delay_days
        self.score_decay_rate = score_decay_rate
        self.logger = get_logger(__name__)

        self.logger.info("Initializing ExposureLogOddsEngine")
        self.logger.info(
            f"  lambda_smooth={'auto' if lambda_smooth is None else lambda_smooth}"
        )
        self.logger.info(f"  use_surprisal={use_surprisal}")
        if use_surprisal:
            self.logger.info(f"  surprisal_T={surprisal_T}")
            self.logger.info(f"  surprisal_iters={surprisal_iters}")
        if min_exposure:
            self.logger.info(f"  min_exposure={min_exposure}")

        # Store intermediate results for analysis
        self.exposure_vector_: np.ndarray | None = None
        self.exposure_teleport_: np.ndarray | None = None
        self.win_pagerank_: np.ndarray | None = None
        self.loss_pagerank_: np.ndarray | None = None
        self.logodds_scores_: np.ndarray | None = None
        self.lambda_used_: float | None = None

    def rank_players(
        self, matches: pl.DataFrame, players: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Rank players using exposure log-odds algorithm.

        Parameters
        ----------
        matches : pl.DataFrame
            Matches DataFrame
        players : pl.DataFrame
            Players DataFrame with tournament_id, team_id, user_id columns

        Returns
        -------
        pl.DataFrame
            Player rankings with columns: id, player_rank, win_pr, loss_pr, exposure
        """
        self.logger.info("Starting exposure log-odds player ranking")
        log_dataframe_stats(self.logger, matches, "matches")
        log_dataframe_stats(self.logger, players, "players")

        with log_timing(self.logger, "exposure log-odds calculation"):
            # First, get active players and tournament influences using standard method
            self.logger.info("Getting active players via standard ranking...")
            standard_result = self._tick_tock_loop(
                matches=matches,
                players=players,
                node_from="loser_user_id",
                node_to="winner_user_id",
            )

            active_players = standard_result["id"].to_list()
            self.logger.info(f"Found {len(active_players)} active players")

            tournament_influence = self.tournament_influence_ or {}

            # Convert matches to the required format
            self.logger.info("Converting match data...")
            try:
                matches_df = self._convert_matches_dataframe(
                    matches, players, tournament_influence
                )
                use_optimized = True
            except Exception as e:
                self.logger.debug(
                    f"Falling back to non-optimized conversion: {e}"
                )
                converted_matches = self._convert_matches_format(
                    matches, players, tournament_influence
                )
                use_optimized = False

            if self.use_surprisal and not use_optimized:
                (
                    scores,
                    win_pr,
                    loss_pr,
                    rho,
                    lambda_used,
                ) = self._exposure_logodds_with_surprisal(
                    converted_matches, active_players
                )
            elif use_optimized:
                (
                    scores,
                    win_pr,
                    loss_pr,
                    rho,
                    lambda_used,
                ) = self._exposure_logodds_optimized(matches_df, active_players)
            else:
                (
                    scores,
                    win_pr,
                    loss_pr,
                    rho,
                    lambda_used,
                ) = self._exposure_logodds(converted_matches, active_players)

            self.win_pagerank_ = win_pr
            self.loss_pagerank_ = loss_pr
            self.exposure_teleport_ = rho
            self.logodds_scores_ = scores
            self.lambda_used_ = lambda_used

            if use_optimized:
                node_to_idx = {p: i for i, p in enumerate(active_players)}
                exposure = np.zeros(len(active_players))

                # Explode winners and losers with their weights
                winners_exp = (
                    matches_df.select(["winners", "weight"])
                    .explode("winners")
                    .rename({"winners": "id"})
                )

                losers_exp = (
                    matches_df.select(["losers", "weight"])
                    .explode("losers")
                    .rename({"losers": "id"})
                )

                exp_df = (
                    pl.concat([winners_exp, losers_exp])
                    .group_by("id")
                    .agg(pl.col("weight").sum().alias("e"))
                )

                for row in exp_df.iter_rows(named=True):
                    if row["id"] in node_to_idx:
                        exposure[node_to_idx[row["id"]]] = row["e"]
            else:
                node_to_idx = {p: i for i, p in enumerate(active_players)}
                exposure = np.zeros(len(active_players))
                for match in converted_matches:
                    w = match["weight"]
                    for p in match["winners"]:
                        if p in node_to_idx:
                            exposure[node_to_idx[p]] += w
                    for p in match["losers"]:
                        if p in node_to_idx:
                            exposure[node_to_idx[p]] += w

            self.exposure_vector_ = exposure

            player_last_match = {}
            if use_optimized:
                winners_ts = (
                    matches_df.select(["winners", "ts"])
                    .explode("winners")
                    .rename({"winners": "id"})
                )

                losers_ts = (
                    matches_df.select(["losers", "ts"])
                    .explode("losers")
                    .rename({"losers": "id"})
                )

                ts_df = (
                    pl.concat([winners_ts, losers_ts])
                    .group_by("id")
                    .agg(pl.col("ts").max().alias("last_ts"))
                )

                for row in ts_df.iter_rows(named=True):
                    player_last_match[row["id"]] = row["last_ts"]
            else:
                for match in converted_matches:
                    match_ts = self.now_ts  # Default to now
                    if "timestamp" in match:
                        match_ts = match["timestamp"]

                    for p in match["winners"]:
                        if p in node_to_idx:
                            player_last_match[p] = max(
                                player_last_match.get(p, 0), match_ts
                            )
                    for p in match["losers"]:
                        if p in node_to_idx:
                            player_last_match[p] = max(
                                player_last_match.get(p, 0), match_ts
                            )

            now_ts = self.now_ts
            decay_factors = np.ones(len(active_players))

            for i, player_id in enumerate(active_players):
                last_match_ts = player_last_match.get(player_id, now_ts)
                days_inactive = (now_ts - last_match_ts) / 86400.0

                if days_inactive > self.score_decay_delay_days:
                    days_to_decay = days_inactive - self.score_decay_delay_days
                    decay_factors[i] = (
                        1 - self.score_decay_rate
                    ) ** days_to_decay

                    self.logger.debug(
                        f"Player {player_id}: {days_inactive:.1f} days inactive, "
                        f"decay factor: {decay_factors[i]:.3f}"
                    )

            scores = scores * decay_factors
            result = pl.DataFrame(
                {
                    "id": active_players,
                    "player_rank": scores.tolist(),
                    "win_pr": win_pr.tolist(),
                    "loss_pr": loss_pr.tolist(),
                    "exposure": exposure.tolist(),
                }
            )

            if self.min_exposure is not None:
                pre_filter = result.height
                result = result.filter(pl.col("exposure") >= self.min_exposure)
                self.logger.info(
                    f"Filtered {pre_filter - result.height} players with exposure < {self.min_exposure}"
                )

            result = result.sort("player_rank", descending=True)

        self.logger.info(f"Score statistics:")
        self.logger.info(
            f"  Win PR - mean: {win_pr.mean():.6f}, std: {win_pr.std():.6f}"
        )
        self.logger.info(
            f"  Loss PR - mean: {loss_pr.mean():.6f}, std: {loss_pr.std():.6f}"
        )
        self.logger.info(
            f"  Log-odds - mean: {scores.mean():.6f}, std: {scores.std():.6f}"
        )
        self.logger.info(f"  Lambda used: {lambda_used:.6f}")

        positive_scores = np.sum(scores > 0)
        self.logger.info(
            f"Players with positive log-odds: {positive_scores} "
            f"({100 * positive_scores / len(scores):.1f}%)"
        )

        log_dataframe_stats(self.logger, result, "exposure_logodds_result")
        self.logger.info(
            f"Exposure log-odds completed: {result.height} players ranked"
        )

        return result

    def rank_teams(self, matches: pl.DataFrame) -> pl.DataFrame:
        """
        Rank teams using exposure log-odds algorithm.

        Parameters
        ----------
        matches : pl.DataFrame
            Matches DataFrame with winner_team_id, loser_team_id columns

        Returns
        -------
        pl.DataFrame
            Team rankings with columns: id, team_rank, win_pr, loss_pr, exposure
        """
        self.logger.info("Starting exposure log-odds team ranking")
        log_dataframe_stats(self.logger, matches, "team_matches")

        with log_timing(self.logger, "exposure log-odds team calculation"):
            self.logger.info("Getting active teams via standard ranking...")
            standard_result = self._tick_tock_loop(
                matches=matches,
                players=None,
                node_from="loser_team_id",
                node_to="winner_team_id",
            )

            active_teams = standard_result["id"].to_list()
            self.logger.info(f"Found {len(active_teams)} active teams")

            tournament_influence = self.tournament_influence_ or {}

            converted_matches = self._convert_team_matches(
                matches, tournament_influence
            )

            if self.use_surprisal:
                (
                    scores,
                    win_pr,
                    loss_pr,
                    rho,
                    lambda_used,
                ) = self._exposure_logodds_with_surprisal(
                    converted_matches, active_teams
                )
            else:
                (
                    scores,
                    win_pr,
                    loss_pr,
                    rho,
                    lambda_used,
                ) = self._exposure_logodds(converted_matches, active_teams)

            node_to_idx = {t: i for i, t in enumerate(active_teams)}
            exposure = np.zeros(len(active_teams))
            for match in converted_matches:
                w = match["weight"]
                for t in match["winners"]:
                    if t in node_to_idx:
                        exposure[node_to_idx[t]] += w
                for t in match["losers"]:
                    if t in node_to_idx:
                        exposure[node_to_idx[t]] += w

            result = pl.DataFrame(
                {
                    "id": active_teams,
                    "team_rank": scores.tolist(),
                    "win_pr": win_pr.tolist(),
                    "loss_pr": loss_pr.tolist(),
                    "exposure": exposure.tolist(),
                }
            ).sort("team_rank", descending=True)

        log_dataframe_stats(self.logger, result, "exposure_logodds_team_result")
        self.logger.info(
            f"Exposure log-odds team ranking completed: {result.height} teams ranked"
        )

        return result

    def _convert_matches_dataframe(
        self,
        matches: pl.DataFrame,
        players: pl.DataFrame,
        tournament_influence: dict[int, float],
        *,
        rosters: pl.DataFrame | None = None,
        include_share: bool = True,
        streaming: bool = False,
    ) -> pl.DataFrame:
        """
        Build a compact matches table with winners/losers roster lists and weights.

        Returns columns:
          [match_id, tournament_id, winners(list[user_id]), losers(list[user_id]), weight(float), ts(int)]
        If include_share=True, also returns:
          [wlen, llen, share]
        """

        now_ts = getattr(self, "now_ts", None) or int(self.now.timestamp())
        decay_rate = self.decay_rate
        beta = float(self.beta)

        needed = [
            "match_id",
            "tournament_id",
            "winner_team_id",
            "loser_team_id",
        ]
        if "last_game_finished_at" in matches.columns:
            needed.append("last_game_finished_at")
        if "match_created_at" in matches.columns:
            needed.append("match_created_at")
        if "is_bye" in matches.columns:
            needed.append("is_bye")

        m = matches.select([c for c in needed if c in matches.columns])

        filt = (
            pl.col("winner_team_id").is_not_null()
            & pl.col("loser_team_id").is_not_null()
            & (
                ~pl.col("is_bye").fill_null(False)
                if "is_bye" in m.columns
                else True
            )
        )
        m = m.filter(filt)

        ts_exprs = []
        if "last_game_finished_at" in m.columns:
            ts_exprs.append(pl.col("last_game_finished_at"))
        if "match_created_at" in m.columns:
            ts_exprs.append(pl.col("match_created_at"))
        ts_exprs.append(pl.lit(now_ts))
        ts_expr = pl.coalesce(ts_exprs).cast(pl.Int64)

        m = m.with_columns(ts_expr.alias("ts"))

        if tournament_influence:
            s_df = pl.DataFrame(
                {
                    "tournament_id": list(tournament_influence.keys()),
                    "S": list(tournament_influence.values()),
                }
            )
            m = m.join(s_df, on="tournament_id", how="left").with_columns(
                pl.col("S").fill_null(1.0)
            )
        else:
            m = m.with_columns(pl.lit(1.0).alias("S"))

        time_decay = (
            ((pl.lit(now_ts) - pl.col("ts").cast(pl.Float64)) / 86400.0)
            .mul(-decay_rate)
            .exp()
        )
        if beta == 0.0:
            weight_expr = time_decay
        else:
            weight_expr = time_decay * (pl.col("S") ** beta)

        m = m.with_columns(weight_expr.alias("weight"))

        if rosters is None:
            used_teams = (
                pl.concat(
                    [
                        m.select(
                            pl.col("tournament_id"),
                            pl.col("winner_team_id").alias("team_id"),
                        ),
                        m.select(
                            pl.col("tournament_id"),
                            pl.col("loser_team_id").alias("team_id"),
                        ),
                    ]
                )
                .unique()
                .with_columns(
                    [
                        pl.col("team_id").cast(pl.Int64),
                        pl.col("tournament_id").cast(pl.Int64),
                    ]
                )
            )

            rosters = (
                players.select(["tournament_id", "team_id", "user_id"])
                .with_columns(
                    [
                        pl.col("tournament_id").cast(pl.Int64),
                        pl.col("team_id").cast(pl.Int64),
                    ]
                )
                .join(used_teams, on=["tournament_id", "team_id"], how="semi")
                .group_by(["tournament_id", "team_id"])
                .agg(pl.col("user_id").alias("roster"))
            )
        else:
            rosters = rosters.with_columns(
                [
                    pl.col("tournament_id").cast(pl.Int64),
                    pl.col("team_id").cast(pl.Int64),
                ]
            )

        m = m.with_columns(
            [
                pl.col("tournament_id").cast(pl.Int64),
                pl.col("winner_team_id").cast(pl.Int64),
                pl.col("loser_team_id").cast(pl.Int64),
            ]
        )

        m = (
            m.join(
                rosters,
                left_on=["tournament_id", "winner_team_id"],
                right_on=["tournament_id", "team_id"],
                how="inner",
            )
            .rename({"roster": "winners"})
            .join(
                rosters,
                left_on=["tournament_id", "loser_team_id"],
                right_on=["tournament_id", "team_id"],
                how="inner",
            )
            .rename({"roster": "losers"})
            .select(
                [
                    "match_id",
                    "tournament_id",
                    "winners",
                    "losers",
                    "weight",
                    "ts",
                ]
            )
        )

        if include_share:
            m = m.with_columns(
                [
                    pl.col("winners").list.len().alias("wlen"),
                    pl.col("losers").list.len().alias("llen"),
                ]
            ).with_columns(
                (pl.col("weight") / (pl.col("wlen") * pl.col("llen"))).alias(
                    "share"
                )
            )

        if streaming and isinstance(m, pl.LazyFrame):
            return m.collect(streaming=True)

        return m

    def _convert_matches_format(
        self,
        matches: pl.DataFrame,
        players: pl.DataFrame,
        tournament_influence: dict[int, float],
    ) -> list[dict]:
        """
        Convert polars DataFrame matches to list of dicts with winners/losers lists.
        """
        converted = []

        for row in matches.iter_rows(named=True):
            if row.get("is_bye", False):
                continue

            winner_team = row.get("winner_team_id")
            loser_team = row.get("loser_team_id")

            if not winner_team or not loser_team:
                continue

            tid = row["tournament_id"]

            winner_players = players.filter(
                (pl.col("tournament_id") == tid)
                & (pl.col("team_id") == winner_team)
            )["user_id"].to_list()

            loser_players = players.filter(
                (pl.col("tournament_id") == tid)
                & (pl.col("team_id") == loser_team)
            )["user_id"].to_list()

            if not winner_players or not loser_players:
                continue

            t_influence = tournament_influence.get(tid, 1.0)
            if "last_game_finished_at" in row and row["last_game_finished_at"]:
                ts = row["last_game_finished_at"]
            elif "match_created_at" in row and row["match_created_at"]:
                ts = row["match_created_at"]
            else:
                ts = self.now_ts

            days_ago = (self.now_ts - ts) / 86400.0
            time_decay = math.exp(-self.decay_rate * days_ago)

            weight = time_decay * (t_influence**self.beta)

            converted.append(
                {
                    "winners": winner_players,
                    "losers": loser_players,
                    "weight": weight,
                    "tournament_id": tid,
                    "match_id": row.get("match_id"),
                    "timestamp": ts,  # Include timestamp for decay calculation
                }
            )

        self.logger.info(f"Converted {len(converted)} valid matches")
        return converted

    def _convert_team_matches(
        self, matches: pl.DataFrame, tournament_influence: Dict[int, float]
    ) -> list[dict]:
        """
        Convert team matches to required format.
        """
        converted = []

        for row in matches.iter_rows(named=True):
            if row.get("is_bye", False):
                continue

            winner_team = row.get("winner_team_id")
            loser_team = row.get("loser_team_id")

            if not winner_team or not loser_team:
                continue

            tid = row["tournament_id"]
            t_influence = tournament_influence.get(tid, 1.0)

            # Time decay
            if "last_game_finished_at" in row and row["last_game_finished_at"]:
                ts = row["last_game_finished_at"]
            elif "match_created_at" in row and row["match_created_at"]:
                ts = row["match_created_at"]
            else:
                ts = self.now_ts

            days_ago = (self.now_ts - ts) / 86400.0
            time_decay = math.exp(-self.decay_rate * days_ago)

            weight = time_decay * (t_influence**self.beta)

            converted.append(
                {
                    "winners": [winner_team],
                    "losers": [loser_team],
                    "weight": weight,
                    "tournament_id": tid,
                    "match_id": row.get("match_id"),
                    "timestamp": ts,  # Include timestamp for decay calculation
                }
            )

        return converted

    def _build_exposure_triplets(
        self, m: pl.DataFrame, node_to_idx: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Produce COO triplets (row=winner_idx, col=loser_idx, data=share).
        Uses list-explode to form the cartesian product winner×loser per match
        but stays inside Polars/NumPy for speed.
        """
        pairs = (
            m.explode("winners")
            .explode("losers")
            .select(
                [
                    pl.col("winners").alias("winner_id"),
                    pl.col("losers").alias("loser_id"),
                    "share",
                ]
            )
        )
        idx_map = pl.DataFrame(
            {"id": list(node_to_idx.keys()), "idx": list(node_to_idx.values())}
        )
        pairs = (
            pairs.join(idx_map, left_on="winner_id", right_on="id", how="inner")
            .rename({"idx": "w_idx"})
            .join(idx_map, left_on="loser_id", right_on="id", how="inner")
            .rename({"idx": "l_idx"})
            .select(["w_idx", "l_idx", "share"])
        )
        pairs = pairs.group_by(["w_idx", "l_idx"]).agg(
            pl.col("share").sum().alias("w_sum")
        )
        return (
            pairs["w_idx"].to_numpy().astype(np.int64),
            pairs["l_idx"].to_numpy().astype(np.int64),
            pairs["w_sum"].to_numpy().astype(np.float64),
        )

    def _pagerank_col_stochastic_sparse(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        data: np.ndarray,
        rho: np.ndarray,
        n: int,
        alpha: float = 0.85,
        tol: float = 1e-12,
        max_iter: int = 200,
    ) -> np.ndarray:
        """
        Column-stochastic PageRank using sparse assembly: A[dst, src].
        """
        if sp is not None:
            adjacency_matrix = sp.csc_matrix((data, (rows, cols)), shape=(n, n))
            column_sums = np.asarray(adjacency_matrix.sum(axis=0)).ravel()
            inverse_sums = np.zeros_like(column_sums)
            nonzero_mask = column_sums > 0
            inverse_sums[nonzero_mask] = 1.0 / column_sums[nonzero_mask]
            transition_matrix = adjacency_matrix @ sp.diags(inverse_sums)
            rank_vector = rho / rho.sum()
            for _ in range(max_iter):
                rank_vector_new = (
                    alpha * (transition_matrix @ rank_vector)
                    + (1 - alpha) * rho
                )
                dangling_mass = alpha * rank_vector[~nonzero_mask].sum()
                if dangling_mass:
                    rank_vector_new += dangling_mass * rho
                rank_vector_new /= rank_vector_new.sum()
                if np.linalg.norm(rank_vector_new - rank_vector, 1) < tol:
                    break
                rank_vector = rank_vector_new
            return rank_vector
        else:
            column_sums = np.zeros(n)
            np.add.at(column_sums, cols, data)
            inverse_sums = np.zeros_like(column_sums)
            nonzero_mask = column_sums > 0
            inverse_sums[nonzero_mask] = 1.0 / column_sums[nonzero_mask]
            weights = data * inverse_sums[cols]
            rank_vector = rho / rho.sum()
            for _ in range(max_iter):
                rank_vector_new = np.zeros(n)
                np.add.at(
                    rank_vector_new, rows, alpha * weights * rank_vector[cols]
                )
                dangling_mass = alpha * rank_vector[~nonzero_mask].sum()
                rank_vector_new += (1 - alpha + dangling_mass) * rho
                rank_vector_new /= rank_vector_new.sum()
                if np.linalg.norm(rank_vector_new - rank_vector, 1) < tol:
                    break
                rank_vector = rank_vector_new
            return rank_vector

    def _build_edges_loser_to_winner(
        self,
        matches: List[Dict],
        node_to_idx: dict,
        surprisal_ratings: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Build adjacency matrix A where A[dst, src] = weight.
        Edges go from LOSER -> WINNER.
        Team weights are divided evenly across all player pairs.
        """
        num_nodes = len(node_to_idx)
        adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=float)

        for match in matches:
            winners = match["winners"]
            losers = match["losers"]

            if not winners or not losers:
                continue

            time_weight = match["weight"]
            upset_weight = 1.0

            if surprisal_ratings is not None and self.use_surprisal:
                # Compute surprisal weight based on average team ratings
                winner_indices = [
                    node_to_idx.get(w) for w in winners if w in node_to_idx
                ]
                loser_indices = [
                    node_to_idx.get(l) for l in losers if l in node_to_idx
                ]

                if winner_indices and loser_indices:
                    avg_winner = np.mean(
                        [surprisal_ratings[i] for i in winner_indices]
                    )
                    avg_loser = np.mean(
                        [surprisal_ratings[i] for i in loser_indices]
                    )
                    rating_diff = avg_winner - avg_loser
                    p_win = 1.0 / (
                        1.0 + math.exp(-rating_diff / self.surprisal_T)
                    )
                    upset_weight = -math.log(max(p_win, 1e-10))

            match_weight = time_weight * upset_weight

            # CRITICAL: Divide by number of pairs to preserve total match weight
            share = match_weight / (len(winners) * len(losers))

            for loser_player in losers:
                loser_idx = node_to_idx.get(loser_player)
                if loser_idx is None:
                    continue

                for winner_player in winners:
                    winner_idx = node_to_idx.get(winner_player)
                    if winner_idx is None:
                        continue

                    # A[dst, src] = weight; loser -> winner means src=loser, dst=winner
                    adjacency_matrix[winner_idx, loser_idx] += share

        return adjacency_matrix

    def _pagerank_col_stochastic(
        self,
        A: np.ndarray,
        rho: np.ndarray,
        alpha: float = 0.85,
        tol: float = 1e-12,
        max_iter: int = 200,
    ) -> np.ndarray:
        """
        PageRank with column-stochastic normalization.
        A[dst, src] adjacency matrix.
        """
        n = A.shape[0]

        column_sums = A.sum(axis=0)
        transition_matrix = np.zeros_like(A)

        for j in range(n):
            if column_sums[j] > 0:
                transition_matrix[:, j] = A[:, j] / column_sums[j]

        # Normalize teleport
        rho = np.asarray(rho, dtype=float)
        rho = rho / rho.sum()
        rank_vector = rho.copy()

        for iteration in range(max_iter):
            rank_vector_new = (
                alpha * (transition_matrix @ rank_vector) + (1 - alpha) * rho
            )

            dangling_mass = alpha * rank_vector[column_sums == 0].sum()
            if dangling_mass > 0:
                rank_vector_new += dangling_mass * rho

            rank_vector_new /= rank_vector_new.sum()

            if np.linalg.norm(rank_vector_new - rank_vector, 1) < tol:
                self.logger.debug(
                    f"PageRank converged at iteration {iteration}"
                )
                break

            rank_vector = rank_vector_new

        return rank_vector

    def _exposure_logodds_optimized(
        self,
        matches_df: pl.DataFrame,
        active_nodes: list,
        alpha: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Optimized exposure log-odds using vectorized operations.
        """
        alpha = alpha or self.damping_factor
        node_to_idx = {p: i for i, p in enumerate(active_nodes)}
        n = len(node_to_idx)

        winners_exp = (
            matches_df.select(["winners", "share"])
            .explode("winners")
            .rename({"winners": "id"})
        )

        losers_exp = (
            matches_df.select(["losers", "share"])
            .explode("losers")
            .rename({"losers": "id"})
        )

        exp_df = (
            pl.concat([winners_exp, losers_exp])
            .group_by("id")
            .agg(pl.col("share").sum().alias("e"))
            .join(pl.DataFrame({"id": active_nodes}), on="id", how="right")
            .fill_null(0.0)
            .select("e")
        )
        exposure_array = exp_df["e"].to_numpy()
        rho = exposure_array + self.epsilon
        rho = rho / rho.sum()

        rows, cols, data = self._build_exposure_triplets(
            matches_df, node_to_idx
        )
        win_pagerank = self._pagerank_col_stochastic_sparse(
            rows, cols, data, rho, n, alpha=alpha
        )
        loss_pagerank = self._pagerank_col_stochastic_sparse(
            cols, rows, data, rho, n, alpha=alpha
        )

        if self.lambda_smooth is None:
            target = 0.025 * np.median(win_pagerank)
            med_rho = np.median(rho)
            lambda_smooth = 0.0 if med_rho == 0 else max(target / med_rho, 0.0)
        else:
            lambda_smooth = self.lambda_smooth

        score = np.log(
            (win_pagerank + lambda_smooth * rho)
            / (loss_pagerank + lambda_smooth * rho)
        )
        return score, win_pagerank, loss_pagerank, rho, lambda_smooth

    def _exposure_logodds(
        self,
        matches: list[dict],
        active_nodes: list,
        alpha: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Core exposure log-odds computation following plan.md.

        Returns: (scores, win_pr, loss_pr, rho, lambda_used)
        """
        if alpha is None:
            alpha = self.damping_factor

        node_to_idx = {p: i for i, p in enumerate(active_nodes)}
        n = len(node_to_idx)

        exposure_vector = np.zeros(n)
        for match in matches:
            w = match["weight"]
            for p in match["winners"]:
                if p in node_to_idx:
                    exposure_vector[node_to_idx[p]] += w
            for p in match["losers"]:
                if p in node_to_idx:
                    exposure_vector[node_to_idx[p]] += w

        # Create teleport vector
        rho = exposure_vector + self.epsilon
        rho = rho / rho.sum()

        self.logger.info(
            f"Exposure stats: min={exposure_vector.min():.2f}, max={exposure_vector.max():.2f}, "
            f"mean={exposure_vector.mean():.2f}, median={np.median(exposure_vector):.2f}"
        )

        A_win = self._build_edges_loser_to_winner(matches, node_to_idx)

        A_loss = A_win.T.copy()

        assert np.allclose(
            A_loss, A_win.T, atol=1e-12
        ), "Loss graph not exact transpose!"
        assert np.isclose(
            A_win.sum(), A_loss.sum(), atol=1e-8
        ), "Total weights don't match!"

        self.logger.info(
            f"Built adjacency matrices: total weight = {A_win.sum():.2f}"
        )

        win_pagerank = self._pagerank_col_stochastic(A_win, rho, alpha=alpha)
        loss_pagerank = self._pagerank_col_stochastic(A_loss, rho, alpha=alpha)

        if self.lambda_smooth is None:
            target = 0.025 * np.median(win_pagerank)
            med_rho = np.median(rho)
            lambda_smooth = max(target / med_rho, 0.0)
            self.logger.info(f"Auto-tuned lambda: {lambda_smooth:.6f}")
        else:
            lambda_smooth = self.lambda_smooth

        score = np.log(
            (win_pagerank + lambda_smooth * rho)
            / (loss_pagerank + lambda_smooth * rho)
        )

        assert np.isclose(
            win_pagerank.sum(), 1.0, atol=1e-10
        ), f"Win PR doesn't sum to 1: {win_pagerank.sum()}"
        assert np.isclose(
            loss_pagerank.sum(), 1.0, atol=1e-10
        ), f"Loss PR doesn't sum to 1: {loss_pagerank.sum()}"
        assert np.isclose(
            rho.sum(), 1.0, atol=1e-10
        ), f"Teleport doesn't sum to 1: {rho.sum()}"

        return score, win_pagerank, loss_pagerank, rho, lambda_smooth

    def _exposure_logodds_with_surprisal(
        self, matches: list[dict], active_nodes: list
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Exposure log-odds with iterative surprisal refinement.
        """
        node_to_idx = {p: i for i, p in enumerate(active_nodes)}
        n = len(node_to_idx)

        # Initialize with flat ratings
        provisional_ratings = np.zeros(n)

        for iteration in range(self.surprisal_iters):
            self.logger.info(
                f"Surprisal iteration {iteration + 1}/{self.surprisal_iters}"
            )

            if iteration == 0:
                exposure_vector = np.zeros(n)
                for match in matches:
                    w = match["weight"]
                    for p in match["winners"]:
                        if p in node_to_idx:
                            exposure_vector[node_to_idx[p]] += w
                    for p in match["losers"]:
                        if p in node_to_idx:
                            exposure_vector[node_to_idx[p]] += w

                rho = exposure_vector + self.epsilon
                rho = rho / rho.sum()

            A_win = self._build_edges_loser_to_winner(
                matches,
                node_to_idx,
                surprisal_ratings=provisional_ratings
                if iteration > 0
                else None,
            )
            A_loss = A_win.T.copy()

            win_pagerank = self._pagerank_col_stochastic(
                A_win, rho, alpha=self.damping_factor
            )
            loss_pagerank = self._pagerank_col_stochastic(
                A_loss, rho, alpha=self.damping_factor
            )

            if iteration == 0:
                if self.lambda_smooth is None:
                    target = 0.025 * np.median(win_pagerank)
                    med_rho = np.median(rho)
                    lambda_smooth = max(target / med_rho, 0.0)
                    self.logger.info(f"Auto-tuned lambda: {lambda_smooth:.6f}")
                else:
                    lambda_smooth = self.lambda_smooth

            score = np.log(
                (win_pagerank + lambda_smooth * rho)
                / (loss_pagerank + lambda_smooth * rho)
            )

            provisional_ratings = (score - score.mean()) / (score.std() + 1e-8)

        return score, win_pagerank, loss_pagerank, rho, lambda_smooth

    def post_process_rankings(
        self,
        rankings: pl.DataFrame,
        players_df: pl.DataFrame,
        min_tournaments: int = 3,
        inactivity_drop_days: float | None = None,
        rank_cutoffs: list[float] | None = None,
        rank_labels: list[str] | None = None,
        score_multiplier: float = 25.0,
        score_offset: float = 0.0,
        tournaments_df: pl.DataFrame | None = None,
        use_score_grade_system: bool = True,
    ) -> pl.DataFrame:
        """
        Post-process rankings with tournament filtering and grade assignment.

        NOTE: This method now delegates to the standalone post-processing module
        for better separation of concerns. The signature is maintained for
        backward compatibility.

        Parameters
        ----------
        rankings : pl.DataFrame
            Raw rankings from rank_players
        players_df : pl.DataFrame
            Players DataFrame with tournament_id, user_id, username
        min_tournaments : int, default=3
            Minimum tournaments required (filters to > min_tournaments-1)
        inactivity_drop_days : float | None, default=None
            If specified, drops players who haven't played in this many days
            If None, allows normal decay via add_activity_decay
        rank_cutoffs : Optional[List[float]], default=None
            Score cutoffs for rank labels (used only if use_score_grade_system=False)
        rank_labels : Optional[List[str]], default=None
            Labels for rank grades (used only if use_score_grade_system=False)
        score_multiplier : float, default=25.0
            Multiplier for final score display
        score_offset : float, default=0.0
            Offset to add to scores before applying multiplier
        tournaments_df : Optional[pl.DataFrame], default=None
            Tournaments DataFrame with tournament_id and start_time columns
            If provided, adds last_active date for each player
        use_score_grade_system : bool, default=True
            If True, uses ScoreGradeSystem for grades. If False, uses legacy cut method

        Returns
        -------
        pl.DataFrame
            Processed rankings with grades and filtered players
        """
        # Import the standalone post-processing function
        from rankings.postprocess import (
            post_process_rankings as standalone_post_process,
        )

        # The old engine expects "player_rank" column but needs to map to "score"
        # Also need to ensure the id column is named correctly
        rankings_formatted = rankings
        if "player_rank" in rankings.columns:
            rankings_formatted = rankings_formatted.rename(
                {"player_rank": "score"}
            )

        # Call the standalone post-processing function
        return standalone_post_process(
            rankings=rankings_formatted,
            players_df=players_df,
            min_tournaments=min_tournaments,
            inactivity_drop_days=inactivity_drop_days,
            rank_cutoffs=rank_cutoffs,
            rank_labels=rank_labels,
            score_multiplier=score_multiplier,
            score_offset=score_offset,
            tournaments_df=tournaments_df,
            id_column="id",
            score_column="score",
            use_score_grade_system=use_score_grade_system,
        )
