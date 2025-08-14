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
from typing import Dict, List, Optional

import numpy as np
import polars as pl

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
        lambda_smooth: Optional[float] = None,
        use_surprisal: bool = False,
        surprisal_T: float = 1.0,
        surprisal_iters: int = 2,
        epsilon: float = 1e-9,
        min_exposure: Optional[float] = None,
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
        self.exposure_vector_: Optional[np.ndarray] = None
        self.exposure_teleport_: Optional[np.ndarray] = None
        self.win_pagerank_: Optional[np.ndarray] = None
        self.loss_pagerank_: Optional[np.ndarray] = None
        self.logodds_scores_: Optional[np.ndarray] = None
        self.lambda_used_: Optional[float] = None

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
            converted_matches = self._convert_matches_format(
                matches, players, tournament_influence
            )

            # Run exposure log-odds with optional surprisal iterations
            if self.use_surprisal:
                (
                    scores,
                    win_pr,
                    loss_pr,
                    rho,
                    lambda_used,
                ) = self._exposure_logodds_with_surprisal(
                    converted_matches, active_players
                )
            else:
                (
                    scores,
                    win_pr,
                    loss_pr,
                    rho,
                    lambda_used,
                ) = self._exposure_logodds(converted_matches, active_players)

            # Store results
            self.win_pagerank_ = win_pr
            self.loss_pagerank_ = loss_pr
            self.exposure_teleport_ = rho
            self.logodds_scores_ = scores
            self.lambda_used_ = lambda_used

            # Compute raw exposure for reporting
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

            # Calculate last match timestamp for each player
            player_last_match = {}
            for match in converted_matches:
                # Get match timestamp (already calculated in conversion)
                match_ts = int(self.now.timestamp())  # Default to now
                if "timestamp" in match:
                    match_ts = match["timestamp"]

                # Update last match for all players in the match
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

            # Apply time-based score decay
            now_ts = int(self.now.timestamp())
            decay_factors = np.ones(len(active_players))

            for i, player_id in enumerate(active_players):
                last_match_ts = player_last_match.get(player_id, now_ts)
                days_inactive = (now_ts - last_match_ts) / 86400.0

                # Apply decay only after the delay period
                if days_inactive > self.score_decay_delay_days:
                    days_to_decay = days_inactive - self.score_decay_delay_days
                    # Exponential decay: score * (1 - rate)^days
                    decay_factors[i] = (
                        1 - self.score_decay_rate
                    ) ** days_to_decay

                    self.logger.debug(
                        f"Player {player_id}: {days_inactive:.1f} days inactive, "
                        f"decay factor: {decay_factors[i]:.3f}"
                    )

            # Apply decay to scores
            scores = scores * decay_factors

            # Create result DataFrame
            result = pl.DataFrame(
                {
                    "id": active_players,
                    "player_rank": scores.tolist(),
                    "win_pr": win_pr.tolist(),
                    "loss_pr": loss_pr.tolist(),
                    "exposure": exposure.tolist(),
                }
            )

            # Apply minimum exposure filter if specified
            if self.min_exposure is not None:
                pre_filter = result.height
                result = result.filter(pl.col("exposure") >= self.min_exposure)
                self.logger.info(
                    f"Filtered {pre_filter - result.height} players with exposure < {self.min_exposure}"
                )

            result = result.sort("player_rank", descending=True)

        # Log statistics
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
            # Get active teams and tournament influences
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

            # Convert matches for team mode
            converted_matches = self._convert_team_matches(
                matches, tournament_influence
            )

            # Run exposure log-odds
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

            # Compute exposure
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

            # Create result
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

    def _convert_matches_format(
        self,
        matches: pl.DataFrame,
        players: pl.DataFrame,
        tournament_influence: Dict[int, float],
    ) -> List[Dict]:
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

            # Get players from teams
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

            # Compute match weight
            t_influence = tournament_influence.get(tid, 1.0)

            # Time decay
            if "last_game_finished_at" in row and row["last_game_finished_at"]:
                ts = row["last_game_finished_at"]
            elif "match_created_at" in row and row["match_created_at"]:
                ts = row["match_created_at"]
            else:
                ts = int(self.now.timestamp())

            days_ago = (int(self.now.timestamp()) - ts) / 86400.0
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
    ) -> List[Dict]:
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
                ts = int(self.now.timestamp())

            days_ago = (int(self.now.timestamp()) - ts) / 86400.0
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

    def _build_edges_loser_to_winner(
        self,
        matches: List[Dict],
        node_to_idx: Dict,
        surprisal_ratings: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Build adjacency matrix A where A[dst, src] = weight.
        Edges go from LOSER -> WINNER.
        Team weights are divided evenly across all player pairs.
        """
        n = len(node_to_idx)
        A = np.zeros((n, n), dtype=float)

        for match in matches:
            winners = match["winners"]
            losers = match["losers"]

            if not winners or not losers:
                continue

            w_time = match["weight"]
            u = 1.0

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
                    u = -math.log(max(p_win, 1e-10))

            w_match = w_time * u

            # CRITICAL: Divide by number of pairs to preserve total match weight
            share = w_match / (len(winners) * len(losers))

            for lp in losers:
                ls = node_to_idx.get(lp)
                if ls is None:
                    continue

                for wp in winners:
                    ws = node_to_idx.get(wp)
                    if ws is None:
                        continue

                    # A[dst, src] = weight; loser -> winner means src=loser, dst=winner
                    A[ws, ls] += share

        return A

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

        # Column sums (outgoing mass per source)
        col = A.sum(axis=0)
        P = np.zeros_like(A)

        for j in range(n):
            if col[j] > 0:
                P[:, j] = A[:, j] / col[j]  # Column-stochastic
            # else leave column zeros (dangling)

        # Normalize teleport
        rho = np.asarray(rho, dtype=float)
        rho = rho / rho.sum()
        s = rho.copy()

        for iteration in range(max_iter):
            s_new = alpha * (P @ s) + (1 - alpha) * rho

            # Redistribute dangling mass
            dangling_mass = alpha * s[col == 0].sum()
            if dangling_mass > 0:
                s_new += dangling_mass * rho

            # Normalize for safety
            s_new /= s_new.sum()

            if np.linalg.norm(s_new - s, 1) < tol:
                self.logger.debug(
                    f"PageRank converged at iteration {iteration}"
                )
                break

            s = s_new

        return s

    def _exposure_logodds(
        self, matches: List[Dict], active_nodes: List, alpha: float = None
    ) -> tuple:
        """
        Core exposure log-odds computation following plan.md.

        Returns: (scores, win_pr, loss_pr, rho, lambda_used)
        """
        if alpha is None:
            alpha = self.damping_factor

        node_to_idx = {p: i for i, p in enumerate(active_nodes)}
        n = len(node_to_idx)

        # 1) Compute exposure vector
        e = np.zeros(n)
        for match in matches:
            w = match["weight"]
            for p in match["winners"]:
                if p in node_to_idx:
                    e[node_to_idx[p]] += w
            for p in match["losers"]:
                if p in node_to_idx:
                    e[node_to_idx[p]] += w

        # Create teleport vector
        rho = e + self.epsilon
        rho = rho / rho.sum()

        self.logger.info(
            f"Exposure stats: min={e.min():.2f}, max={e.max():.2f}, "
            f"mean={e.mean():.2f}, median={np.median(e):.2f}"
        )

        # 2) Build loser->winner adjacency A_win[dst, src]
        A_win = self._build_edges_loser_to_winner(matches, node_to_idx)

        # 3) Loss graph is the EXACT mirror: A_loss = A_win.T
        A_loss = A_win.T.copy()

        # Sanity checks
        assert np.allclose(
            A_loss, A_win.T, atol=1e-12
        ), "Loss graph not exact transpose!"
        assert np.isclose(
            A_win.sum(), A_loss.sum(), atol=1e-8
        ), "Total weights don't match!"

        self.logger.info(
            f"Built adjacency matrices: total weight = {A_win.sum():.2f}"
        )

        # 4) PageRanks with the SAME teleport rho
        s = self._pagerank_col_stochastic(A_win, rho, alpha=alpha)
        l = self._pagerank_col_stochastic(A_loss, rho, alpha=alpha)

        # Auto lambda if not given
        if self.lambda_smooth is None:
            target = 0.025 * np.median(s)  # 2.5% of typical PR mass
            med_rho = np.median(rho)
            lambda_smooth = max(target / med_rho, 0.0)
            self.logger.info(f"Auto-tuned lambda: {lambda_smooth:.6f}")
        else:
            lambda_smooth = self.lambda_smooth

        # 5) Log-odds (smoothed)
        score = np.log((s + lambda_smooth * rho) / (l + lambda_smooth * rho))

        # Validations
        assert np.isclose(
            s.sum(), 1.0, atol=1e-10
        ), f"Win PR doesn't sum to 1: {s.sum()}"
        assert np.isclose(
            l.sum(), 1.0, atol=1e-10
        ), f"Loss PR doesn't sum to 1: {l.sum()}"
        assert np.isclose(
            rho.sum(), 1.0, atol=1e-10
        ), f"Teleport doesn't sum to 1: {rho.sum()}"

        return score, s, l, rho, lambda_smooth

    def _exposure_logodds_with_surprisal(
        self, matches: List[Dict], active_nodes: List
    ) -> tuple:
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

            # Compute exposure (same for all iterations)
            if iteration == 0:
                e = np.zeros(n)
                for match in matches:
                    w = match["weight"]
                    for p in match["winners"]:
                        if p in node_to_idx:
                            e[node_to_idx[p]] += w
                    for p in match["losers"]:
                        if p in node_to_idx:
                            e[node_to_idx[p]] += w

                rho = e + self.epsilon
                rho = rho / rho.sum()

            # Build edges with current surprisal weights
            A_win = self._build_edges_loser_to_winner(
                matches,
                node_to_idx,
                surprisal_ratings=provisional_ratings
                if iteration > 0
                else None,
            )
            A_loss = A_win.T.copy()

            # PageRanks
            s = self._pagerank_col_stochastic(
                A_win, rho, alpha=self.damping_factor
            )
            l = self._pagerank_col_stochastic(
                A_loss, rho, alpha=self.damping_factor
            )

            # Auto-tune lambda on first iteration
            if iteration == 0:
                if self.lambda_smooth is None:
                    target = 0.025 * np.median(s)
                    med_rho = np.median(rho)
                    lambda_smooth = max(target / med_rho, 0.0)
                    self.logger.info(f"Auto-tuned lambda: {lambda_smooth:.6f}")
                else:
                    lambda_smooth = self.lambda_smooth

            # Compute scores
            score = np.log(
                (s + lambda_smooth * rho) / (l + lambda_smooth * rho)
            )

            # Update provisional ratings (z-scored)
            provisional_ratings = (score - score.mean()) / (score.std() + 1e-8)

        return score, s, l, rho, lambda_smooth

    def post_process_rankings(
        self,
        rankings: pl.DataFrame,
        players_df: pl.DataFrame,
        min_tournaments: int = 3,
        rank_cutoffs: Optional[List[float]] = None,
        rank_labels: Optional[List[str]] = None,
        score_multiplier: float = 25.0,
        score_offset: float = 0.0,
    ) -> pl.DataFrame:
        """
        Post-process rankings with tournament filtering and grade assignment.

        Parameters
        ----------
        rankings : pl.DataFrame
            Raw rankings from rank_players
        players_df : pl.DataFrame
            Players DataFrame with tournament_id, user_id, username
        min_tournaments : int, default=3
            Minimum tournaments required (filters to > min_tournaments-1)
        rank_cutoffs : Optional[List[float]], default=None
            Score cutoffs for rank labels
        rank_labels : Optional[List[str]], default=None
            Labels for rank grades
        score_multiplier : float, default=25.0
            Multiplier for final score display
        score_offset : float, default=0.0
            Offset to add to scores before applying multiplier

        Returns
        -------
        pl.DataFrame
            Processed rankings with grades and filtered players
        """
        if rank_cutoffs is None:
            rank_cutoffs = [-3, -1, 0, 1, 2, 3, 4, 5]
        if rank_labels is None:
            rank_labels = ["A-", "A", "A+", "S-", "S", "S+", "X", "X+", "X★"]

        # Add raw rank before any filtering
        final_rankings = (
            rankings.with_columns(pl.arange(1, pl.len() + 1).alias("raw_rank"))
            .join(
                players_df.group_by("user_id")
                .agg(
                    [
                        pl.col("username").last(),
                        pl.col("tournament_id")
                        .n_unique()
                        .alias("tournament_count"),
                    ]
                )
                .select(["user_id", "username", "tournament_count"]),
                left_on="id",
                right_on="user_id",
                how="left",
            )
            .sort("raw_rank")
            .select(
                "raw_rank",
                "username",
                pl.col("id").alias("player_id"),
                pl.col("player_rank").alias("score"),
                pl.col("exposure").alias("exposure"),
                pl.col("win_pr").alias("win_pr"),
                pl.col("loss_pr").alias("loss_pr"),
                (pl.col("win_pr") / pl.col("loss_pr")).alias("win_loss_ratio"),
                (pl.col("win_pr") - pl.col("loss_pr")).alias("win_loss_diff"),
                pl.col("tournament_count"),
            )
            .filter(pl.col("tournament_count") >= min_tournaments)
            .select(
                pl.arange(1, pl.len() + 1).alias("rank"),
                "raw_rank",
                "username",
                "player_id",
                "score",
                "exposure",
                "win_pr",
                "loss_pr",
                "win_loss_ratio",
                "win_loss_diff",
                "tournament_count",
            )
        )

        # Add grades
        final_rankings_with_grades = final_rankings.with_columns(
            pl.col("score")
            .cut(
                breaks=rank_cutoffs,
                labels=rank_labels,
            )
            .alias("rank_label")
        ).select(
            "rank",
            "rank_label",
            "username",
            "player_id",
            ((pl.col("score") + score_offset) * score_multiplier).alias(
                "display_score"
            ),
            "score",
            "win_loss_ratio",
            "tournament_count",
        )

        return final_rankings_with_grades
