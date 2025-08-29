from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from rankings.core import (
    Clock,
    ExposureLogOddsConfig,
    ExposureLogOddsResult,
    PageRankConfig,
    apply_inactivity_decay,
    build_exposure_triplets,
    convert_matches_dataframe,
    pagerank_sparse,
)
from rankings.core.logging import get_logger, log_timing

if TYPE_CHECKING:
    from rankings.analysis.loo_analyzer import LOOAnalyzer


class ExposureLogOddsEngine:
    """Exposure Log-Odds rating engine using modular components.

    Updated to mirror the legacy implementation semantics:
      - Teleport ρ is exposure-based (sum of match pair 'share' per player)
      - Two PageRanks with the SAME ρ (col-stochastic)
      - Lambda auto-tuned to ~2.5% of median PR per node (dividing by median ρ)
      - Reporting 'exposure' uses sum of 'weight' (legacy reporting semantics)
      - Optional time-decay after inactivity delay
      - Output schema: id + player_rank (legacy-compatible)
    """

    def __init__(
        self,
        config: ExposureLogOddsConfig | None = None,
        *,
        now_ts: float | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.config = config or ExposureLogOddsConfig()
        self.logger = get_logger(self.__class__.__name__)

        if clock is not None:
            self.clock = clock
        else:
            self.clock = Clock(
                now_timestamp=(now_ts if now_ts is not None else time.time())
            )

        self.last_result: ExposureLogOddsResult | None = None
        self._last_rho: np.ndarray | None = None
        self._loo_analyzer: LOOAnalyzer | None = None
        self._loo_matches_df: pl.DataFrame | None = None
        self._loo_players_df: pl.DataFrame | None = None
        self._converted_matches_df: pl.DataFrame | None = None

    def rank_players(
        self,
        matches: pl.DataFrame,
        players: pl.DataFrame,
        tournament_influence: dict[int, float] | None = None,
        *,
        appearances: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Rank players using the Exposure Log-Odds algorithm.

        Args:
            matches: DataFrame containing match data.
            players: DataFrame containing player data.
            tournament_influence: Optional mapping of tournament IDs to influence weights.

        Returns:
            DataFrame containing player rankings.
        """
        start_time = time.time()

        # 1) Get active players and tournament influences via tick‑tock
        if self.config.use_tick_tock_active:
            self.logger.info(
                "Running tick-tock to obtain active players & tournament influences..."
            )
            tick = self._run_tick_tock_for_active_players(matches, players)
            active_players = tick["active_players"]
            tournament_influence = tick["tournament_influence"]
            if not active_players:
                self.logger.warning(
                    "No active players from tick-tock; returning empty result."
                )
                return pl.DataFrame()
        else:
            self.logger.info("Skipping tick-tock (use_tick_tock_active=False)")
            active_players = players["player_id"].to_list()
            tournament_influence = tournament_influence or {}

        # 2) Convert matches into compact per-match lists with roster expansion and weights
        self.logger.info("Converting matches...")
        mdf = convert_matches_dataframe(
            matches,
            players,
            tournament_influence or {},
            self.clock.now,
            self.config.decay.decay_rate,
            self.config.engine.beta,
            appearances=appearances,
            include_share=True,
            streaming=False,
        )
        self._converted_matches_df = mdf
        if mdf.is_empty():
            self.logger.warning(
                "No valid matches after conversion; returning empty result."
            )
            return pl.DataFrame()

        # 3) Restrict nodes to active players
        node_ids = active_players
        node_to_idx = {pid: idx for idx, pid in enumerate(node_ids)}
        num_nodes = len(node_ids)

        # 4) Build teleport ρ from exposure mass
        winners_share = (
            mdf.select(["winners", "share"])
            .explode("winners")
            .rename({"winners": "id"})
        )
        losers_share = (
            mdf.select(["losers", "share"])
            .explode("losers")
            .rename({"losers": "id"})
        )
        exp_share_df = (
            pl.concat([winners_share, losers_share])
            .group_by("id")
            .agg(pl.col("share").sum().alias("e_share"))
        )
        e_share = np.zeros(num_nodes, dtype=float)
        for row in exp_share_df.iter_rows(named=True):
            idx = node_to_idx.get(row["id"])
            if idx is not None:
                e_share[idx] = float(row["e_share"])

        eps = 1e-12
        rho = e_share + eps
        if rho.sum() == 0.0 or not np.isfinite(rho.sum()):
            rho[:] = 1.0
        rho = rho / rho.sum()
        self._last_rho = rho

        # 5) Triplets for A_win (loser -> winner), then mirror for A_loss
        rows, cols, data = build_exposure_triplets(mdf, node_to_idx)

        pr_cfg = PageRankConfig(
            alpha=self.config.pagerank.alpha,
            tol=self.config.pagerank.tol,
            max_iter=self.config.pagerank.max_iter,
            orientation="col",
            redistribute_dangling=True,
        )

        # 6) Win & loss PageRanks with SAME ρ
        self.logger.info("Computing win PageRank...")
        win_pagerank = pagerank_sparse(rows, cols, data, num_nodes, rho, pr_cfg)
        self.logger.info("Computing loss PageRank...")
        loss_pagerank = pagerank_sparse(
            cols, rows, data, num_nodes, rho, pr_cfg
        )

        # 7) Lambda smoothing
        if self.config.fixed_lambda is not None:
            lambda_smooth = float(self.config.fixed_lambda)
        elif self.config.lambda_mode == "auto":
            target = 0.025 * float(np.median(win_pagerank))
            median_rho = float(np.median(rho))
            lambda_smooth = (
                0.0 if median_rho == 0.0 else max(target / median_rho, 0.0)
            )
        else:
            lambda_smooth = self.config.engine.lambda_smooth or 1e-4
        self.logger.info(f"Lambda used: {lambda_smooth:.6f}")

        # 8) Log-odds scores
        win_smooth = win_pagerank + lambda_smooth * rho
        loss_smooth = loss_pagerank + lambda_smooth * rho
        scores = (
            np.log(win_smooth / loss_smooth)
            if self.config.apply_log_transform
            else (win_smooth / loss_smooth)
        )

        # 9) Inactivity decay
        if self.config.engine.score_decay_rate > 0:
            last_ts = self._last_activity_times(mdf, node_to_idx, num_nodes)
            scores = apply_inactivity_decay(
                scores,
                last_ts,
                self.clock.now,
                delay_days=self.config.engine.score_decay_delay_days,
                decay_rate=self.config.engine.score_decay_rate,
            )

        # 10) Calculate reporting exposure
        winners_w = (
            mdf.select(["winners", "weight"])
            .explode("winners")
            .rename({"winners": "id"})
        )
        losers_w = (
            mdf.select(["losers", "weight"])
            .explode("losers")
            .rename({"losers": "id"})
        )
        exp_w_df = (
            pl.concat([winners_w, losers_w])
            .group_by("id")
            .agg(pl.col("weight").sum().alias("exposure"))
        )
        exposure = np.zeros(num_nodes, dtype=float)
        for row in exp_w_df.iter_rows(named=True):
            idx = node_to_idx.get(row["id"])
            if idx is not None:
                exposure[idx] = float(row["exposure"])

        # 11) Apply minimum exposure filter if configured
        mask = np.ones(num_nodes, dtype=bool)
        if self.config.engine.min_exposure is not None:
            mask = exposure >= float(self.config.engine.min_exposure)

        # 12) Build result dataframe
        result = (
            pl.DataFrame(
                {
                    "id": node_ids,
                    "player_rank": scores.tolist(),
                    "score": scores.tolist(),
                    "win_pr": win_pagerank.tolist(),
                    "loss_pr": loss_pagerank.tolist(),
                    "exposure": exposure.tolist(),
                }
            )
            .filter(pl.Series(mask.tolist()))
            .sort("player_rank", descending=True)
        )

        # 13) Store diagnostics for downstream analysis
        self.last_result = ExposureLogOddsResult(
            scores=scores,
            ids=node_ids,
            win_pagerank=win_pagerank,
            loss_pagerank=loss_pagerank,
            exposure=rho,
            lambda_used=lambda_smooth,
            active_mask=mask,
            raw_scores=scores.copy(),
        )

        # Store tournament influence for analysis tools
        self.tournament_influence = tournament_influence

        log_timing("exposure_log_odds_total", time.time() - start_time)
        return result

    def prepare_loo_analyzer(
        self,
        matches_df: pl.DataFrame | None = None,
        players_df: pl.DataFrame | None = None,
        force_rebuild: bool = False,
    ) -> None:
        """
        Prepare LOO analyzer with pre-factorized solvers for fast repeated analysis.

        This should be called after rank_players() to set up the LOO infrastructure.
        The analyzer is cached and reused unless force_rebuild is True.

        Args:
            matches_df: Optional matches DataFrame (uses internal converted if None)
            players_df: Optional players DataFrame (uses last ranking data if None)
            force_rebuild: Force rebuild even if analyzer exists
        """
        if self.last_result is None:
            raise ValueError(
                "Must call rank_players() before preparing LOO analyzer"
            )

        # Use internal converted matches if not provided
        if matches_df is None:
            if self._converted_matches_df is None:
                raise ValueError(
                    "No converted matches available. Call rank_players() first."
                )
            matches_df = self._converted_matches_df
            self.logger.info("Using internally converted matches dataframe")

        # Use last result's player data if not provided
        if players_df is None:
            # Create a simple players dataframe from last result
            players_df = pl.DataFrame(
                {
                    "player_id": self.last_result.ids,
                    "name": [f"Player_{pid}" for pid in self.last_result.ids],
                }
            )
            self.logger.info("Using player IDs from last ranking result")

        # Check if we need to rebuild
        if not force_rebuild and self._loo_analyzer is not None:
            # Check if data has changed
            if (
                self._loo_matches_df is not None
                and matches_df.equals(self._loo_matches_df)
                and self._loo_players_df is not None
                and players_df.equals(self._loo_players_df)
            ):
                self.logger.info(
                    "LOO analyzer already prepared, reusing existing infrastructure"
                )
                return

        self.logger.info(
            "Preparing LOO analyzer with pre-factorized solvers..."
        )
        from rankings.analysis.loo_analyzer import LOOAnalyzer

        start_time = time.time()
        self._loo_analyzer = LOOAnalyzer(self, matches_df, players_df)
        self._loo_matches_df = matches_df
        self._loo_players_df = players_df

        prep_time = time.time() - start_time
        self.logger.info(f"LOO analyzer prepared in {prep_time:.2f}s")
        self.logger.info(f"  - {len(self._loo_analyzer.node_to_idx)} nodes")
        self.logger.info(f"  - {self._loo_analyzer.A_win.nnz} win edges")
        self.logger.info(f"  - {self._loo_analyzer.A_loss.nnz} loss edges")
        self.logger.info(
            f"  - {len(self._loo_analyzer._match_cache)} matches cached"
        )

    def analyze_match_impact(
        self, match_id: int, player_id: int, include_teleport: bool = True
    ) -> dict[str, Any]:
        """
        Analyze the impact of removing a match on a player's score.

        Automatically prepares LOO analyzer if needed.
        Uses pre-factorized solvers for extremely fast analysis.

        Args:
            match_id: Match to analyze
            player_id: Player to check impact for
            include_teleport: Whether to include teleport vector changes

        Returns:
            Dictionary with impact analysis
        """
        # Auto-prepare LOO analyzer if not ready
        if self._loo_analyzer is None:
            self.logger.info("LOO analyzer not prepared, initializing now...")
            self.prepare_loo_analyzer()

        return self._loo_analyzer.impact_of_match_on_player(
            match_id, player_id, include_teleport
        )

    def analyze_player_matches(
        self,
        player_id: int,
        limit: int | None = None,
        include_teleport: bool = True,
        parallel: bool = True,
        max_workers: int = 4,
    ) -> pl.DataFrame:
        """
        Analyze impact of all matches involving a player.

        Automatically prepares LOO analyzer if needed.
        Uses pre-factorized solvers and parallel processing for speed.

        Args:
            player_id: Player to analyze
            limit: Maximum number of matches to analyze
            include_teleport: Whether to include teleport changes
            parallel: Use parallel processing
            max_workers: Number of parallel workers

        Returns:
            DataFrame with match impacts sorted by score change
        """
        # Auto-prepare LOO analyzer if not ready
        if self._loo_analyzer is None:
            self.logger.info("LOO analyzer not prepared, initializing now...")
            self.prepare_loo_analyzer()

        return self._loo_analyzer.analyze_player_matches(
            player_id,
            limit,
            include_teleport,
            use_flux_ranking=True,
            parallel=parallel,
            max_workers=max_workers,
        )

    def get_loo_analyzer(self):
        """
        Get the underlying LOO analyzer for advanced usage.

        Returns:
            LOOAnalyzer instance or None if not prepared
        """
        return self._loo_analyzer

    # ---------------------------------------------------------------------
    # internals
    # ---------------------------------------------------------------------
    def _run_tick_tock_for_active_players(
        self,
        matches: pl.DataFrame,
        players: pl.DataFrame,
    ) -> dict[str, Any]:
        from rankings.algorithms.tick_tock import TickTockEngine

        engine = TickTockEngine(self.config.tick_tock)
        # Pass deterministic time if we were given one
        if hasattr(self.clock, "now"):
            try:
                engine.clock = self.clock
            except Exception:
                pass

        tt_df = engine.rank_players(matches, players)
        active_players = (
            tt_df["player_id"].to_list()
            if "player_id" in tt_df.columns
            else (tt_df["id"].to_list() if "id" in tt_df.columns else [])
        )
        tournament_influence_data = (
            getattr(engine, "tournament_influence", {}) or {}
        )
        return {
            "active_players": active_players,
            "tournament_influence": tournament_influence_data,
        }

    def _last_activity_times(
        self,
        matches_df: pl.DataFrame,
        node_to_idx: dict[int, int],
        num_nodes: int,
    ) -> np.ndarray:
        """Get the last activity timestamp for each player.

        Args:
            matches_df: DataFrame containing match data.
            node_to_idx: Mapping from node IDs to indices.
            num_nodes: Total number of nodes.

        Returns:
            Array of last activity timestamps.
        """
        last_ts = np.zeros(num_nodes, dtype=float)

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
            idx = node_to_idx.get(row["id"])
            if idx is not None:
                last_ts[idx] = float(row["last_ts"])

        last_ts[last_ts == 0.0] = float(self.clock.now)
        return last_ts
