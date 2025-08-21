# === PATCH: Refactored ExposureLogOddsEngine (parity with old) ===

import math
import time
from typing import Dict, Optional

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


class ExposureLogOddsEngine:
    """
    Exposure Log-Odds rating engine using modular components, updated to mirror the
    legacy implementation semantics:
      - Teleport ρ is exposure-based (sum of match pair 'share' per player)
      - Two PageRanks with the SAME ρ (col-stochastic)
      - Lambda auto-tuned to ~2.5% of median PR per node (dividing by median ρ)
      - Reporting 'exposure' uses sum of 'weight' (legacy reporting semantics)
      - Optional time-decay after inactivity delay
      - Output schema: id + player_rank (legacy-compatible)
    """

    def __init__(
        self,
        config: Optional[ExposureLogOddsConfig] = None,
        *,
        now_ts: Optional[float] = None,
        clock: Optional[Clock] = None,
    ):
        self.config = config or ExposureLogOddsConfig()
        self.logger = get_logger(self.__class__.__name__)

        # Deterministic time if provided; otherwise wall-clock
        if clock is not None:
            self.clock = clock
        else:
            self.clock = Clock(
                now_ts=(now_ts if now_ts is not None else time.time())
            )

        self.last_result: Optional[ExposureLogOddsResult] = None
        # keep last teleport for diagnostics
        self._last_rho: Optional[np.ndarray] = None

    def rank_players(
        self,
        matches: pl.DataFrame,
        players: pl.DataFrame,
        tournament_influence: Optional[Dict[int, float]] = None,
    ) -> pl.DataFrame:
        start_time = time.time()

        # 1) Get active players and tournament influences via tick‑tock, like the legacy path
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

        # 2) Convert matches into compact per-match lists with roster expansion and weights
        self.logger.info("Converting matches...")
        mdf = convert_matches_dataframe(
            matches,
            players,
            tournament_influence or {},
            self.clock.now,
            self.config.decay.decay_rate,
            self.config.engine.beta,
            include_share=True,  # IMPORTANT: we need 'share' for exposure teleport
            streaming=False,
        )
        if mdf.is_empty():
            self.logger.warning(
                "No valid matches after conversion; returning empty result."
            )
            return pl.DataFrame()

        # 3) Restrict nodes to *active players*, preserving legacy behavior
        node_ids = active_players
        node_to_idx = {pid: i for i, pid in enumerate(node_ids)}
        n = len(node_ids)

        # 4) Build teleport ρ from exposure mass (sum of 'share' over winners+losers)
        #    This mirrors the legacy optimized path (_exposure_logodds_optimized).
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
        e_share = np.zeros(n, dtype=float)
        for row in exp_share_df.iter_rows(named=True):
            idx = node_to_idx.get(row["id"])
            if idx is not None:
                e_share[idx] = float(row["e_share"])

        eps = 1e-12  # Use a small epsilon for numerical stability
        rho = e_share + eps
        if rho.sum() == 0.0 or not np.isfinite(rho.sum()):
            rho[:] = 1.0  # fallback to uniform if degenerate
        rho = rho / rho.sum()
        self._last_rho = rho

        # 5) Triplets for A_win (loser -> winner), then mirror for A_loss
        rows, cols, data = build_exposure_triplets(mdf, node_to_idx)

        pr_cfg = PageRankConfig(
            alpha=self.config.pagerank.alpha,
            tol=self.config.pagerank.tol,
            max_iter=self.config.pagerank.max_iter,
            orientation="col",  # Column-stochastic (A[dst, src])
            redistribute_dangling=True,  # Mirror legacy handling
        )

        # 6) Win & loss PageRanks with SAME ρ
        self.logger.info("Computing win PageRank...")
        s = pagerank_sparse(rows, cols, data, n, rho, pr_cfg)
        self.logger.info("Computing loss PageRank...")
        l = pagerank_sparse(cols, rows, data, n, rho, pr_cfg)  # transpose graph

        # 7) Lambda smoothing (legacy calibration)
        if self.config.fixed_lambda is not None:
            lam = float(self.config.fixed_lambda)
        elif self.config.lambda_mode == "auto":
            target = 0.025 * float(np.median(s))
            med_rho = float(np.median(rho))
            lam = 0.0 if med_rho == 0.0 else max(target / med_rho, 0.0)
        else:
            lam = self.config.engine.lambda_smooth or 1e-4
        self.logger.info(f"Lambda used: {lam:.6f}")

        # 8) Log-odds scores
        s_smooth = s + lam * rho
        l_smooth = l + lam * rho
        scores = (
            np.log(s_smooth / l_smooth)
            if self.config.apply_log_transform
            else (s_smooth / l_smooth)
        )

        # 9) Inactivity decay (same semantics as legacy)
        if self.config.engine.score_decay_rate > 0:
            last_ts = self._last_activity_times(mdf, node_to_idx, n)
            scores = apply_inactivity_decay(
                scores,
                last_ts,
                self.clock.now,
                delay_days=self.config.engine.score_decay_delay_days,
                decay_rate=self.config.engine.score_decay_rate,
            )

        # 10) Reporting exposure (legacy: sum of match 'weight' over winners+losers)
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
        exposure = np.zeros(n, dtype=float)
        for row in exp_w_df.iter_rows(named=True):
            idx = node_to_idx.get(row["id"])
            if idx is not None:
                exposure[idx] = float(row["exposure"])

        # 11) Optional minimum exposure filter (legacy semantics use reporting exposure)
        mask = np.ones(n, dtype=bool)
        if self.config.engine.min_exposure is not None:
            mask = exposure >= float(self.config.engine.min_exposure)

        # 12) Build result with legacy column names + score for compatibility
        result = (
            pl.DataFrame(
                {
                    "id": node_ids,
                    "player_rank": scores.tolist(),
                    "score": scores.tolist(),  # Add score as alias for player_rank
                    "win_pr": s.tolist(),
                    "loss_pr": l.tolist(),
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
            win_pr=s,
            loss_pr=l,
            exposure=rho,  # store teleport used for scoring
            lambda_used=lam,
            active_mask=mask,
            raw_scores=scores.copy(),
        )

        log_timing("exposure_log_odds_total", time.time() - start_time)
        return result

    # ---------------------------------------------------------------------
    # internals
    # ---------------------------------------------------------------------
    def _run_tick_tock_for_active_players(
        self,
        matches: pl.DataFrame,
        players: pl.DataFrame,
    ) -> Dict:
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
        t_infl = getattr(engine, "tournament_influence", {}) or {}
        return {
            "active_players": active_players,
            "tournament_influence": t_infl,
        }

    def _last_activity_times(
        self,
        matches_df: pl.DataFrame,
        node_to_idx: Dict,
        n: int,
    ) -> np.ndarray:
        """Max ts over appearances; fill missing with 'now' (legacy behavior)."""
        last_ts = np.zeros(n, dtype=float)

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

        # players that never matched in window: treat as "just now" (no decay)
        last_ts[last_ts == 0.0] = float(self.clock.now)
        return last_ts
