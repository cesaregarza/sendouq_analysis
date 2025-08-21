# === PATCH: LogOddsBackend (parity to legacy log-odds semantics) ===

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
    ):
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
        active_ids: list,
        tournament_influence: dict[int, float],
        **kwargs,
    ) -> pl.DataFrame:
        if players is None:
            self.logger.warning(
                "No player data provided, cannot compute ratings"
            )
            return pl.DataFrame()

        mdf = convert_matches_dataframe(
            matches,
            players,
            tournament_influence or {},
            self.clock.now,
            self.decay_rate,
            self.beta,
            include_share=True,
            streaming=False,
        )
        if mdf.is_empty():
            self.logger.warning("No valid matches after conversion")
            return pl.DataFrame()

        # Nodes = ALL who appeared in converted matches (legacy backend doesn't filter to 'active_ids')
        all_ids = set()
        for row in mdf.iter_rows(named=True):
            all_ids.update(row["winners"])
            all_ids.update(row["losers"])
        node_ids = sorted(list(all_ids))
        node_to_idx = {pid: i for i, pid in enumerate(node_ids)}
        n = len(node_ids)

        # Teleport ρ from exposure share
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
        e_df = (
            pl.concat([winners_share, losers_share])
            .group_by("id")
            .agg(pl.col("share").sum().alias("e_share"))
        )

        rho = np.full(n, self.epsilon, dtype=float)
        for row in e_df.iter_rows(named=True):
            idx = node_to_idx.get(row["id"])
            if idx is not None:
                rho[idx] += float(row["e_share"])
        rho /= rho.sum() if rho.sum() > 0 else 1.0
        self._last_rho = rho

        # Triplets
        rows, cols, data = build_exposure_triplets(mdf, node_to_idx)

        pr_cfg = PageRankConfig(
            alpha=self.alpha,
            tol=self.pagerank_tol,
            max_iter=self.pagerank_max_iter,
            orientation="col",
            redistribute_dangling=True,
        )

        s = pagerank_sparse(rows, cols, data, n, rho, pr_cfg)
        l = pagerank_sparse(cols, rows, data, n, rho, pr_cfg)

        # Lambda (legacy auto rule)
        if self.lambda_mode == "fixed" and self.fixed_lambda is not None:
            lam = float(self.fixed_lambda)
        elif self.lambda_mode == "auto":
            target = 0.025 * float(np.median(s))
            med_rho = float(np.median(rho))
            lam = 0.0 if med_rho == 0.0 else max(target / med_rho, 0.0)
        else:
            lam = 1e-4

        s_s = s + lam * rho
        l_s = l + lam * rho
        scores = np.log(s_s / l_s)
        quality_mass = s_s / (s_s + l_s)

        # Reporting exposure = sum of 'weight' (legacy)
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
        ew_df = (
            pl.concat([winners_w, losers_w])
            .group_by("id")
            .agg(pl.col("weight").sum().alias("exposure"))
        )

        exposure = np.zeros(n, dtype=float)
        for row in ew_df.iter_rows(named=True):
            idx = node_to_idx.get(row["id"])
            if idx is not None:
                exposure[idx] = float(row["exposure"])

        return pl.DataFrame(
            {
                "id": node_ids,
                "score": scores.tolist(),
                "quality_mass": quality_mass.tolist(),
                "win_pr": s.tolist(),
                "loss_pr": l.tolist(),
                "exposure": exposure.tolist(),  # reporting exposure
                "lambda_used": [lam] * n,
            }
        )
