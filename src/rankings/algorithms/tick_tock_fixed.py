# === PATCH: Refactored TickTockEngine (parity to legacy tick–tock) ===

import math
import time
from typing import Dict, List, Optional

import numpy as np
import polars as pl

from rankings.core import (
    Clock,
    PageRankConfig,
    TickTockConfig,
    TickTockResult,
    UniformTeleport,
    VolumeInverseTeleport,
    WinsProportional,
    build_player_edges,
    compute_denominators,
    edges_to_triplets,
    normalize_edges,
    pagerank_sparse,
)
from rankings.core.logging import get_logger, log_timing


class TickTockEngine:
    """
    Refactored Tick-Tock rating engine with legacy-compatible 'tock':
      - participants = players who actually appeared (via used teams)
      - id-keyed influence with prior fill, legacy aggregations
      - mean-normalize influences to 1.0
      - tight convergence default (1e-4)
    """

    def __init__(
        self,
        config: Optional[TickTockConfig] = None,
        *,
        now_ts: Optional[float] = None,
        clock: Optional[Clock] = None,
    ):
        self.config = config or TickTockConfig()
        # enforce legacy-like tolerance if config left wide
        if (
            getattr(self.config, "convergence_tol", None) is None
            or self.config.convergence_tol > 1e-4
        ):
            self.config.convergence_tol = 1e-4

        self.logger = get_logger(self.__class__.__name__)
        self.logger.info(
            f"TickTockEngine initialized: max_ticks={self.config.max_ticks}, "
            f"convergence_tol={self.config.convergence_tol}, "
            f"influence_method={self.config.influence_method}"
        )

        if clock is not None:
            self.clock = clock
        else:
            self.clock = Clock(
                now_timestamp=(now_ts if now_ts is not None else time.time())
            )

        # Teleport
        if self.config.teleport_mode == "uniform":
            self.teleport_strategy = UniformTeleport()
        else:
            self.teleport_strategy = VolumeInverseTeleport()

        self.smoothing_strategy = WinsProportional(
            gamma=self.config.engine.gamma,
            cap_ratio=self.config.engine.cap_ratio,
        )

        self.last_result: Optional[TickTockResult] = None
        self.tournament_influence: Dict[int, float] = {}

    # ------------------------------------------------------------------ public
    def rank_players(
        self,
        matches: pl.DataFrame,
        players: pl.DataFrame,
        initial_influence: Optional[Dict[int, float]] = None,
    ) -> pl.DataFrame:
        start_time = time.time()

        # Initial S
        if initial_influence:
            self.tournament_influence = initial_influence.copy()
        else:
            tids = matches["tournament_id"].unique().to_list()
            self.tournament_influence = {int(t): 1.0 for t in tids}

        tick_history, tock_history = [], []
        converged = False
        tick_result = None

        for it in range(self.config.max_ticks):
            self.logger.info(
                f"Tick-tock iteration {it + 1}/{self.config.max_ticks}"
            )

            tick_result = self._tick(
                matches, players, self.tournament_influence
            )
            if tick_result["pagerank"].size == 0:
                self.logger.warning("No PageRank computed; breaking.")
                break

            tick_history.append(tick_result["pagerank"])

            # Build participants *who actually played* (legacy parity)
            participants = self._participants_by_tournament_played(
                matches, players
            )

            # Legacy-compatible influence computation (id-keyed with prior)
            scores_by_id = dict(
                zip(tick_result["node_ids"], tick_result["pagerank"])
            )
            global_prior = float(np.quantile(tick_result["pagerank"], 0.05))
            new_infl = self._compute_tournament_influence_compat(
                scores_by_id,
                participants,
                method=self.config.influence_method,
                global_prior=global_prior,
            )

            # Mean-normalize to 1.0 (legacy)
            if new_infl:
                vals = np.array(list(new_infl.values()), dtype=float)
                mean_val = float(vals.mean())
                if mean_val > 0:
                    new_infl = {k: (v / mean_val) for k, v in new_infl.items()}

            tock_history.append(new_infl)

            # Convergence on post-normalization values
            keys = set(self.tournament_influence.keys()) | set(new_infl.keys())
            delta = sum(
                abs(
                    new_infl.get(k, 1.0) - self.tournament_influence.get(k, 1.0)
                )
                for k in keys
            ) / max(len(keys), 1)
            self.logger.info(f"  Tournament influence delta: {delta:.6f}")

            self.tournament_influence = new_infl
            if delta < self.config.convergence_tol:
                self.logger.info(
                    f"Tick-tock converged at iteration {it + 1} (delta={delta:.6f})"
                )
                converged = True
                break

        if not tick_result:
            return pl.DataFrame()

        # Retain last results
        self.last_result = TickTockResult(
            scores=tick_result["pagerank"],
            ids=tick_result["node_ids"],
            win_pagerank=tick_result["pagerank"],
            iterations=(it + 1),
            converged=converged,
            tournament_influence=self.tournament_influence,
            retrospective_strength=None,
            tick_history=tick_history,
            tock_history=tock_history,
            denominators=tick_result.get("denominators"),
        )

        out = pl.DataFrame(
            {
                "player_id": tick_result["node_ids"],
                "rating": tick_result["pagerank"].tolist(),
            }
        )
        log_timing("tick_tock_total", time.time() - start_time)
        return out.sort("rating", descending=True)

    # ------------------------------------------------------------------ tick
    def _tick(
        self,
        matches: pl.DataFrame,
        players: pl.DataFrame,
        tournament_influence: Dict[int, float],
    ) -> Dict:
        edges = build_player_edges(
            matches,
            players,
            tournament_influence,
            self.clock.now,
            self.config.decay.decay_rate,
            self.config.engine.beta,
        )
        if edges.is_empty():
            return {
                "pagerank": np.array([]),
                "node_ids": [],
                "node_to_idx": {},
                "edges": edges,
                "denominators": None,
            }

        # Nodes
        ids = set(edges["loser_user_id"].unique().to_list())
        ids.update(edges["winner_user_id"].unique().to_list())
        node_ids = sorted(list(ids))
        node_to_idx = {n: i for i, n in enumerate(node_ids)}
        n = len(node_ids)

        # Denominators + normalization (legacy smoothing semantics)
        denoms = compute_denominators(
            edges,
            self.smoothing_strategy,
            loser_column="loser_user_id",
            winner_column="winner_user_id",
        )
        edges = normalize_edges(edges, denoms, loser_column="loser_user_id")

        # Triplets
        rows, cols, w = edges_to_triplets(
            edges,
            node_to_idx,
            source_column="loser_user_id",
            target_column="winner_user_id",
            weight_column="normalized_weight",
        )

        # Teleport
        v = self.teleport_strategy(node_ids, edges, "loser_user_id")

        # Row PR with deficit redistribution (pagerank_sparse should handle dangling/deficit when configured)
        pr_cfg = PageRankConfig(
            alpha=self.config.pagerank.alpha,
            tol=self.config.pagerank.tol,
            max_iter=self.config.pagerank.max_iter,
            orientation="row",
            redistribute_dangling=True,
        )
        r = pagerank_sparse(rows, cols, w, n, v, pr_cfg)

        return {
            "pagerank": r,
            "node_ids": node_ids,
            "node_to_idx": node_to_idx,
            "edges": edges,
            "denominators": denoms,
        }

    # ------------------------------------------------------ participants (played)
    def _participants_by_tournament_played(
        self,
        matches: pl.DataFrame,
        players: pl.DataFrame,
    ) -> Dict[int, List]:
        """Players who actually appeared in a tournament (join used teams -> rosters)."""
        used_teams = pl.concat(
            [
                matches.select(
                    pl.col("tournament_id"),
                    pl.col("winner_team_id").alias("team_id"),
                ),
                matches.select(
                    pl.col("tournament_id"),
                    pl.col("loser_team_id").alias("team_id"),
                ),
            ]
        ).unique()
        # roster subset for used teams
        rosters = (
            players.join(
                used_teams, on=["tournament_id", "team_id"], how="inner"
            )
            .select(["tournament_id", "user_id"])
            .unique()
        )
        grouped = rosters.group_by("tournament_id").agg(
            pl.col("user_id").alias("participants")
        )

        result: Dict[int, List] = {}
        for row in grouped.iter_rows(named=True):
            result[int(row["tournament_id"])] = list(row["participants"])
        return result

    # ------------------------------------------------------ legacy-compatible tock
    def _compute_tournament_influence_compat(
        self,
        scores_by_id: Dict,
        participants: Dict[int, List],
        *,
        method: str,
        global_prior: float,
    ) -> Dict[int, float]:
        """Mirror legacy _compute_tournament_influence aggregation."""
        out: Dict[int, float] = {}
        for tid, ids in participants.items():
            vals = [float(scores_by_id.get(pid, global_prior)) for pid in ids]
            if not vals:
                # if empty, treat as single prior to avoid NaN and keep tournament alive
                vals = [global_prior]

            arr = np.asarray(vals, dtype=float)

            if method == "mean":
                S_raw = float(arr.mean())
            elif method == "sum":
                S_raw = float(arr.sum())
            elif method == "median":
                S_raw = float(np.median(arr))
            elif method == "top_20_sum":
                S_raw = float(np.sort(arr)[-min(20, arr.size) :].sum())
            elif method == "log_top_20_sum":
                top = float(np.sort(arr)[-min(20, arr.size) :].sum())
                # normalize per-tournament later; keep raw for now
                S_raw = top
            elif method == "sqrt_top_20_sum":
                top = float(np.sort(arr)[-min(20, arr.size) :].sum())
                S_raw = math.sqrt(max(top, 0.0))
            elif method == "top_10_sum":
                S_raw = float(np.sort(arr)[-min(10, arr.size) :].sum())
            elif method == "top_20_mean":
                k = min(20, arr.size)
                S_raw = float(np.sort(arr)[-k:].mean())
            else:
                raise ValueError(f"Unknown influence_agg_method: {method}")

            out[int(tid)] = S_raw

        # Mean-normalize
        vals = np.array(list(out.values()), dtype=float)
        m = float(vals.mean()) if vals.size else 1.0
        if m == 0.0 or not np.isfinite(m):
            m = 1.0
        out = {k: (v / m) for k, v in out.items()}

        # Special handling for log_top_20_sum (legacy: log1p after normalization, then renormalize)
        if method == "log_top_20_sum":
            vals = np.array(list(out.values()), dtype=float)
            vals = np.log1p(np.maximum(vals, 0.0))
            m2 = float(vals.mean()) if vals.size else 1.0
            m2 = m2 if m2 > 0 else 1.0
            out = {k: (vals[i] / m2) for i, k in enumerate(out.keys())}

        return out
