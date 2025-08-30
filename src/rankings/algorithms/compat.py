"""Compatibility wrapper exposing a RatingEngine API backed by algorithms.

This preserves the historical `rankings.RatingEngine` interface while
internally delegating to the modern engines in `rankings.algorithms`.

Currently, `rankings.RatingEngine.rank_players` is implemented via
`TickTockEngine` to provide a PageRank-based player ranking with
legacy-compatible defaults and output schema (id + player_rank).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import polars as pl

from rankings.algorithms.tick_tock import TickTockEngine
from rankings.core import TickTockConfig


def _map_influence_method(method: str | None) -> str:
    """Map legacy influence aggregation names to TickTockConfig semantics."""
    if method is None:
        return "arithmetic"
    m = method.lower()
    if m in ("mean", "arithmetic"):
        return "arithmetic"
    if m in ("geometric", "geom"):
        return "geometric"
    if m in ("top_20_sum", "top20_sum", "top20"):
        return "top_20_sum"
    if m in ("top_20_geom", "top20_geom"):
        return "top_20_geom"
    # Fallback
    return "arithmetic"


@dataclass
class RatingEngine:
    """Legacy-compatible RatingEngine facade using TickTockEngine under the hood.

    Supported init kwargs (others are ignored with defaults):
    - decay_half_life_days: float
    - damping_factor: float (PageRank alpha)
    - beta: float (tournament influence exponent)
    - influence_agg_method: str ("mean"|"geometric"|"top_20_sum"|"top_20_geom")
    - max_tick_tock: int
    - tick_tock_stabilize_tol: float
    - max_pagerank_iter: int
    - pagerank_tol: float
    """

    # Store only the parameters we translate; additional kwargs are accepted
    decay_half_life_days: float = 30.0
    damping_factor: float = 0.85
    beta: float = 1.0
    influence_agg_method: str = "mean"
    max_tick_tock: int = 5
    tick_tock_stabilize_tol: float = 1e-4
    max_pagerank_iter: int = 200
    pagerank_tol: float = 1e-8

    def __init__(self, **kwargs: Any) -> None:  # type: ignore[override]
        # Extract known params; keep defaults for the rest
        self.decay_half_life_days = float(
            kwargs.get("decay_half_life_days", self.decay_half_life_days)
        )
        self.damping_factor = float(
            kwargs.get("damping_factor", self.damping_factor)
        )
        self.beta = float(kwargs.get("beta", self.beta))
        self.influence_agg_method = str(
            kwargs.get("influence_agg_method", self.influence_agg_method)
        )
        self.max_tick_tock = int(
            kwargs.get("max_tick_tock", self.max_tick_tock)
        )
        self.tick_tock_stabilize_tol = float(
            kwargs.get("tick_tock_stabilize_tol", self.tick_tock_stabilize_tol)
        )
        self.max_pagerank_iter = int(
            kwargs.get("max_pagerank_iter", self.max_pagerank_iter)
        )
        self.pagerank_tol = float(kwargs.get("pagerank_tol", self.pagerank_tol))

        # Build TickTockEngine configuration
        cfg = TickTockConfig()
        cfg.decay.half_life_days = self.decay_half_life_days
        cfg.pagerank.alpha = self.damping_factor
        cfg.engine.beta = self.beta
        cfg.max_ticks = self.max_tick_tock
        cfg.convergence_tol = self.tick_tock_stabilize_tol
        cfg.pagerank.max_iter = self.max_pagerank_iter
        cfg.pagerank.tol = self.pagerank_tol
        cfg.influence_method = _map_influence_method(self.influence_agg_method)

        self._engine = TickTockEngine(config=cfg)

    # ------------------------------------------------------------------ API
    def rank_players(
        self,
        *,
        matches: pl.DataFrame,
        players: pl.DataFrame,
    ) -> pl.DataFrame:
        """Compute player rankings; returns legacy schema (id + player_rank)."""
        out = self._engine.rank_players(matches, players)
        if out.is_empty():
            return out
        # Adapt to legacy column names
        cols = out.columns
        # TickTockEngine returns [player_id, rating]
        rename_map = {}
        if "player_id" in cols:
            rename_map["player_id"] = "id"
        if "rating" in cols:
            rename_map["rating"] = "player_rank"
        return out.rename(rename_map)
