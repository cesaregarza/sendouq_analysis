from __future__ import annotations

"""Serialization helpers for ranking engine state.

This captures the minimal information needed to reproduce match influence
queries later without recomputing the full engine:
  - Tournament influence map
  - Player vectors (win/loss PageRank, exposure) and ids from last run
  - Basic config (decay rate, alpha, beta) and clock timestamp

It also provides a light-weight adapter object compatible with
`rankings.analysis.utils.matches._get_engine_attributes` so downstream
helpers can be used with a saved state.
"""

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


@dataclass
class _ResultNS:
    ids: List[int]
    scores: List[float]
    win_pagerank: Optional[List[float]]
    loss_pagerank: Optional[List[float]]
    exposure: Optional[List[float]]
    lambda_used: Optional[float]


def _get_config_namespace(engine) -> Any:
    cfg = getattr(engine, "config", None)
    if cfg is None:
        return SimpleNamespace(
            decay=SimpleNamespace(decay_rate=0.0),
            pagerank=SimpleNamespace(alpha=0.85, tol=1e-8, max_iter=100),
            engine=SimpleNamespace(beta=0.0),
        )
    return cfg


def save_engine_state(engine, path: str) -> None:
    """Serialize essential engine state to JSON file."""
    cfg = _get_config_namespace(engine)
    last = getattr(engine, "last_result", None)
    state = {
        "now": getattr(getattr(engine, "clock", None), "now", None),
        "config": {
            "decay_rate": getattr(
                getattr(cfg, "decay", None), "decay_rate", 0.0
            ),
            "alpha": getattr(getattr(cfg, "pagerank", None), "alpha", 0.85),
            "beta": getattr(getattr(cfg, "engine", None), "beta", 0.0),
        },
        "tournament_influence": getattr(engine, "tournament_influence", {})
        or {},
        "result": None,
    }
    if last is not None:

        def _to_list_or_empty(value):
            if value is None:
                return []
            try:
                return (
                    value.tolist() if hasattr(value, "tolist") else list(value)
                )
            except Exception:
                # Fallback for scalars or non-iterables
                return [value] if not isinstance(value, list) else value

        ids = getattr(last, "ids", None)
        scores = getattr(last, "scores", None)
        win_pr = getattr(last, "win_pagerank", None)
        loss_pr = getattr(last, "loss_pagerank", None)
        exposure = getattr(last, "exposure", None)

        state["result"] = {
            "ids": _to_list_or_empty(ids),
            "scores": _to_list_or_empty(scores),
            "win_pagerank": _to_list_or_empty(win_pr),
            "loss_pagerank": _to_list_or_empty(loss_pr),
            "exposure": _to_list_or_empty(exposure),
            "lambda_used": getattr(last, "lambda_used", None),
        }

    with open(path, "w") as f:
        json.dump(state, f)


def load_engine_state(path: str):
    """Load saved engine state and return a light-weight adapter.

    The adapter exposes attributes used by `analysis.utils.matches`:
      - `tournament_influence`
      - `last_result` with ids/scores/win_pr/loss_pr/exposure
      - `config` with decay_rate, alpha, beta
      - `clock.now`
    """
    with open(path, "r") as f:
        state = json.load(f)

    cfg = SimpleNamespace(
        decay=SimpleNamespace(
            decay_rate=state.get("config", {}).get("decay_rate", 0.0)
        ),
        pagerank=SimpleNamespace(
            alpha=state.get("config", {}).get("alpha", 0.85)
        ),
        engine=SimpleNamespace(beta=state.get("config", {}).get("beta", 0.0)),
    )

    res = state.get("result") or {}
    last = _ResultNS(
        ids=res.get("ids", []) or [],
        scores=res.get("scores", []) or [],
        win_pagerank=res.get("win_pagerank", []) or [],
        loss_pagerank=res.get("loss_pagerank", []) or [],
        exposure=res.get("exposure", []) or [],
        lambda_used=res.get("lambda_used"),
    )

    adapter = SimpleNamespace(
        tournament_influence=state.get("tournament_influence", {}) or {},
        last_result=last,
        config=cfg,
        clock=SimpleNamespace(now=(state.get("now"))),
    )
    return adapter
