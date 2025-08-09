"""
Teleport vector utilities for PageRank calculations.

This module provides functions for creating custom teleport vectors
used in PageRank-based ranking algorithms.
"""

import numpy as np
import polars as pl


def make_participation_inverse_teleport(
    players: pl.DataFrame,
    now_ts: int,
    decay_rate: float,
    eta: float = 0.3,
    gamma: float = 0.7,
    epsilon: float = 1.0,
) -> dict[int, float]:
    """Create participation-inverse teleport vector for PageRank.

    This function creates a teleport vector that inversely weights players
    based on their tournament participation, helping to reduce the advantage
    of players who participate in many tournaments.

    Parameters
    ----------
    players : pl.DataFrame
        DataFrame with columns [user_id, tournament_id, event_ts]
    now_ts : int
        Current timestamp for decay calculation
    decay_rate : float
        Decay rate for time weighting (typically ln(2) / half_life_days)
    eta : float, default=0.3
        Weight for participation-inverse component (0=uniform, 1=pure inverse)
    gamma : float, default=0.7
        Exponent for participation scaling (higher = stronger inverse effect)
    epsilon : float, default=1.0
        Smoothing parameter for zero participation

    Returns
    -------
    dict[int, float]
        Dictionary mapping user_id to teleport probability

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> players_df = pl.DataFrame({
    ...     "user_id": [1, 1, 2, 2, 3],
    ...     "tournament_id": [1, 2, 1, 3, 1],
    ...     "event_ts": [1704067200] * 5
    ... })
    >>> now = datetime.now()
    >>> teleport = make_participation_inverse_teleport(
    ...     players_df,
    ...     now_ts=int(now.timestamp()),
    ...     decay_rate=0.00577,  # 120-day half-life
    ...     eta=0.3,
    ...     gamma=0.7
    ... )
    >>> len(teleport)
    3
    """
    # One row per (user_id, tournament_id) with representative event_ts
    base = players.select(["user_id", "tournament_id", "event_ts"]).unique()

    # Calculate decayed event participation for each user
    decayed = (
        base.with_columns(
            (
                ((now_ts - pl.col("event_ts").cast(pl.Float64)) / 86400.0)
                .mul(-decay_rate)
                .exp()
            ).alias("w")
        )
        .group_by("user_id")
        .agg(pl.col("w").sum().alias("dec_evts"))
    )

    # Compute participation-inverse scores: 1 / (epsilon + decayed_events)^gamma
    v_vol = (epsilon + decayed["dec_evts"]).pow(-gamma)
    v_vol = v_vol / v_vol.sum()

    # Get user IDs and create mixed teleport vector
    uids = decayed["user_id"].to_list()
    n = len(uids)
    v_uniform = np.ones(n) / n
    v = (1 - eta) * v_uniform + eta * v_vol.to_numpy()

    return dict(zip(uids, v.tolist()))
