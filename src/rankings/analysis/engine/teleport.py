"""Teleport vector utilities for PageRank calculations."""

from __future__ import annotations

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
    base = players.select(["user_id", "tournament_id", "event_ts"]).unique()

    decayed = (
        base.with_columns(
            (
                ((now_ts - pl.col("event_ts").cast(pl.Float64)) / 86400.0)
                .mul(-decay_rate)
                .exp()
            ).alias("w")
        )
        .group_by("user_id")
        .agg(pl.col("w").sum().alias("decayed_events"))
    )

    volume_vector = (epsilon + decayed["decayed_events"]).pow(-gamma)
    volume_vector = volume_vector / volume_vector.sum()

    user_ids = decayed["user_id"].to_list()
    num_users = len(user_ids)
    uniform_vector = np.ones(num_users) / num_users
    teleport_vector = (
        1 - eta
    ) * uniform_vector + eta * volume_vector.to_numpy()

    return dict(zip(user_ids, teleport_vector.tolist()))
