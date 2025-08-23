"""Protocol definitions for pluggable ranking components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import polars as pl

if TYPE_CHECKING:
    from typing import Any


@runtime_checkable
class RatingBackend(Protocol):
    """Protocol for rating computation backends.

    This allows tick-tock orchestration to use different rating algorithms
    (e.g., classic PageRank, exposure log-odds) while maintaining the same
    tournament influence update mechanism.
    """

    def compute(
        self,
        matches: pl.DataFrame,
        players: pl.DataFrame | None,
        active_ids: list[Any],
        tournament_influence: dict[int, float],
        **kwargs: Any,
    ) -> pl.DataFrame:
        """Compute ratings given current tournament influences.

        Args:
            matches: Match data.
            players: Optional player/roster data.
            active_ids: List of active player/team IDs.
            tournament_influence: Current tournament influence scores.
            **kwargs: Additional backend-specific parameters.

        Returns:
            DataFrame with columns: id (player/team identifier), score (rating score),
            quality_mass (bounded quality measure for influence update), and optionally
            win_pr, loss_pr, exposure, lambda_used.
        """
        ...
