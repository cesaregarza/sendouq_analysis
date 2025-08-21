"""Protocol definitions for pluggable ranking components."""

from typing import Dict, List, Optional, Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class RatingBackend(Protocol):
    """
    Protocol for rating computation backends.

    This allows tick-tock orchestration to use different rating algorithms
    (e.g., classic PageRank, exposure log-odds) while maintaining the same
    tournament influence update mechanism.
    """

    def compute(
        self,
        matches: pl.DataFrame,
        players: Optional[pl.DataFrame],
        active_ids: List,
        tournament_influence: Dict[int, float],
        **kwargs,
    ) -> pl.DataFrame:
        """
        Compute ratings given current tournament influences.

        Args:
            matches: Match data
            players: Optional player/roster data
            active_ids: List of active player/team IDs
            tournament_influence: Current tournament influence scores (S)
            **kwargs: Additional backend-specific parameters

        Returns:
            DataFrame with columns:
            - id: Player/team identifier
            - score: Rating score
            - quality_mass: Bounded quality measure for influence update (p_i)
            - (optional) win_pr: Win PageRank
            - (optional) loss_pr: Loss PageRank
            - (optional) exposure: Exposure values
            - (optional) lambda_used: Lambda smoothing parameter used
        """
        ...
