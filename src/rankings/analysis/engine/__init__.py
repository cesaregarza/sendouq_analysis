"""
Unified ranking system module for tournament-based rating calculations.

This module provides a unified ranking system with two modes:
- Exposure Log-Odds Mode: Volume-bias resistant rankings using dual PageRank
- Tick-Tock Mode: Traditional PageRank with tournament strength modeling

Use the build_ranking_engine() factory function for the unified entry point.
"""

from typing import Union, Dict, Any

from .core import RatingEngine
from .exposure_logodds import ExposureLogOddsEngine
from .teleport import make_participation_inverse_teleport


def build_ranking_engine(mode: str = "exposure_logodds", **kwargs) -> Union[RatingEngine, ExposureLogOddsEngine]:
    """
    Factory function to create a ranking engine with the specified mode.
    
    This function provides a unified entry point for the ranking system, allowing
    you to select between different ranking modes while maintaining a consistent
    interface.
    
    Parameters
    ----------
    mode : str, default="exposure_logodds"
        The ranking mode to use. Options:
        - "exposure_logodds": Exposure Log-Odds mode (recommended, volume-bias resistant)
        - "tick_tock": Tick-Tock mode (traditional PageRank with tournament strength modeling)
    **kwargs
        Additional keyword arguments passed to the selected engine constructor.
        
    Returns
    -------
    Union[RatingEngine, ExposureLogOddsEngine]
        The configured ranking engine instance.
        
    Raises
    ------
    ValueError
        If the specified mode is not supported.
        
    Examples
    --------
    >>> # Recommended: Exposure Log-Odds mode
    >>> engine = build_ranking_engine(mode="exposure_logodds", beta=1.0)
    >>> rankings = engine.rank_players(matches_df, players_df)
    
    >>> # Alternative: Tick-Tock mode
    >>> engine = build_ranking_engine(mode="tick_tock", beta=1.0, max_tick_tock=5)
    >>> rankings = engine.rank_players(matches_df, players_df)
    """
    if mode == "exposure_logodds":
        return ExposureLogOddsEngine(**kwargs)
    elif mode == "tick_tock":
        return RatingEngine(**kwargs)
    else:
        raise ValueError(f"Unknown ranking mode: {mode}. Supported modes: 'exposure_logodds', 'tick_tock'")


__all__ = [
    "RatingEngine",
    "ExposureLogOddsEngine", 
    "build_ranking_engine",
    "make_participation_inverse_teleport",
]
