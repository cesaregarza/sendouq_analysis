"""Compatibility shim: expose MatchSimulator at synthetic_data.match_simulator.

Prefer importing from synthetic_data.core.match_simulator directly in new code.
"""

from .core.match_simulator import MatchSimulator  # re-export

__all__ = ["MatchSimulator"]
