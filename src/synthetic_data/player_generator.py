"""Compatibility shim: expose PlayerGenerator at synthetic_data.player_generator.

Prefer importing from synthetic_data.core.player_generator directly in new code.
"""

from .core.player_generator import PlayerGenerator  # re-export

__all__ = ["PlayerGenerator"]
