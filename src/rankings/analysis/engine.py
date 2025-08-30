"""Compatibility shim for the engine module.

This file maintains backward compatibility for code that imports from
rankings.analysis.engine directly. New code should import from the
engine submodule instead.

DEPRECATED: This file will be removed in a future version.
Please update imports to use:
    from rankings.analysis.engine import RatingEngine, make_participation_inverse_teleport
"""

from __future__ import annotations

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from rankings.analysis.engine is deprecated. "
    "Please import from rankings.analysis.engine module instead:\n"
    "  from rankings.analysis.engine import RatingEngine, make_participation_inverse_teleport",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the new module structure for compatibility
from rankings.algorithms.compat import RatingEngine
from rankings.analysis.engine.teleport import (
    make_participation_inverse_teleport,
)

__all__ = [
    "RatingEngine",
    "make_participation_inverse_teleport",
]
