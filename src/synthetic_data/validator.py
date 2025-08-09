"""Compatibility shim: expose DataValidator at synthetic_data.validator.

Prefer importing from synthetic_data.io.validator directly in new code.
"""

from .io.validator import DataValidator  # re-export

__all__ = ["DataValidator"]
