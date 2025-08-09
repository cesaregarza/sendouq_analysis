"""Compatibility shim: expose DataSerializer at synthetic_data.data_serializer.

Prefer importing from synthetic_data.io.data_serializer directly in new code.
"""

from .io.data_serializer import DataSerializer  # re-export

__all__ = ["DataSerializer"]
