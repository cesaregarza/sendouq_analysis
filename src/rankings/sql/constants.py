from __future__ import annotations

import os


def _default_schema() -> str:
    schema = (
        os.getenv("RANKINGS_DB_SCHEMA", "comp_rankings").strip()
        or "comp_rankings"
    )
    # Only allow either "rankings" or "comp_rankings" to avoid typos
    if schema not in {"rankings", "comp_rankings"}:
        # Fall back safely
        return "comp_rankings"
    return schema


# Database schema used for rankings tables
SCHEMA: str = _default_schema()
