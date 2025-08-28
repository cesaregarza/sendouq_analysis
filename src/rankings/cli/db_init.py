from __future__ import annotations

"""
Initialize the rankings database schema and tables.

Usage:
  poetry run rankings_db_init --schema comp_rankings --sslmode require
  # or with explicit URL
  poetry run rankings_db_init --db-url postgresql://user:pass@host/db?sslmode=require --schema comp_rankings

This should be run with an admin user that has CREATE privileges. Normal app
users (insert/update/delete) can be used afterwards for import/update tasks.
"""

import argparse
import os


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create rankings schema and tables (idempotent)"
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=os.getenv("RANKINGS_DATABASE_URL") or os.getenv("DATABASE_URL"),
        help="Database URL (overrides env)",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default=os.getenv("RANKINGS_DB_SCHEMA", "comp_rankings"),
        help="Target schema (rankings or comp_rankings)",
    )
    parser.add_argument(
        "--sslmode",
        type=str,
        choices=[
            "disable",
            "allow",
            "prefer",
            "require",
            "verify-ca",
            "verify-full",
        ],
        default=None,
        help="Set libpq sslmode in the URL",
    )

    args = parser.parse_args(argv)

    # Ensure the chosen schema is used by rankings.sql.constants
    os.environ["RANKINGS_DB_SCHEMA"] = args.schema

    # Lazy import after setting env so constants pick it up
    from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

    from rankings.sql import create_all as rankings_create_all
    from rankings.sql import create_engine as rankings_create_engine

    db_url = args.db_url
    if db_url and args.sslmode:
        parts = urlparse(db_url)
        q = dict(parse_qsl(parts.query, keep_blank_values=True))
        q["sslmode"] = args.sslmode
        parts = parts._replace(query=urlencode(q))
        db_url = urlunparse(parts)

    if not db_url and args.sslmode:
        os.environ["RANKINGS_DB_SSLMODE"] = args.sslmode
    engine = rankings_create_engine(db_url)
    rankings_create_all(engine)
    print(f"Initialized schema '{args.schema}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
