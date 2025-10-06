from __future__ import annotations

import argparse
import logging
import sys

import requests
from sqlalchemy import text

from rankings.cli.repull import _scrape_with_players, import_with_rollback
from rankings.sql import create_all as rankings_create_all
from rankings.sql import create_engine as rankings_create_engine
from rankings.sql.constants import SCHEMA as RANKINGS_SCHEMA

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-scrape tournaments that have zero roster entries recorded in the "
            "rankings database."
        )
    )
    parser.add_argument(
        "--db-url",
        dest="db_url",
        default=None,
        help=(
            "Override database URL. Defaults to env RANKINGS_DATABASE_URL or "
            "DATABASE_URL"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of tournaments to re-pull.",
    )
    parser.add_argument(
        "--include-unfinalized",
        action="store_true",
        help="Also re-pull tournaments that are not marked finalized.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level for stdout output",
    )
    return parser.parse_args(argv)


def _find_tournaments_without_players(
    engine, *, include_unfinalized: bool, limit: int | None
) -> list[int]:
    where_clause = (
        ""
        if include_unfinalized
        else "WHERE COALESCE(t.is_finalized, false) = true"
    )
    sql = f"""
        SELECT t.tournament_id
        FROM {RANKINGS_SCHEMA}.tournaments AS t
        LEFT JOIN {RANKINGS_SCHEMA}.roster_entries AS r
            ON r.tournament_id = t.tournament_id
        {where_clause}
        GROUP BY t.tournament_id
        HAVING COUNT(r.player_id) = 0
        ORDER BY t.start_time_ms DESC NULLS LAST
    """
    if limit and limit > 0:
        sql += " LIMIT :limit"
    params: dict[str, int] = {}
    if limit and limit > 0:
        params["limit"] = int(limit)

    with engine.connect() as conn:
        result = conn.execute(text(sql), params)
        return [int(row[0]) for row in result]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    engine = rankings_create_engine(args.db_url)
    rankings_create_all(engine)

    target_ids = _find_tournaments_without_players(
        engine,
        include_unfinalized=args.include_unfinalized,
        limit=args.limit,
    )

    if not target_ids:
        logger.info(
            "No tournaments found without roster entries; nothing to do."
        )
        return 0

    logger.info(
        "Found %d tournament(s) without roster entries: %s",
        len(target_ids),
        target_ids,
    )

    session = requests.Session()
    payloads = []
    for tid in target_ids:
        try:
            payloads.append(_scrape_with_players(tid, session=session))
            logger.info("Fetched tournament %s", tid)
        except requests.RequestException as exc:
            logger.error("Failed to scrape tournament %s: %s", tid, exc)
            return 1

    try:
        inserted = import_with_rollback(
            engine, payloads, target_ids, log=logger
        )
    except Exception:
        logger.exception(
            "Import failed for tournaments %s; aborting fix run", target_ids
        )
        return 1

    logger.info(
        "Re-imported %d tournament(s); import attempted %d row groups",
        len(payloads),
        inserted,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
