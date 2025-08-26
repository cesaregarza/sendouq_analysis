#!/usr/bin/env python3
"""
Find and optionally scrape missing, completed tournaments week-by-week.

Workflow
- Uses the public API calendar by ISO week to list tournaments.
- Filters out tournaments already present in data_dir.
- Checks completion via /api/tournament/{id} (isFinalized == true).
- Scrapes full tournament JSON for those missing and finalized.

Requires SENDOU_KEY in env or .env at repo root.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from typing import List, Set

from rankings.scraping.batch import scrape_tournament_batch
from rankings.scraping.calendar_api import (
    fetch_tournament_metadata,
    is_tournament_finalized,
    list_weekly_tournaments,
)
from rankings.scraping.missing import get_existing_tournament_ids


def configure_logging(verbose: bool) -> None:
    """Configure logging at INFO level, suppressing noisy library debugs.

    Even if verbose is True, we keep the root logger at INFO to avoid
    third-party DEBUG spam (e.g., urllib3 connection logs). If detailed debug
    is needed in the future, target module-specific loggers instead.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Quiet noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    # Allow our own module to be more verbose only if desired, but default INFO
    logging.getLogger("weekly_missing_scraper").setLevel(logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Discover missing tournaments week-by-week and scrape completed ones."
        )
    )
    parser.add_argument(
        "--weeks-back",
        type=int,
        default=12,
        help="How many ISO weeks back to scan (default: 12)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/tournaments",
        help=(
            "Directory used to detect existing tournaments (default: data/tournaments)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to write scraped data (default: same as --data-dir). "
            "Use a new subfolder to avoid overwriting existing batch files."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Scrape batch size (default: 50)",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.2,
        help="Delay between metadata requests to be polite (default: 0.2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report missing finalized tournaments; do not scrape",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()
    configure_logging(args.verbose)
    logger = logging.getLogger("weekly_missing_scraper")

    # Identify existing tournaments on disk
    existing: Set[int] = get_existing_tournament_ids(args.data_dir)
    logger.info(
        "Found %d existing tournaments in %s", len(existing), args.data_dir
    )

    # Discover tournaments week-by-week
    weekly = list_weekly_tournaments(date.today(), weeks_back=args.weeks_back)
    discovered_ids: List[int] = [w.tournament_id for w in weekly]
    unique_ids = sorted(set(discovered_ids))
    logger.info(
        "Discovered %d tournaments across %d weeks (unique: %d)",
        len(discovered_ids),
        args.weeks_back,
        len(unique_ids),
    )

    # Filter to those not yet present
    missing: List[int] = [tid for tid in unique_ids if tid not in existing]
    if not missing:
        logger.info("No missing tournaments found in the scanned weeks.")
        return

    logger.info("%d tournaments are not in the local set", len(missing))

    # Check which are finalized
    finalized_missing: List[int] = []
    for tid in missing:
        try:
            meta = fetch_tournament_metadata(tid)
            if is_tournament_finalized(meta):
                finalized_missing.append(tid)
        except Exception as e:
            logger.warning("Failed to fetch metadata for %s: %s", tid, e)
        if args.delay_seconds:
            import time

            time.sleep(args.delay_seconds)

    if not finalized_missing:
        logger.info("No missing tournaments are finalized yet.")
        return

    logger.info(
        "%d missing tournaments appear finalized: %s",
        len(finalized_missing),
        finalized_missing[:20],
    )

    # Either report or scrape
    if args.dry_run:
        logger.info("Dry run mode: skipping scrape.")
        return

    output_dir = args.output_dir or args.data_dir
    if output_dir != args.data_dir:
        logger.info("Writing scraped data to %s", output_dir)

    res = scrape_tournament_batch(
        tournament_ids=finalized_missing,
        output_dir=output_dir,
        batch_size=args.batch_size,
    )
    logger.info(
        "Scrape complete. scraped=%s failed=%s failed_ids_head=%s",
        res.get("scraped"),
        res.get("failed"),
        (res.get("failed_ids") or [])[:10],
    )


if __name__ == "__main__":
    main()
