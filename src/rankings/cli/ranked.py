from __future__ import annotations

import argparse
import logging
from datetime import date
from typing import Iterable, List, Set, Tuple

from rankings.scraping.batch import scrape_tournament_batch
from rankings.scraping.calendar_api import (
    fetch_calendar_week,
    fetch_tournament_metadata,
    is_tournament_finalized,
    iter_weeks_back,
)
from rankings.scraping.missing import get_existing_tournament_ids


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def _discover_tournaments_for_weeks(
    weeks: Iterable[Tuple[int, int]]
) -> List[int]:
    ids: List[int] = []
    seen: Set[int] = set()
    for y, w in weeks:
        events = fetch_calendar_week(y, w)
        for e in events:
            tid = e.get("tournamentId")
            if isinstance(tid, int) and tid not in seen:
                seen.add(tid)
                ids.append(tid)
    return ids


def _filter_finalized(
    ids: List[int], *, delay_seconds: float = 0.2
) -> List[int]:
    import time

    out: List[int] = []
    log = logging.getLogger("rankings.cli.ranked")
    for tid in ids:
        try:
            meta = fetch_tournament_metadata(tid)
            if is_tournament_finalized(meta):
                out.append(tid)
        except Exception as e:
            log.warning("metadata fetch failed for %s: %s", tid, e)
        if delay_seconds:
            time.sleep(delay_seconds)
    return out


def main() -> int:
    """
    Ranked ingest step (scraping phase only).

    High-level (future):
      1) Load which tournaments exist (DB)
      2) Check current + previous ISO week for tournaments
      3) For finalized ones not yet present, scrape and save (DB)
      4) Recompute graph, persist results, and pickle model (S3)

    This CLI implements steps 2–3 with file-based existing detection for now.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Discover tournaments over recent ISO weeks, filter finalized, and scrape."
        )
    )
    parser.add_argument(
        "--weeks-back",
        type=int,
        default=2,
        help="How many ISO weeks back to scan (default: 2)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/tournaments",
        help="Directory used to detect existing tournaments (recursive)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to write scraped JSON (default: same as --data-dir). "
            "Use a dated subfolder to avoid overwriting batches."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip scraping IDs already present under --data-dir",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50, help="Scrape batch size"
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.2,
        help="Delay between metadata calls (politeness)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report finalized IDs; do not scrape",
    )

    args = parser.parse_args()
    _configure_logging()
    logger = logging.getLogger("rankings.cli.ranked")

    # Determine ISO weeks to scan: today and going back weeks_back-1 more
    weeks = list(
        iter_weeks_back(date.today(), max_weeks=max(args.weeks_back, 1))
    )
    logger.info("Scanning ISO weeks: %s", weeks)

    # Discover tournaments across the weeks
    ids = _discover_tournaments_for_weeks(weeks)
    if not ids:
        logger.info("No tournaments discovered across %d weeks", len(weeks))
        return 0
    logger.info("Discovered %d tournaments (unique)", len(ids))

    # Filter finalized via public metadata
    finalized = _filter_finalized(ids, delay_seconds=args.delay_seconds)
    if not finalized:
        logger.info(
            "No finalized tournaments found across %d weeks", len(weeks)
        )
        return 0
    logger.info(
        "%d tournaments are finalized: head=%s", len(finalized), finalized[:15]
    )

    to_scrape = finalized
    if args.skip_existing:
        existing = get_existing_tournament_ids(args.data_dir)
        before = len(to_scrape)
        to_scrape = [tid for tid in to_scrape if tid not in existing]
        logger.info(
            "Skip-existing enabled: %d -> %d to scrape (existing=%d)",
            before,
            len(to_scrape),
            len(existing),
        )
        if not to_scrape:
            logger.info("Nothing to scrape after skipping existing IDs.")
            return 0

    if args.dry_run:
        logger.info(
            "Dry run: would scrape %d IDs: %s", len(to_scrape), to_scrape[:25]
        )
        return 0

    output_dir = args.output_dir or args.data_dir
    logger.info(
        "Scraping %d tournaments to %s (batch=%d)",
        len(to_scrape),
        output_dir,
        args.batch_size,
    )
    res = scrape_tournament_batch(
        tournament_ids=to_scrape,
        output_dir=output_dir,
        batch_size=args.batch_size,
    )
    logger.info(
        "Scrape result: scraped=%s failed=%s failed_ids_head=%s",
        res.get("scraped"),
        res.get("failed"),
        (res.get("failed_ids") or [])[:10],
    )
    return 0
