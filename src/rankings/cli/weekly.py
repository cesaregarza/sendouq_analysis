from __future__ import annotations

import argparse
import logging
import os
from datetime import date
from typing import List, Set, Tuple

from rankings.scraping.batch import scrape_tournament_batch
from rankings.scraping.calendar_api import (
    fetch_calendar_week,
    fetch_tournament_metadata,
    is_tournament_finalized,
)
from rankings.scraping.missing import get_existing_tournament_ids


def _init_sentry() -> None:
    dsn = os.getenv("SENTRY_DSN") or os.getenv("RANKINGS_SENTRY_DSN")
    if not dsn:
        return
    try:
        import sentry_sdk  # type: ignore
        from sentry_sdk.integrations.logging import (
            LoggingIntegration,  # type: ignore
        )
    except Exception:
        return
    env = os.getenv("SENTRY_ENV") or os.getenv("ENV") or "development"
    try:
        traces = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.0"))
    except Exception:
        traces = 0.0
    try:
        profiles = float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "0.0"))
    except Exception:
        profiles = 0.0
    logging_integration = LoggingIntegration(
        level=logging.INFO, event_level=logging.ERROR
    )
    try:
        sentry_sdk.init(
            dsn=dsn,
            environment=env,
            integrations=[logging_integration],
            traces_sample_rate=traces,
            profiles_sample_rate=profiles,
        )
        sentry_sdk.set_tag("service", "weekly_cli")
    except Exception:
        pass


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def _current_iso_year_week(today: date | None = None) -> Tuple[int, int]:
    d = today or date.today()
    y, w, _ = d.isocalendar()
    return int(y), int(w)


def _discover_week_tournament_ids(year: int, week: int) -> List[int]:
    events = fetch_calendar_week(year, week)
    ids: List[int] = []
    seen: Set[int] = set()
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
    for tid in ids:
        try:
            meta = fetch_tournament_metadata(tid)
            if is_tournament_finalized(meta):
                out.append(tid)
        except Exception as e:
            logging.getLogger("rankings.cli.weekly").warning(
                "metadata fetch failed for %s: %s", tid, e
            )
        if delay_seconds:
            time.sleep(delay_seconds)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Discover this ISO week's tournaments, filter finalized, and scrape."
        )
    )
    parser.add_argument(
        "--year", type=int, default=None, help="ISO year override"
    )
    parser.add_argument(
        "--week", type=int, default=None, help="ISO week override"
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
    _init_sentry()
    _configure_logging()
    logger = logging.getLogger("rankings.cli.weekly")

    y, w = (
        (args.year, args.week)
        if (args.year and args.week)
        else _current_iso_year_week()
    )
    logger.info("Target ISO week: %s/%s", y, w)

    ids = _discover_week_tournament_ids(y, w)
    if not ids:
        logger.info("No tournaments discovered for %s/%s", y, w)
        return 0
    logger.info("Discovered %d tournaments for %s/%s", len(ids), y, w)

    finalized = _filter_finalized(ids, delay_seconds=args.delay_seconds)
    if not finalized:
        logger.info("No finalized tournaments found for %s/%s", y, w)
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
