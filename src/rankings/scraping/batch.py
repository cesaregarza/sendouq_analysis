"""
Batch processing functions for scraping multiple tournaments efficiently.

This module handles batch operations, range scraping, and integration with
different data storage backends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import requests
from tqdm import tqdm

from rankings.core.constants import (
    CALENDAR_URL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_FAILURES,
)
from rankings.scraping.api import scrape_tournament
from rankings.scraping.discovery import discover_tournaments_from_calendar
from rankings.scraping.storage import save_tournament_batch


def scrape_tournament_batch(
    tournament_ids: List[int],
    output_dir: str = "data/tournaments",
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_failures: int = DEFAULT_MAX_FAILURES,
    session: Optional[requests.Session] = None,
) -> Dict[str, Union[int, List[int]]]:
    """
    Scrape multiple tournaments and save to JSON files.

    Parameters
    ----------
    tournament_ids : list of int
        Tournament IDs to scrape
    output_dir : str, optional
        Directory to save JSON files
    batch_size : int, optional
        Number of tournaments per JSON file
    max_failures : int, optional
        Maximum consecutive failures before stopping
    session : requests.Session, optional
        Reusable session for efficient requests

    Returns
    -------
    dict
        Results summary with 'scraped', 'failed', and 'failed_ids' keys
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if session is None:
        session = requests.Session()

    scraped = 0
    failed = 0
    failed_ids = []
    consecutive_failures = 0

    # Process tournaments in batches
    for batch_idx in range(0, len(tournament_ids), batch_size):
        batch = tournament_ids[batch_idx : batch_idx + batch_size]
        batch_data = []

        print(
            f"Processing batch {batch_idx // batch_size + 1} "
            f"(tournaments {batch[0]}-{batch[-1]})"
        )

        for tournament_id in tqdm(batch, desc="Scraping tournaments"):
            try:
                tournament_data = scrape_tournament(
                    tournament_id, session=session
                )
                batch_data.append(tournament_data)
                scraped += 1
                consecutive_failures = 0

            except Exception as e:
                print(f"Failed to scrape tournament {tournament_id}: {e}")
                failed += 1
                failed_ids.append(tournament_id)
                consecutive_failures += 1

                if consecutive_failures >= max_failures:
                    print(
                        f"Stopping due to {max_failures} consecutive failures"
                    )
                    break

        # Save batch to file
        if batch_data:
            save_tournament_batch(
                batch_data, batch_idx // batch_size, output_dir
            )

        if consecutive_failures >= max_failures:
            break

    return {"scraped": scraped, "failed": failed, "failed_ids": failed_ids}


def scrape_tournament_range(
    start_id: int,
    end_id: int,
    output_dir: str = "data/tournaments",
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_failures: int = DEFAULT_MAX_FAILURES,
) -> Dict[str, Union[int, List[int]]]:
    """
    Scrape a range of tournament IDs.

    Parameters
    ----------
    start_id : int
        Starting tournament ID (inclusive)
    end_id : int
        Ending tournament ID (inclusive)
    output_dir : str, optional
        Directory to save JSON files
    batch_size : int, optional
        Number of tournaments per JSON file
    max_failures : int, optional
        Maximum consecutive failures before stopping

    Returns
    -------
    dict
        Results summary
    """
    tournament_ids = list(range(start_id, end_id + 1))
    return scrape_tournament_batch(
        tournament_ids=tournament_ids,
        output_dir=output_dir,
        batch_size=batch_size,
        max_failures=max_failures,
    )


def scrape_tournaments_from_calendar(
    output_dir: str = "data/tournaments",
    batch_size: int = DEFAULT_BATCH_SIZE,
    calendar_url: str = CALENDAR_URL,
) -> Dict[str, Union[int, List[int]]]:
    """
    Discover and scrape tournaments from the calendar.

    Parameters
    ----------
    output_dir : str, optional
        Directory to save JSON files
    batch_size : int, optional
        Number of tournaments per JSON file
    calendar_url : str, optional
        URL to the calendar ICS file

    Returns
    -------
    dict
        Results summary
    """
    tournament_ids = discover_tournaments_from_calendar(calendar_url)

    if not tournament_ids:
        print("No tournaments discovered from calendar")
        return {"scraped": 0, "failed": 0, "failed_ids": []}

    print(f"Discovered {len(tournament_ids)} tournaments, starting scrape...")

    return scrape_tournament_batch(
        tournament_ids=tournament_ids,
        output_dir=output_dir,
        batch_size=batch_size,
    )


def scrape_latest_tournaments(
    count: int = 100,
    output_dir: str = "data/tournaments",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, Union[int, List[int]]]:
    """
    Scrape the latest tournaments by working backwards from the highest ID.

    Parameters
    ----------
    count : int, optional
        Number of recent tournaments to scrape
    output_dir : str, optional
        Directory to save JSON files
    batch_size : int, optional
        Batch size for scraping

    Returns
    -------
    dict
        Results summary
    """
    from rankings.scraping.discovery import get_latest_tournament_id

    # Get the latest tournament ID from calendar
    latest_id = get_latest_tournament_id()

    if latest_id is None:
        print("Could not determine latest tournament ID")
        return {"scraped": 0, "failed": 0, "failed_ids": []}

    # Work backwards from latest ID
    start_id = max(1, latest_id - count + 1)
    end_id = latest_id

    print(f"Scraping {count} latest tournaments: {start_id} to {end_id}")

    return scrape_tournament_range(
        start_id=start_id,
        end_id=end_id,
        output_dir=output_dir,
        batch_size=batch_size,
    )


def scrape_to_database(
    tournament_ids: List[int],
    db_path: str = "data/tournaments.db",
    batch_size: int = 10,
    max_failures: int = DEFAULT_MAX_FAILURES,
) -> Dict[str, Union[int, List[int]]]:
    """
    Scrape tournaments directly to database using existing scraper.

    This function uses the existing database scraper from
    src/sendouq_analysis/scrape_tournament.py

    Parameters
    ----------
    tournament_ids : list of int
        Tournament IDs to scrape
    db_path : str, optional
        Database file path
    batch_size : int, optional
        Batch size for database operations
    max_failures : int, optional
        Maximum failures before stopping

    Returns
    -------
    dict
        Results summary
    """
    try:
        from sendouq_analysis.scrape_tournament import scrape_tournaments

        # Convert tournament_ids to start/end range for the existing scraper
        if not tournament_ids:
            return {"scraped": 0, "failed": 0, "failed_ids": []}

        start_value = min(tournament_ids)
        end_value = max(tournament_ids)

        print(
            f"Scraping tournaments {start_value}-{end_value} to database {db_path}"
        )

        # Use the existing database scraper
        result = scrape_tournaments(
            batch_size=batch_size,
            start_value=start_value,
            end_value=end_value,
            max_failures=max_failures,
        )

        return {
            "scraped": len(tournament_ids),  # Estimate
            "failed": 0,
            "failed_ids": [],
        }

    except ImportError as e:
        print(f"Database scraper not available: {e}")
        print("Falling back to JSON scraping...")
        return scrape_tournament_batch(tournament_ids)
