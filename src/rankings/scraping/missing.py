"""
Functions for identifying and scraping missing tournaments.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple

from rankings.scraping.api import scrape_tournament
from rankings.scraping.discovery import get_latest_tournament_id

logger = logging.getLogger(__name__)


def get_existing_tournament_ids(data_dir: str = "data/tournaments") -> Set[int]:
    """
    Get list of existing tournament IDs from files.

    Parameters
    ----------
    data_dir : str
        Directory containing tournament data files

    Returns
    -------
    Set[int]
        Set of existing tournament IDs
    """
    existing_ids = set()

    if not os.path.exists(data_dir):
        return existing_ids

    # Check individual tournament files
    for file in os.listdir(data_dir):
        if file.startswith("tournament_") and file.endswith(".json"):
            match = re.search(r"tournament_(\d+)\.json", file)
            if match:
                existing_ids.add(int(match.group(1)))

    # Also check the scraped file
    scraped_file = os.path.join(data_dir, "tournaments_20250814_scraped.json")
    if os.path.exists(scraped_file):
        try:
            with open(scraped_file, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for tournament in data:
                        if "id" in tournament:
                            existing_ids.add(tournament["id"])
                elif isinstance(data, dict):
                    for tid in data.keys():
                        if tid.isdigit():
                            existing_ids.add(int(tid))
        except Exception as e:
            logger.warning(f"Error reading scraped file: {e}")

    # Check continuous scraping files
    for file in os.listdir(data_dir):
        if file.startswith("tournaments_continuous_") and file.endswith(
            ".json"
        ):
            try:
                with open(os.path.join(data_dir, file), "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for tournament in data:
                            if "id" in tournament:
                                existing_ids.add(tournament["id"])
            except Exception as e:
                logger.warning(f"Error reading continuous file {file}: {e}")

    return existing_ids


def find_missing_tournament_ids(
    start_id: Optional[int] = None,
    end_id: Optional[int] = None,
    data_dir: str = "data/tournaments",
) -> List[int]:
    """
    Find missing tournament IDs in a range.

    Parameters
    ----------
    start_id : Optional[int]
        Starting tournament ID (defaults to max existing + 1)
    end_id : Optional[int]
        Ending tournament ID (defaults to latest from sendou.ink)
    data_dir : str
        Directory containing tournament data files

    Returns
    -------
    List[int]
        List of missing tournament IDs
    """
    existing_ids = get_existing_tournament_ids(data_dir)

    if existing_ids:
        max_existing = max(existing_ids)
        logger.info(
            f"Found {len(existing_ids)} existing tournaments (max ID: {max_existing})"
        )
    else:
        max_existing = 0
        logger.info("No existing tournaments found")

    # Get the latest tournament ID if not specified
    if end_id is None:
        logger.info("Getting latest tournament ID from sendou.ink...")
        end_id = get_latest_tournament_id()
        logger.info(f"Latest tournament ID on sendou.ink: {end_id}")

    # Set start ID if not specified
    if start_id is None:
        start_id = max_existing + 1

    # Find missing IDs in range
    missing_ids = []
    for tid in range(start_id, end_id + 1):
        if tid not in existing_ids:
            missing_ids.append(tid)

    logger.info(
        f"Found {len(missing_ids)} missing tournaments in range {start_id}-{end_id}"
    )

    return missing_ids


def scrape_missing_tournaments(
    start_id: Optional[int] = None,
    end_id: Optional[int] = None,
    max_tournaments: int = 50,
    data_dir: str = "data/tournaments",
    delay: float = 1.0,
    verbose: bool = True,
) -> Tuple[int, List[Tuple[int, str]]]:
    """
    Scrape missing tournaments from sendou.ink.

    Parameters
    ----------
    start_id : Optional[int]
        Starting tournament ID (defaults to max existing + 1)
    end_id : Optional[int]
        Ending tournament ID (defaults to latest from sendou.ink)
    max_tournaments : int
        Maximum number of tournaments to scrape (default: 50)
    data_dir : str
        Directory to save tournament data files
    delay : float
        Delay in seconds between requests (default: 1.0)
    verbose : bool
        Whether to print progress messages

    Returns
    -------
    Tuple[int, List[Tuple[int, str]]]
        Number of successfully scraped tournaments and list of errors (id, error_message)
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Find missing IDs
    missing_ids = find_missing_tournament_ids(start_id, end_id, data_dir)

    # Limit the number to scrape
    if len(missing_ids) > max_tournaments:
        if verbose:
            logger.info(f"Limiting to {max_tournaments} tournaments")
        missing_ids = missing_ids[:max_tournaments]

    if not missing_ids:
        if verbose:
            logger.info("No tournaments to scrape!")
        return 0, []

    if verbose:
        logger.info(
            f"Will scrape tournament IDs: {missing_ids[:10]}{'...' if len(missing_ids) > 10 else ''}"
        )

    # Scrape tournaments
    scraped_count = 0
    errors = []

    for i, tid in enumerate(missing_ids):
        if verbose:
            logger.info(
                f"\n[{i+1}/{len(missing_ids)}] Scraping tournament {tid}..."
            )

        try:
            # Use the scrape_tournament function
            tournament_data = scrape_tournament(tid)

            if tournament_data:
                # Save to individual file
                output_file = data_path / f"tournament_{tid}.json"
                with open(output_file, "w") as f:
                    json.dump(tournament_data, f, indent=2)

                if verbose:
                    logger.info(f"  ✓ Saved to {output_file}")
                scraped_count += 1

                # Check if it's ranked
                if (
                    verbose
                    and "data" in tournament_data
                    and "tournament" in tournament_data["data"]
                ):
                    tournament = tournament_data["data"]["tournament"]
                    settings = tournament.get("settings", {})
                    is_ranked = settings.get("isRanked", False)
                    name = tournament.get("name", "Unknown")
                    logger.info(
                        f"  → {'RANKED' if is_ranked else 'Not ranked'}: {name[:60]}"
                    )
            else:
                if verbose:
                    logger.info(f"  ✗ No data returned")
                errors.append((tid, "No data returned"))

        except Exception as e:
            error_msg = str(e)
            if verbose:
                # Don't print full error for 404s (tournament doesn't exist)
                if "404" in error_msg:
                    logger.info(f"  ✗ Tournament does not exist (404)")
                else:
                    logger.info(f"  ✗ Error: {error_msg}")
            errors.append((tid, error_msg))

        # Rate limiting - be nice to the server
        if i < len(missing_ids) - 1:
            time.sleep(delay)

    # Summary
    if verbose:
        logger.info(f"\n{'='*60}")
        logger.info(f"Scraping complete!")
        logger.info(
            f"  Successfully scraped: {scraped_count}/{len(missing_ids)} tournaments"
        )

        if errors:
            # Count 404 errors separately
            not_found = [e for e in errors if "404" in e[1]]
            other_errors = [e for e in errors if "404" not in e[1]]

            if not_found:
                logger.info(f"  Not found (404): {len(not_found)} tournaments")

            if other_errors:
                logger.info(f"\nOther errors ({len(other_errors)}):")
                for tid, error in other_errors[:10]:
                    logger.info(f"  Tournament {tid}: {error}")
                if len(other_errors) > 10:
                    logger.info(f"  ... and {len(other_errors) - 10} more")

    return scraped_count, errors
