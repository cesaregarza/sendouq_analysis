"""
Tournament discovery functions for finding tournaments via calendar and other methods.

This module handles discovering tournament IDs from external sources like the
Sendou.ink calendar system.
"""

from __future__ import annotations

import re
from typing import List, Optional

import requests

from rankings.core.constants import CALENDAR_URL, DEFAULT_TIMEOUT


def discover_tournaments_from_calendar(
    calendar_url: str = CALENDAR_URL,
) -> List[int]:
    """
    Discover tournament IDs from the Sendou.ink calendar.

    Parameters
    ----------
    calendar_url : str, optional
        URL to the calendar ICS file

    Returns
    -------
    list of int
        List of discovered tournament IDs
    """
    try:
        response = requests.get(calendar_url, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        calendar_content = response.text

        # Extract tournament URLs from calendar content
        # Pattern: https://sendou.ink//to/2041 (note double slash)
        pattern = r"https://sendou\.ink//to/(\d+)"
        matches = re.findall(pattern, calendar_content)

        tournament_ids = [int(match) for match in matches]

        if tournament_ids:
            max_id = max(tournament_ids)
            print(f"Discovered {len(tournament_ids)} tournaments from calendar")
            print(f"Highest tournament ID: {max_id}")
        else:
            print("No tournament IDs found in calendar")

        return sorted(set(tournament_ids))  # Remove duplicates and sort

    except requests.RequestException as e:
        print(f"Failed to fetch calendar: {e}")
        return []


def get_latest_tournament_id(calendar_url: str = CALENDAR_URL) -> Optional[int]:
    """
    Get the latest tournament ID from the calendar.

    Parameters
    ----------
    calendar_url : str, optional
        URL to the calendar ICS file

    Returns
    -------
    int or None
        Latest tournament ID, or None if discovery fails
    """
    tournament_ids = discover_tournaments_from_calendar(calendar_url)
    return max(tournament_ids) if tournament_ids else None
