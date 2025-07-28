"""
Core API functions for scraping tournament data from Sendou.ink.

This module contains the fundamental HTTP operations and data validation
for interacting with the Sendou.ink tournament API.
"""

from __future__ import annotations

import json
import re
import time
from typing import Dict, Optional
from urllib.parse import urlparse

import requests

from rankings.core.constants import (
    DEFAULT_BACKOFF_FACTOR,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    SENDOU_BASE_URL,
    SENDOU_DATA_SUFFIX,
)


def build_tournament_url(tournament_id: int) -> str:
    """Build the API URL for a specific tournament ID."""
    return f"{SENDOU_BASE_URL}{tournament_id}{SENDOU_DATA_SUFFIX}"


def scrape_tournament(
    tournament_id: int,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    session: Optional[requests.Session] = None,
) -> Dict:
    """
    Scrape tournament data from Sendou.ink API.

    Parameters
    ----------
    tournament_id : int
        The tournament ID to scrape
    timeout : float, optional
        Request timeout in seconds
    max_retries : int, optional
        Maximum number of retry attempts
    backoff_factor : float, optional
        Exponential backoff factor for retries
    session : requests.Session, optional
        Reusable session for efficient multiple requests

    Returns
    -------
    dict
        Tournament data from the API

    Raises
    ------
    RuntimeError
        If the tournament cannot be fetched after max_retries
    """
    url = build_tournament_url(tournament_id)

    if session is None:
        session = requests.Session()

    for attempt in range(max_retries):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt < max_retries - 1:
                wait_time = backoff_factor**attempt
                print(
                    f"Attempt {attempt + 1} failed for tournament {tournament_id}, "
                    f"retrying in {wait_time:.1f}s: {e}"
                )
                time.sleep(wait_time)
            else:
                raise RuntimeError(
                    f"Failed to fetch tournament {tournament_id} after {max_retries} attempts"
                ) from e


def validate_tournament_data(tournament_data: Dict) -> bool:
    """
    Validate that tournament data has the expected structure.

    Parameters
    ----------
    tournament_data : dict
        Tournament data to validate

    Returns
    -------
    bool
        True if data appears valid
    """
    try:
        tournament = tournament_data.get("tournament", {})
        ctx = tournament.get("ctx", {})
        data = tournament.get("data", {})

        # Basic validation checks
        has_id = "id" in ctx
        has_name = "name" in ctx
        has_structure = bool(data)

        return has_id and has_name and has_structure

    except (AttributeError, TypeError):
        return False


def extract_tournament_id_from_url(url: str) -> Optional[int]:
    """
    Extract tournament ID from a Sendou.ink URL.

    Parameters
    ----------
    url : str
        URL containing tournament ID

    Returns
    -------
    int or None
        Extracted tournament ID, or None if not found
    """
    # Handle various URL formats
    patterns = [
        r"/to/(\d+)",  # Standard format
        r"tournament[_=](\d+)",  # Alternative formats
        r"id[_=](\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return int(match.group(1))

    return None
