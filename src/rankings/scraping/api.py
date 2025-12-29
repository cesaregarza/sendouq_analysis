"""
Core API functions for scraping tournament data from Sendou.ink.

This module contains the fundamental HTTP operations and data validation
for interacting with the Sendou.ink tournament API.

Note: As of late 2025, Sendou.ink uses React Router's Single Fetch feature,
which returns data in turbo-stream format instead of plain JSON. This module
supports both formats with automatic fallback for resilience.
"""

from __future__ import annotations

import json
import logging
import re
import time
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

from rankings.core.constants import (
    DEFAULT_BACKOFF_FACTOR,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    SENDOU_BASE_URL,
    SENDOU_DATA_SUFFIX,
    SENDOU_DATA_SUFFIX_LEGACY,
)
from rankings.scraping.turbo_stream import extract_tournament_data


def build_tournament_url(tournament_id: int, legacy: bool = False) -> str:
    """Build the API URL for a specific tournament ID."""
    suffix = SENDOU_DATA_SUFFIX_LEGACY if legacy else SENDOU_DATA_SUFFIX
    return f"{SENDOU_BASE_URL}{tournament_id}{suffix}"


def _try_turbo_stream_endpoint(
    tournament_id: int,
    session: requests.Session,
    timeout: float,
) -> dict | None:
    """Try fetching from the new turbo-stream .data endpoint."""
    url = build_tournament_url(tournament_id, legacy=False)
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()

        raw_data = response.json()
        tournament_data = extract_tournament_data(raw_data)

        if tournament_data and validate_tournament_data(tournament_data):
            logger.debug(f"Successfully fetched tournament {tournament_id} via turbo-stream endpoint")
            return tournament_data
    except Exception as e:
        logger.debug(f"Turbo-stream endpoint failed for tournament {tournament_id}: {e}")

    return None


def _try_legacy_endpoint(
    tournament_id: int,
    session: requests.Session,
    timeout: float,
) -> dict | None:
    """Try fetching from the legacy ?_data endpoint (plain JSON)."""
    url = build_tournament_url(tournament_id, legacy=True)
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()

        data = response.json()

        if data and validate_tournament_data(data):
            logger.debug(f"Successfully fetched tournament {tournament_id} via legacy endpoint")
            return data
    except Exception as e:
        logger.debug(f"Legacy endpoint failed for tournament {tournament_id}: {e}")

    return None


def scrape_tournament(
    tournament_id: int,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    session: requests.Session | None = None,
) -> dict:
    """Scrape tournament data from Sendou.ink API with turbo-stream and legacy fallback."""
    if session is None:
        session = requests.Session()

    last_error = None

    for attempt in range(max_retries):
        # Try turbo-stream endpoint first (new format)
        result = _try_turbo_stream_endpoint(tournament_id, session, timeout)
        if result is not None:
            return result

        # Fall back to legacy endpoint
        result = _try_legacy_endpoint(tournament_id, session, timeout)
        if result is not None:
            return result

        # Both failed, retry with backoff
        if attempt < max_retries - 1:
            wait_time = backoff_factor ** attempt
            logger.warning(
                f"Attempt {attempt + 1} failed for tournament {tournament_id}, "
                f"retrying in {wait_time:.1f}s"
            )
            time.sleep(wait_time)

    raise RuntimeError(
        f"Failed to fetch tournament {tournament_id} after {max_retries} attempts "
        f"(tried both turbo-stream and legacy endpoints)"
    )


def validate_tournament_data(tournament_data: dict) -> bool:
    """Validate that tournament data has the expected structure."""
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


def extract_tournament_id_from_url(url: str) -> int | None:
    """Extract tournament ID from a Sendou.ink URL."""
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
