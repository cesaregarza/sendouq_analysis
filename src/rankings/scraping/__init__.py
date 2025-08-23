"""Tournament scraping module for Sendou.ink data."""

from __future__ import annotations

# Core API functions
from rankings.scraping.api import (
    build_tournament_url,
    extract_tournament_id_from_url,
    scrape_tournament,
    validate_tournament_data,
)

# Batch processing functions
from rankings.scraping.batch import (
    scrape_latest_tournaments,
    scrape_to_database,
    scrape_tournament_batch,
    scrape_tournament_range,
    scrape_tournaments_from_calendar,
)

# Discovery functions
from rankings.scraping.discovery import (
    discover_tournaments_from_calendar,
    get_latest_tournament_id,
)

# Missing tournament utilities
from rankings.scraping.missing import (
    find_missing_tournament_ids,
    get_existing_tournament_ids,
    scrape_missing_tournaments,
)

# Storage and data management
from rankings.scraping.storage import (
    get_tournament_summary,
    load_scraped_tournaments,
)

__all__ = [
    # API
    "scrape_tournament",
    "build_tournament_url",
    "validate_tournament_data",
    "extract_tournament_id_from_url",
    # Discovery
    "discover_tournaments_from_calendar",
    "get_latest_tournament_id",
    # Batch processing
    "scrape_tournament_batch",
    "scrape_tournament_range",
    "scrape_latest_tournaments",
    "scrape_tournaments_from_calendar",
    "scrape_to_database",
    # Storage
    "load_scraped_tournaments",
    "get_tournament_summary",
    # Missing tournaments
    "get_existing_tournament_ids",
    "find_missing_tournament_ids",
    "scrape_missing_tournaments",
]
