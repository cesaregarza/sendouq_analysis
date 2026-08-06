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
    SENDOU_BRACKETS_DATA_SUFFIX,
    SENDOU_DATA_SUFFIX,
    SENDOU_DATA_SUFFIX_LEGACY,
)
from rankings.scraping.calendar_api import fetch_tournament_teams
from rankings.scraping.turbo_stream import (
    extract_tournament_bracket_data,
    extract_tournament_data,
)

BRACKET_DATA_KEYS = ("stage", "group", "round", "match")


def build_tournament_url(tournament_id: int, legacy: bool = False) -> str:
    """Build the API URL for a specific tournament ID."""
    suffix = SENDOU_DATA_SUFFIX_LEGACY if legacy else SENDOU_DATA_SUFFIX
    return f"{SENDOU_BASE_URL}{tournament_id}{suffix}"


def build_tournament_bracket_url(tournament_id: int, bracket_idx: int) -> str:
    """Build the bracket-scoped data URL introduced by Sendou in August 2026."""
    return (
        f"{SENDOU_BASE_URL}{tournament_id}{SENDOU_BRACKETS_DATA_SUFFIX}"
        f"?idx={bracket_idx}"
    )


def _normalize_layout_context(tournament_data: dict) -> None:
    """Add aliases expected by the parser for fields renamed in slim layouts."""
    tournament = tournament_data.get("tournament", {})
    ctx = tournament.get("ctx", {})
    if "startTime" not in ctx and "startsAt" in ctx:
        ctx["startTime"] = ctx["startsAt"]


def _reconstruct_bracket_data(
    tournament_id: int,
    tournament_data: dict,
    session: requests.Session,
    timeout: float,
) -> dict:
    """Hydrate a slim tournament layout with every bracket's match data."""
    tournament = tournament_data.get("tournament")
    if not isinstance(tournament, dict):
        raise ValueError("tournament layout is missing the tournament object")

    _normalize_layout_context(tournament_data)
    existing_data = tournament.get("data")
    if isinstance(existing_data, dict) and existing_data:
        return tournament_data

    brackets_meta = tournament.get("bracketsMeta")
    if not isinstance(brackets_meta, list) or not brackets_meta:
        raise ValueError(
            "tournament layout has no match data or bracket metadata"
        )

    merged = {key: [] for key in BRACKET_DATA_KEYS}
    for bracket_idx in range(len(brackets_meta)):
        url = build_tournament_bracket_url(tournament_id, bracket_idx)
        response = session.get(url, timeout=timeout)
        response.raise_for_status()

        route_data = extract_tournament_bracket_data(response.json())
        actual_idx = route_data.get("bracketIdx") if route_data else None
        if actual_idx != bracket_idx:
            raise ValueError(
                f"bracket response index {actual_idx!r} did not match "
                f"requested index {bracket_idx}"
            )
        bracket = route_data.get("bracket") if route_data else None
        bracket_data = (
            bracket.get("data") if isinstance(bracket, dict) else None
        )
        if not isinstance(bracket_data, dict):
            raise ValueError(
                f"bracket {bracket_idx} response is missing bracket.data"
            )

        for key in BRACKET_DATA_KEYS:
            records = bracket_data.get(key)
            if not isinstance(records, list):
                raise ValueError(
                    f"bracket {bracket_idx} response has invalid {key} data"
                )
            merged[key].extend(records)

    tournament["data"] = merged
    logger.info(
        "Reconstructed tournament %s from %s bracket payloads (%s matches)",
        tournament_id,
        len(brackets_meta),
        len(merged["match"]),
    )
    return tournament_data


def _try_turbo_stream_endpoint(
    tournament_id: int,
    session: requests.Session,
    timeout: float,
) -> dict | None:
    """Try fetching from the new turbo-stream .data endpoint."""
    url = build_tournament_url(tournament_id, legacy=False)
    response = session.get(url, timeout=timeout)
    response.raise_for_status()

    tournament_data = extract_tournament_data(response.json())
    if not tournament_data:
        raise ValueError("response did not contain a tournament layout")

    tournament_data = _reconstruct_bracket_data(
        tournament_id, tournament_data, session, timeout
    )
    if not validate_tournament_data(tournament_data):
        raise ValueError("tournament layout is missing id, name, or match data")

    logger.debug(
        "Successfully fetched tournament %s via turbo-stream endpoint",
        tournament_id,
    )
    return tournament_data


def _try_legacy_endpoint(
    tournament_id: int,
    session: requests.Session,
    timeout: float,
) -> dict | None:
    """Try fetching from the legacy ?_data endpoint (plain JSON)."""
    url = build_tournament_url(tournament_id, legacy=True)
    response = session.get(url, timeout=timeout)
    response.raise_for_status()

    data = response.json()
    _normalize_layout_context(data)
    if not data or not validate_tournament_data(data):
        raise ValueError(
            "legacy response did not contain complete tournament data"
        )

    logger.debug(
        "Successfully fetched tournament %s via legacy endpoint", tournament_id
    )
    return data


def scrape_tournament(
    tournament_id: int,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    session: requests.Session | None = None,
) -> dict:
    """Scrape tournament data from Sendou.ink API with turbo-stream and legacy fallback.

    The scraping order is:
    1. Try turbo-stream endpoint (new format as of late 2025)
    2. Fall back to legacy ?_data= endpoint
    3. If teams are missing from the result, fetch from /api/tournament/{id}/teams
    """
    if session is None:
        session = requests.Session()

    last_errors: list[str] = []

    for attempt in range(max_retries):
        last_errors = []
        try:
            result = _try_turbo_stream_endpoint(tournament_id, session, timeout)
            return _enrich_with_teams_api(result, tournament_id)
        except Exception as exc:
            last_errors.append(f"turbo-stream: {exc}")
            logger.debug(
                "Turbo-stream endpoint failed for tournament %s",
                tournament_id,
                exc_info=True,
            )

        try:
            result = _try_legacy_endpoint(tournament_id, session, timeout)
            return _enrich_with_teams_api(result, tournament_id)
        except Exception as exc:
            last_errors.append(f"legacy: {exc}")
            logger.debug(
                "Legacy endpoint failed for tournament %s",
                tournament_id,
                exc_info=True,
            )

        # Both failed, retry with backoff
        if attempt < max_retries - 1:
            wait_time = backoff_factor**attempt
            logger.warning(
                f"Attempt {attempt + 1} failed for tournament {tournament_id}, "
                f"retrying in {wait_time:.1f}s ({'; '.join(last_errors)})"
            )
            time.sleep(wait_time)

    raise RuntimeError(
        f"Failed to fetch tournament {tournament_id} after {max_retries} attempts "
        f"(tried both turbo-stream and legacy endpoints; "
        f"last errors: {'; '.join(last_errors)})"
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


def _enrich_with_teams_api(tournament_data: dict, tournament_id: int) -> dict:
    """Enrich tournament data with teams from the public API if missing.

    This is a fallback for when turbo-stream or legacy scraping doesn't
    return teams data. The public API endpoint /api/tournament/{id}/teams
    returns team rosters in a slightly different format that we normalize.
    """
    try:
        tournament = tournament_data.get("tournament", {})
        ctx = tournament.get("ctx", {})
        existing_teams = ctx.get("teams", []) or []

        has_full_rosters = bool(existing_teams) and all(
            isinstance(team, dict) and isinstance(team.get("members"), list)
            for team in existing_teams
        )
        if has_full_rosters:
            return tournament_data

        logger.info(
            "Teams missing full rosters for tournament %s, fetching from API",
            tournament_id,
        )
        api_teams = fetch_tournament_teams(tournament_id)

        if not api_teams:
            raise ValueError(
                f"teams API returned no teams for tournament {tournament_id}"
            )

        # Normalize API format to match expected ctx.teams structure
        existing_by_id = {
            team.get("id"): team
            for team in existing_teams
            if isinstance(team, dict) and team.get("id") is not None
        }
        normalized_teams = []
        for team in api_teams:
            normalized = {
                "id": team.get("id"),
                "name": team.get("name"),
                "seed": team.get("seed"),
                "prefersNotToHost": 1 if team.get("prefersNotToHost") else 0,
                "droppedOut": 0,
                "inviteCode": None,
                "createdAt": None,
                "activeRosterUserIds": None,
                "startingBracketIdx": None,
                "pickupAvatarUrl": None,
                "members": [],
                "checkIns": [],
                "mapPool": [],
                "team": None,
                "avgSeedingSkillOrdinal": None,
            }
            normalized.update(existing_by_id.get(team.get("id"), {}))
            normalized["members"] = []

            # Normalize members
            for member in team.get("members", []):
                normalized["members"].append(
                    {
                        "userId": member.get("userId"),
                        "username": member.get("name"),
                        "discordId": member.get("discordId"),
                        "discordAvatar": member.get("avatarUrl", "").split("/")[
                            -1
                        ]
                        if member.get("avatarUrl")
                        else None,
                        "customUrl": None,
                        "country": member.get("country"),
                        "twitch": None,
                        "plusTier": None,
                        "isOwner": 1 if member.get("captain") else 0,
                        "createdAt": None,
                        "inGameName": member.get("inGameName"),
                    }
                )

            # Extract seeding skill if available
            seeding = team.get("seedingPower", {})
            if seeding:
                # Use ranked skill as primary
                normalized["avgSeedingSkillOrdinal"] = seeding.get("ranked")

            # Extract team logo info
            if team.get("teamPageUrl"):
                normalized["team"] = {
                    "customUrl": team.get("teamPageUrl", "").split("/")[-1]
                    if team.get("teamPageUrl")
                    else None,
                    "logoUrl": team.get("logoUrl"),
                    "deletedAt": None,
                }

            normalized_teams.append(normalized)

        normalized_ids = {team.get("id") for team in normalized_teams}
        normalized_teams.extend(
            team
            for team in existing_teams
            if team.get("id") not in normalized_ids
        )

        # Update the tournament data with fetched teams
        tournament_data["tournament"]["ctx"]["teams"] = normalized_teams
        logger.info(
            "Enriched tournament %s with %s teams from API",
            tournament_id,
            len(normalized_teams),
        )

    except Exception as exc:
        raise RuntimeError(
            f"failed to enrich team rosters for tournament {tournament_id}: {exc}"
        ) from exc

    return tournament_data


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
