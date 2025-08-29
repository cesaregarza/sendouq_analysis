"""
Calendar + public API helpers for week-by-week tournament discovery.

This module uses the public API described in plan.md:
  - GET /api/calendar/{year}/{week}
  - GET /api/tournament/{id}

Auth is via bearer token SENDOU_KEY in the environment (or .env fallback).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import requests

from rankings.core.constants import DEFAULT_TIMEOUT, SENDOU_PUBLIC_API_BASE_URL


def _load_sendou_key_from_env_or_file() -> Optional[str]:
    """Return SENDOU_KEY from the environment, or try reading a local .env.

    This avoids adding a dependency on python-dotenv while keeping local
    development convenient. The .env parser here is intentionally minimal.
    """
    key = os.environ.get("SENDOU_KEY")
    if key:
        return key

    env_path = os.path.join(os.getcwd(), ".env")
    if not os.path.exists(env_path):
        return None

    try:
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k == "SENDOU_KEY" and v:
                    return v
    except Exception:
        return None
    return None


def _auth_headers() -> Dict[str, str]:
    key = _load_sendou_key_from_env_or_file()
    if not key:
        raise RuntimeError(
            "SENDOU_KEY not set. Set it in environment or .env file."
        )
    return {"Authorization": f"Bearer {key}"}


def fetch_calendar_week(
    year: int, week: int, *, timeout: float = DEFAULT_TIMEOUT
) -> List[dict]:
    """Fetch public calendar events for a given ISO week and filter to tournaments.

    Returns a list of event dicts where `tournamentId` is not null.
    """
    url = f"{SENDOU_PUBLIC_API_BASE_URL}/calendar/{year}/{week}"
    res = requests.get(url, headers=_auth_headers(), timeout=timeout)
    res.raise_for_status()
    events = res.json()
    return [e for e in events if e.get("tournamentId") is not None]


def iter_weeks_back(
    start: date, *, max_weeks: int
) -> Iterable[Tuple[int, int]]:
    """Yield (iso_year, iso_week) pairs going back week by week (inclusive)."""
    seen: set[Tuple[int, int]] = set()
    current = start
    for _ in range(max_weeks):
        y, w, _ = current.isocalendar()
        key = (y, w)
        if key not in seen:
            seen.add(key)
            yield key
        current = current - timedelta(days=7)


def fetch_tournament_metadata(
    tournament_id: int, *, timeout: float = DEFAULT_TIMEOUT
) -> dict:
    """Fetch public tournament metadata by ID (includes `isFinalized`)."""
    url = f"{SENDOU_PUBLIC_API_BASE_URL}/tournament/{tournament_id}"
    res = requests.get(url, headers=_auth_headers(), timeout=timeout)
    res.raise_for_status()
    return res.json()


def fetch_tournament_players(
    tournament_id: int, *, timeout: float = DEFAULT_TIMEOUT
) -> dict:
    """Fetch public tournament players route by ID.

    Returns the raw JSON from GET /api/tournament/{id}/players, which contains
    data about which matches players actually played in. Authentication uses
    the same SENDOU_KEY bearer token as other public API calls.
    """
    url = f"{SENDOU_PUBLIC_API_BASE_URL}/tournament/{tournament_id}/players"
    res = requests.get(url, headers=_auth_headers(), timeout=timeout)
    res.raise_for_status()
    return res.json()


def is_tournament_finalized(meta: dict) -> bool:
    """Return True if the public metadata marks the tournament as finalized."""
    return bool(meta.get("isFinalized", False))


@dataclass(frozen=True)
class WeeklyTournament:
    year: int
    week: int
    tournament_id: int
    name: Optional[str]
    start_time: Optional[str]


def list_weekly_tournaments(
    start: date, *, weeks_back: int
) -> List[WeeklyTournament]:
    """List tournaments discovered from the calendar for the last N weeks."""
    out: List[WeeklyTournament] = []
    for y, w in iter_weeks_back(start, max_weeks=weeks_back):
        events = fetch_calendar_week(y, w)
        for e in events:
            tid = e.get("tournamentId")
            out.append(
                WeeklyTournament(
                    year=y,
                    week=w,
                    tournament_id=int(tid),
                    name=e.get("name"),
                    start_time=e.get("startTime"),
                )
            )
    return out
