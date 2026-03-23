"""Utilities for scraping Sendou Plus voting results."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl
import requests

from rankings.scraping.turbo_stream import decode_turbo_stream

PLUS_VOTING_RESULTS_URL = "https://sendou.ink/plus/voting/results.data"
PLUS_ROUTE_KEY = "features/plus-voting/routes/plus.voting.results"
PLUS_VOTING_COLUMNS = [
    "id",
    "username",
    "custom_url",
    "pass_or_fail",
    "tier",
    "suggested",
]


def fetch_plus_voting_tier_results(
    url: str = PLUS_VOTING_RESULTS_URL,
    *,
    timeout: int = 30,
    route_key: str = PLUS_ROUTE_KEY,
    session: requests.Session | None = None,
) -> list[dict]:
    """Fetch and decode tiered Plus-voting results from the .data endpoint."""
    if session is None:
        session = requests.Session()

    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("Unexpected payload type: expected JSON list")

    decoded = decode_turbo_stream(payload)
    if route_key not in decoded:
        raise ValueError(f"Plus route key missing in payload: {route_key}")

    route_data = decoded[route_key]
    if not isinstance(route_data, dict) or "data" not in route_data:
        raise ValueError("Unexpected route payload shape: missing data field")

    data = route_data["data"]
    if not isinstance(data, dict):
        raise ValueError("Unexpected route data shape: expected object")

    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError("Unexpected results payload: expected results list")
    return results


def extract_plus_voting_rows(
    results: Sequence[dict],
) -> tuple[list[dict], dict[str, int]]:
    """Normalize tiered plus-voting route data into flat rows."""
    rows: list[dict] = []
    seen: set[tuple[int, str, int]] = set()
    counts = {"pass": 0, "fail": 0}

    for tier_entry in results:
        if not isinstance(tier_entry, dict):
            continue
        tier = tier_entry.get("tier")
        if tier is None:
            continue
        try:
            tier_int = int(tier)
        except (TypeError, ValueError):
            continue

        for pass_or_fail in ("passed", "failed"):
            users = tier_entry.get(pass_or_fail, [])
            if not isinstance(users, list):
                continue

            label = "pass" if pass_or_fail == "passed" else "fail"
            for user in users:
                if not isinstance(user, dict):
                    continue
                user_id = user.get("id")
                try:
                    user_id_int = int(user_id)
                except (TypeError, ValueError):
                    continue

                key = (user_id_int, label, tier_int)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "id": user_id_int,
                        "username": user.get("username"),
                        "custom_url": user.get("customUrl"),
                        "pass_or_fail": label,
                        "tier": tier_int,
                        "suggested": 1 if user.get("wasSuggested") else 0,
                    }
                )
                counts[label] += 1

    rows.sort(key=lambda row: (row["tier"], row["pass_or_fail"], row["id"]))
    return rows, counts


def plus_voting_rows_to_dataframe(rows: Sequence[dict]) -> pl.DataFrame:
    """Convert normalized voting rows to a canonical DataFrame shape."""
    if not rows:
        return pl.DataFrame(schema={col: pl.Null for col in PLUS_VOTING_COLUMNS})
    df = pl.DataFrame(rows)
    return df.select(PLUS_VOTING_COLUMNS)


def scrape_plus_voting_dataframe(
    url: str = PLUS_VOTING_RESULTS_URL,
    *,
    timeout: int = 30,
    route_key: str = PLUS_ROUTE_KEY,
    session: requests.Session | None = None,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Fetch + decode + normalize plus voting rows into a DataFrame."""
    results = fetch_plus_voting_tier_results(
        url=url,
        timeout=timeout,
        route_key=route_key,
        session=session,
    )
    rows, counts = extract_plus_voting_rows(results)
    return plus_voting_rows_to_dataframe(rows), counts


def write_plus_voting_csv(df: pl.DataFrame, output_path: Path) -> None:
    """Write canonical plus voting DataFrame to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if df.is_empty():
        empty = pl.DataFrame(
            {
                "id": [],
                "username": [],
                "custom_url": [],
                "pass_or_fail": [],
                "tier": [],
                "suggested": [],
            }
        )
        empty.write_csv(str(output_path))
        return
    df.select(PLUS_VOTING_COLUMNS).write_csv(str(output_path))

