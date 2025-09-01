"""
Data storage and management functions for tournament data.

This module handles loading, saving, and managing scraped tournament data
across different storage formats and locations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import polars as pl
from tqdm import tqdm

logger = logging.getLogger(__name__)


def save_tournament_batch(
    batch_data: List[Dict], batch_idx: int, output_dir: str
) -> None:
    """
    Save a batch of tournament data to a JSON file.

    Parameters
    ----------
    batch_data : list of dict
        Tournament data to save
    batch_idx : int
        Batch index for filename
    output_dir : str
        Output directory path
    """
    if not batch_data:
        return

    batch_filename = f"tournament_{batch_idx}.json"
    batch_path = Path(output_dir) / batch_filename

    with open(batch_path, "w") as f:
        json.dump(batch_data, f, indent=2)

    logger.info(f"Saved {len(batch_data)} tournaments to {batch_filename}")


def load_scraped_tournaments(data_dir: str = "data/tournaments") -> List[Dict]:
    """
    Load all scraped tournament data from JSON files, recursively.

    - Recurses into subdirectories of `data_dir`.
    - Includes per-ID files (e.g., `tournament_2090.json`).
    - Includes batch files (e.g., `tournament_0.json`, lists of tournaments).
    - Includes snapshot files (e.g., `tournaments_*.json` and
      `tournaments_continuous_*.json`).

    Parameters
    ----------
    data_dir : str, optional
        Directory containing tournament JSON files (root; searched recursively)

    Returns
    -------
    list of dict
        Combined tournament data from all files
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning(f"Data directory {data_dir} does not exist")
        return []

    all_tournaments: List[Dict] = []

    # Recursively gather files by pattern
    per_id_files = list(data_path.rglob("tournament_*.json"))
    snapshot_files = list(data_path.rglob("tournaments_*.json"))
    continuous_files = list(data_path.rglob("tournaments_continuous_*.json"))

    # Avoid double counting continuous files in snapshot_files if patterns overlap
    snapshot_only = [
        p
        for p in snapshot_files
        if p.name.startswith("tournaments_")
        and not p.name.startswith("tournaments_continuous_")
    ]

    json_files = sorted(set(per_id_files + snapshot_only + continuous_files))

    if not json_files:
        logger.warning(
            f"No tournament JSON files found under {data_dir} (recursive)"
        )
        return []

    logger.info(
        "Loading %d JSON files recursively (%d per-id/batch, %d snapshots, %d continuous)"
        % (
            len(json_files),
            len(per_id_files),
            len(snapshot_only),
            len(continuous_files),
        )
    )

    for json_file in tqdm(json_files, desc="Loading files"):
        try:
            with open(json_file, "r") as f:
                payload = json.load(f)
            if isinstance(payload, list):
                all_tournaments.extend(payload)
            else:
                all_tournaments.append(payload)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load %s: %s", json_file, e)

    logger.info(f"Loaded {len(all_tournaments)} tournaments total")
    return all_tournaments


def get_tournament_summary(data_dir: str = "data/tournaments") -> pl.DataFrame:
    """
    Get a summary of scraped tournaments using the enhanced parser.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing tournament JSON files

    Returns
    -------
    pl.DataFrame
        Summary with comprehensive tournament metadata including organization,
        settings, staff, and match/team counts
    """
    from rankings.core.parser import parse_tournaments_data

    tournaments = load_scraped_tournaments(data_dir)

    if not tournaments:
        return pl.DataFrame([])

    # Use the enhanced parser to get full tournament metadata
    tables = parse_tournaments_data(tournaments)
    tournament_df = tables.get("tournaments")

    if tournament_df is None:
        return pl.DataFrame([])

    # Return the comprehensive tournament metadata
    return tournament_df


def _coerce_int(v):
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def extract_appearances_from_players_payload(
    tournament_id: int, payload: dict | list
) -> list[dict]:
    """Best-effort extraction of per-match player appearances from the players route payload.

    Supported shapes:
    - Team-based per-match listing (legacy):
      { "matches": [ { "matchId": 123, "teams": [ { "teamId": 1, "players": [{"userId": 10}, ...] }, ... ] }, ... ] }
      or [ { "matchId": 123, "teams": [ ... ] }, ... ]

    - Player-based listing (new):
      [ { "userId": 10, "matchIds": [1,2,3] }, ... ]

    Returns list of rows: {tournament_id, match_id, user_id, team_id?}
    team_id will be None for the new player-based format and can be derived later.
    """
    rows: list[dict] = []

    def _emit(match_id, team_id, players_node):
        if players_node is None:
            return
        # Accept list[int] or list[dict]
        if isinstance(players_node, list):
            for p in players_node:
                if isinstance(p, dict):
                    uid = _coerce_int(p.get("userId") or p.get("id"))
                else:
                    uid = _coerce_int(p)
                if uid is not None and match_id is not None:
                    rows.append(
                        {
                            "tournament_id": int(tournament_id),
                            "match_id": _coerce_int(match_id),
                            "team_id": _coerce_int(team_id),
                            "user_id": uid,
                        }
                    )
        elif isinstance(players_node, dict):
            # Sometimes may be { "userIds": [...] }
            user_ids = players_node.get("userIds") or players_node.get(
                "players"
            )
            if isinstance(user_ids, list):
                _emit(match_id, team_id, user_ids)

    def _parse_match_obj(m: dict):
        mid = m.get("matchId") or m.get("id")
        teams = m.get("teams") or []
        if isinstance(teams, list):
            for t in teams:
                if not isinstance(t, dict):
                    continue
                tid = t.get("teamId") or t.get("id")
                # players can be under 'players' as list[dict] or 'userIds'
                plist = (
                    t.get("players")
                    or t.get("userIds")
                    or t.get("users")
                    or t.get("roster")
                )
                _emit(mid, tid, plist)

    # Unwrap common container keys
    node = payload
    for key in ("data", "payload", "result"):
        if isinstance(node, dict) and key in node:
            node = node[key]

    # Case 1: legacy team-based format under {matches: [...]}
    if isinstance(node, dict) and isinstance(node.get("matches"), list):
        for m in node.get("matches"):
            if isinstance(m, dict):
                _parse_match_obj(m)

    # Case 2: direct list
    elif isinstance(node, list):
        # Subcase 2a: legacy list of matches (each with teams)
        if all(isinstance(x, dict) for x in node) and any(
            ("matchId" in x or "teams" in x) for x in node
        ):
            for m in node:
                _parse_match_obj(m)
        # Subcase 2b: new player-based format
        else:
            for item in node:
                if not isinstance(item, dict):
                    continue
                uid = _coerce_int(
                    item.get("userId") or item.get("user_id") or item.get("id")
                )
                match_ids = (
                    item.get("matchIds")
                    or item.get("match_ids")
                    or item.get("matches")
                )
                if uid is None or not isinstance(match_ids, list):
                    continue
                for mid in match_ids:
                    mid_i = _coerce_int(mid)
                    if mid_i is None:
                        continue
                    rows.append(
                        {
                            "tournament_id": int(tournament_id),
                            "match_id": mid_i,
                            "team_id": None,  # Derived later
                            "user_id": uid,
                        }
                    )

    # Keep only rows that have the minimal keys
    return [
        r
        for r in rows
        if r.get("match_id") is not None and r.get("user_id") is not None
    ]


# Backwards-compat alias (was private helper)
_extract_appearances_from_players_payload = (
    extract_appearances_from_players_payload
)


def load_match_appearances(data_dir: str = "data/tournaments") -> pl.DataFrame:
    """Load per-match player appearances from scraped JSON if present.

    Returns a DataFrame with columns: tournament_id, match_id, team_id, user_id
    or an empty DataFrame if none found.
    """
    tournaments = load_scraped_tournaments(data_dir)
    if not tournaments:
        return pl.DataFrame([])

    rows: list[dict] = []
    for entry in tournaments:
        try:
            t = entry.get("tournament", {})
            ctx = t.get("ctx", {}) if isinstance(t, dict) else {}
            tid = ctx.get("id")
            if tid is None:
                continue
            pm = entry.get("player_matches")
            if not pm:
                continue
            rows.extend(extract_appearances_from_players_payload(int(tid), pm))
        except Exception as e:
            logger.warning("Failed to parse players payload: %s", e)
            continue

    if not rows:
        return pl.DataFrame([])

    df = pl.DataFrame(rows)
    # Cast to proper types and deduplicate
    cast_map = {
        "tournament_id": pl.col("tournament_id").cast(pl.Int64, strict=False),
        "match_id": pl.col("match_id").cast(pl.Int64, strict=False),
        "team_id": pl.col("team_id").cast(pl.Int64, strict=False),
        "user_id": pl.col("user_id").cast(pl.Int64, strict=False),
    }
    df = df.with_columns(list(cast_map.values())).unique(
        subset=["tournament_id", "match_id", "user_id"]
    )
    return df
