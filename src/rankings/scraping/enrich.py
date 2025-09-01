from __future__ import annotations

"""
Best-effort enrichment for appearance team assignment via match API.

This module provides a helper that, for appearance rows that failed roster-based
team inference, calls the public match endpoint `/api/tournament-match/{id}` to
recover the two team IDs and the set of participating users, then assigns
team_id to the unmatched users using roster-majority heuristics. The goal is to
minimize API usage by fetching only for matches that actually need enrichment.
"""

from typing import Iterable, Optional

import polars as pl
import requests

from rankings.core.constants import SENDOU_PUBLIC_API_BASE_URL
from rankings.scraping.calendar_api import _auth_headers


def _fetch_match_info(
    match_id: int,
) -> tuple[int | None, int | None, list[int]]:
    """Return (team_one_id, team_two_id, participants) for a match.

    Participants is the union of user IDs from `mapList[*].participatedUserIds`.
    If the route is unavailable or malformed, returns (None, None, []).
    """
    try:
        url = f"{SENDOU_PUBLIC_API_BASE_URL}/tournament-match/{match_id}"
        res = requests.get(url, headers=_auth_headers(), timeout=15)
        res.raise_for_status()
        obj = res.json()
        t1 = None
        t2 = None
        if isinstance(obj, dict):
            t1 = (
                obj.get("teamOne", {}).get("id")
                if isinstance(obj.get("teamOne"), dict)
                else None
            )
            t2 = (
                obj.get("teamTwo", {}).get("id")
                if isinstance(obj.get("teamTwo"), dict)
                else None
            )
            maps = obj.get("mapList") or []
            parts: set[int] = set()
            for m in maps:
                if not isinstance(m, dict):
                    continue
                ids = m.get("participatedUserIds") or []
                for u in ids:
                    try:
                        parts.add(int(u))
                    except Exception:
                        continue
            return (
                int(t1) if t1 is not None else None,
                int(t2) if t2 is not None else None,
                list(parts),
            )
    except Exception:
        pass
    return (None, None, [])


def enrich_appearances_team_by_match_api(
    appearances: pl.DataFrame,
    matches: pl.DataFrame,
    players: pl.DataFrame,
    *,
    max_calls: Optional[int] = None,
) -> pl.DataFrame:
    """Fill missing appearance.team_id by consulting the match API when needed.

    - Only touches rows where team_id is null after roster inference.
    - Minimizes API calls by fetching only for affected match_ids (optionally limited by max_calls).
    - Heuristic: assign unmatched users to the team that has more rostered players
      among the match participants. If ambiguous (tie or insufficient info), leave null.

    Returns a new DataFrame with the same schema as appearances, with team_id possibly filled.
    """
    if appearances is None or appearances.is_empty():
        return appearances

    # Normalize base columns
    base = appearances.select(
        [
            pl.col("tournament_id").cast(pl.Int64, strict=False),
            pl.col("match_id").cast(pl.Int64, strict=False),
            pl.col("user_id").cast(pl.Int64, strict=False),
            *(
                [pl.col("team_id").cast(pl.Int64, strict=False)]
                if "team_id" in appearances.columns
                else [pl.lit(None).cast(pl.Int64).alias("team_id")]
            ),
        ]
    ).unique(subset=["tournament_id", "match_id", "user_id"])

    # Roster map (tournament_id, user_id) -> team_id
    roster_df = players.select(
        ["tournament_id", "team_id", "user_id"]
    ).with_columns(
        [
            pl.col("tournament_id").cast(pl.Int64, strict=False),
            pl.col("team_id").cast(pl.Int64, strict=False),
            pl.col("user_id").cast(pl.Int64, strict=False),
        ]
    )
    base = (
        base.join(
            roster_df.select(["tournament_id", "user_id", "team_id"]).rename(
                {"team_id": "team_id_roster"}
            ),
            on=["tournament_id", "user_id"],
            how="left",
        )
        .with_columns(
            pl.coalesce([pl.col("team_id"), pl.col("team_id_roster")]).alias(
                "team_id"
            )
        )
        .drop(["team_id_roster"])
    )

    unmatched = base.filter(pl.col("team_id").is_null())
    if unmatched.is_empty():
        return base

    # Prepare per-match enrichment plan
    need = unmatched.select(["tournament_id", "match_id"]).unique().to_dicts()
    if max_calls is not None and len(need) > max_calls:
        need = need[: int(max_calls)]

    # Precompute roster sets for quick membership checks
    # Map (tournament_id, team_id) -> set[user_id]
    roster_sets: dict[tuple[int, int], set[int]] = {}
    for r in (
        players.select(["tournament_id", "team_id", "user_id"])
        .drop_nulls(["team_id"])
        .iter_rows(named=True)
    ):
        key = (int(r["tournament_id"]), int(r["team_id"]))
        roster_sets.setdefault(key, set()).add(int(r["user_id"]))

    # Build mapping for enriched assignments
    assign_rows: list[dict] = []

    for item in need:
        tid = int(item["tournament_id"])
        mid = int(item["match_id"])
        t1, t2, parts = _fetch_match_info(mid)
        if not parts or t1 is None or t2 is None:
            continue
        participants = set(int(u) for u in parts)
        r1 = roster_sets.get((tid, t1), set())
        r2 = roster_sets.get((tid, t2), set())
        c1 = len(participants & r1)
        c2 = len(participants & r2)
        # Heuristics:
        # 1) Prefer assigning to the team with FEWER rostered participants on this match
        #    (common 4v4 case: 3 vs 4 → the unmatched is almost always on the 3 side).
        prefer_smaller: Optional[int] = None
        if c1 != c2:
            prefer_smaller = t1 if c1 < c2 else t2
        # 2) Fallback majority if still ambiguous elsewhere
        majority_team: Optional[int] = None
        if c1 > c2:
            majority_team = t1
        elif c2 > c1:
            majority_team = t2

        # For each unmatched user in this match, assign if clear
        um = unmatched.filter(
            (pl.col("tournament_id") == tid) & (pl.col("match_id") == mid)
        )
        for r in um.iter_rows(named=True):
            uid = int(r["user_id"])
            # If they are in a roster set explicitly, prefer that
            if uid in r1:
                assign_rows.append(
                    {
                        "tournament_id": tid,
                        "match_id": mid,
                        "user_id": uid,
                        "team_id": t1,
                    }
                )
            elif uid in r2:
                assign_rows.append(
                    {
                        "tournament_id": tid,
                        "match_id": mid,
                        "user_id": uid,
                        "team_id": t2,
                    }
                )
            elif prefer_smaller is not None and uid in participants:
                assign_rows.append(
                    {
                        "tournament_id": tid,
                        "match_id": mid,
                        "user_id": uid,
                        "team_id": prefer_smaller,
                    }
                )
            elif majority_team is not None and uid in participants:
                assign_rows.append(
                    {
                        "tournament_id": tid,
                        "match_id": mid,
                        "user_id": uid,
                        "team_id": majority_team,
                    }
                )
            # else leave unassigned

    if assign_rows:
        assign_df = (
            pl.DataFrame(assign_rows)
            .with_columns(
                [
                    pl.col("tournament_id").cast(pl.Int64, strict=False),
                    pl.col("match_id").cast(pl.Int64, strict=False),
                    pl.col("user_id").cast(pl.Int64, strict=False),
                    pl.col("team_id").cast(pl.Int64, strict=False),
                ]
            )
            .unique(subset=["tournament_id", "match_id", "user_id"])
        )
        base = (
            base.join(
                assign_df,
                on=["tournament_id", "match_id", "user_id"],
                how="left",
            )
            .with_columns(
                pl.coalesce([pl.col("team_id_right"), pl.col("team_id")]).alias(
                    "team_id"
                )
            )
            .drop([c for c in ["team_id_right"] if c in base.columns])
        )

    return base


def apply_enrichment_cache(
    appearances: pl.DataFrame, cache_dir: str
) -> pl.DataFrame:
    """Apply cached team_id assignments from local parquet files.

    Expects parquet files under `cache_dir` with columns:
      [tournament_id, match_id, user_id, team_id]

    Returns a new DataFrame with team_id coalesced from cache when present.
    """
    from pathlib import Path

    if appearances is None or appearances.is_empty():
        return appearances

    path = Path(cache_dir)
    if not path.exists() or not path.is_dir():
        return appearances

    # Support both Parquet and Arrow IPC/Feather cache files
    files = sorted([p for p in path.glob("*.parquet") if p.is_file()])
    files += sorted([p for p in path.glob("*.feather") if p.is_file()])
    files += sorted([p for p in path.glob("*.ipc") if p.is_file()])
    if not files:
        # also support subdir layout (e.g., batches/*)
        files = sorted([p for p in path.rglob("*.parquet") if p.is_file()])
        files += sorted([p for p in path.rglob("*.feather") if p.is_file()])
        files += sorted([p for p in path.rglob("*.ipc") if p.is_file()])
    if not files:
        return appearances

    dfs: list[pl.DataFrame] = []
    for p in files:
        try:
            suf = p.suffix.lower()
            if suf in {".feather", ".ipc"}:
                df = pl.read_ipc(str(p))
            else:
                df = pl.read_parquet(str(p))
            # Normalize columns
            cols = set(df.columns)
            needed = {"tournament_id", "match_id", "user_id", "team_id"}
            if not needed.issubset(cols):
                continue
            df = df.select(
                [
                    pl.col("tournament_id").cast(pl.Int64, strict=False),
                    pl.col("match_id").cast(pl.Int64, strict=False),
                    pl.col("user_id").cast(pl.Int64, strict=False),
                    pl.col("team_id").cast(pl.Int64, strict=False),
                ]
            )
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return appearances

    mapping = pl.concat(dfs, how="vertical_relaxed").unique(
        subset=["tournament_id", "match_id", "user_id"]
    )
    out = appearances.join(
        mapping,
        on=["tournament_id", "match_id", "user_id"],
        how="left",
    )
    # prefer cached team id when missing
    if "team_id_right" in out.columns:
        out = out.with_columns(
            pl.coalesce([pl.col("team_id"), pl.col("team_id_right")]).alias(
                "team_id"
            )
        ).drop(["team_id_right"])
    return out


def apply_enrichment_db_cache(appearances: pl.DataFrame, engine) -> pl.DataFrame:
    """Apply cached team_id assignments stored in the database.

    Reads mapping rows [tournament_id, match_id, user_id, team_id] from the
    player appearance team mapping table and coalesces onto the provided
    appearances frame.
    """
    if appearances is None or appearances.is_empty():
        return appearances
    try:
        # Lazy import to avoid heavier dependency at module import time
        from rankings.sql.load import load_player_appearance_teams_df

        mapping = load_player_appearance_teams_df(engine)
        if mapping is None or mapping.is_empty():
            return appearances
        out = appearances.join(
            mapping,
            on=["tournament_id", "match_id", "user_id"],
            how="left",
        )
        # prefer DB team id when present
        if "team_id_right" in out.columns:
            out = out.with_columns(
                pl.coalesce([pl.col("team_id"), pl.col("team_id_right")]).alias(
                    "team_id"
                )
            ).drop(["team_id_right"])
        return out
    except Exception:
        return appearances
