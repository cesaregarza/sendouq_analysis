from __future__ import annotations

"""
Batch-enrich appearance team assignments and save locally in chunks.

This CLI scans unmatched appearance rows (team_id missing after roster join)
for a configured window, fetches match details for those matches, assigns
team_id using heuristics, and writes the assignments to local parquet files
every N matches to allow resume-on-failure.
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import polars as pl
import requests

from rankings.core.constants import SENDOU_PUBLIC_API_BASE_URL
from rankings.scraping.calendar_api import _auth_headers
from rankings.sql import create_engine as rankings_create_engine
from rankings.sql import models as RM
from rankings.sql.load import load_core_tables, load_player_appearances_df
from sqlalchemy.dialects.postgresql import insert as pg_insert


def _load_processed_match_ids(out_dir: Path) -> set[int]:
    done: set[int] = set()
    files = (
        list(out_dir.glob("*.parquet"))
        + list(out_dir.glob("*.feather"))
        + list(out_dir.glob("*.ipc"))
    )
    if not files:
        files = (
            list(out_dir.rglob("*.parquet"))
            + list(out_dir.rglob("*.feather"))
            + list(out_dir.rglob("*.ipc"))
        )
    for p in files:
        try:
            suf = p.suffix.lower()
            if suf in {".feather", ".ipc"}:
                df = pl.read_ipc(str(p))
            else:
                df = pl.read_parquet(str(p))
            if {"match_id"}.issubset(set(df.columns)):
                for mid in df.select("match_id").unique()["match_id"].to_list():
                    try:
                        done.add(int(mid))
                    except Exception:
                        continue
        except Exception:
            continue
    return done


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Batch enrich appearances and save mapping locally."
    )
    ap.add_argument(
        "--db-url",
        type=str,
        default=os.getenv("RANKINGS_DATABASE_URL") or os.getenv("DATABASE_URL"),
    )
    ap.add_argument("--since-days", type=int, default=540)
    ap.add_argument("--only-ranked", action="store_true", default=True)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument(
        "--out-dir", type=str, default="data/enrichment/appearances_enriched"
    )
    ap.add_argument(
        "--format",
        type=str,
        choices=["feather", "parquet"],
        default="feather",
        help="Cache file format; feather (Arrow IPC) is faster for many small files",
    )
    ap.add_argument(
        "--workers", type=int, default=8, help="Concurrent match fetch workers"
    )
    ap.add_argument("--max-matches", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--to-db", action="store_true", help="Write assignments to DB instead of files")
    args = ap.parse_args(argv)

    engine = rankings_create_engine(args.db_url)

    # Load window
    from datetime import datetime, timedelta

    since_ms = (
        int(
            (
                datetime.utcnow() - timedelta(days=int(args.since_days))
            ).timestamp()
            * 1000
        )
        if args.since_days
        else None
    )
    core = load_core_tables(
        engine,
        since_ms=since_ms,
        until_ms=None,
        only_ranked=bool(args.only_ranked),
    )
    matches = core["matches"]
    players = core["players"]
    apps = load_player_appearances_df(engine)
    if apps.is_empty() or matches.is_empty() or players.is_empty():
        print("No data to process.")
        return 0

    # Filter appearances to run keys
    key_df = matches.select(["tournament_id", "match_id"]).unique()
    apps = apps.join(key_df, on=["tournament_id", "match_id"], how="inner")
    # Compute roster join to identify unmatched
    roster_pairs = players.select(["tournament_id", "user_id"]).unique()
    unmatched = apps.join(
        roster_pairs, on=["tournament_id", "user_id"], how="anti"
    )
    if unmatched.is_empty():
        print("No unmatched appearances; nothing to enrich.")
        return 0

    # Distinct match ids needing enrichment
    match_ids = unmatched.select("match_id").unique()["match_id"].to_list()
    match_ids = [int(m) for m in match_ids if m is not None]
    match_ids = sorted(match_ids)
    if args.max_matches is not None:
        match_ids = match_ids[: int(args.max_matches)]

    out_dir = Path(args.out_dir)
    if not args.to_db:
        out_dir.mkdir(parents=True, exist_ok=True)
    processed: set[int] = set()
    if args.resume and not args.to_db:
        processed = _load_processed_match_ids(out_dir)
    todo = [m for m in match_ids if m not in processed]
    print(
        f"Total distinct matches needing enrichment: {len(match_ids)}; to process: {len(todo)}"
    )

    # Precompute roster sets for team inference
    roster_sets: dict[tuple[int, int], set[int]] = {}
    for r in (
        players.select(["tournament_id", "team_id", "user_id"])
        .drop_nulls(["team_id"])
        .iter_rows(named=True)
    ):
        key = (int(r["tournament_id"]), int(r["team_id"]))
        roster_sets.setdefault(key, set()).add(int(r["user_id"]))

    # Build per-match mapping to speed worker lookups: mid -> (tid, [user_ids])
    per_match: dict[int, tuple[int, list[int]]] = {}
    for r in unmatched.select(
        ["tournament_id", "match_id", "user_id"]
    ).iter_rows(named=True):
        tid = int(r["tournament_id"])
        mid = int(r["match_id"])
        uid = int(r["user_id"])
        if mid not in per_match:
            per_match[mid] = (tid, [uid])
        else:
            per_match[mid][1].append(uid)

    # Helper: fetch match details using a session
    def _fetch_match(sess: requests.Session, match_id: int):
        try:
            url = f"{SENDOU_PUBLIC_API_BASE_URL}/tournament-match/{match_id}"
            res = sess.get(url, headers=_auth_headers(), timeout=15)
            res.raise_for_status()
            obj = res.json()
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
            parts = set()
            for m in obj.get("mapList") or []:
                ids = m.get("participatedUserIds") or []
                for u in ids:
                    try:
                        parts.add(int(u))
                    except Exception:
                        continue
            return (
                int(t1) if t1 is not None else None,
                int(t2) if t2 is not None else None,
                parts,
            )
        except Exception:
            return (None, None, set())

    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Iterate in batches
    for i in range(0, len(todo), int(args.batch_size)):
        batch = todo[i : i + int(args.batch_size)]
        if not batch:
            continue

        # Worker function for one match
        def _do_one(mid: int) -> list[dict]:
            tid, uids = per_match.get(mid, (None, []))
            if tid is None or not uids:
                return []
            with requests.Session() as sess:
                t1, t2, parts = _fetch_match(sess, mid)
            if not parts or t1 is None or t2 is None:
                return []
            r1 = roster_sets.get((tid, t1), set())
            r2 = roster_sets.get((tid, t2), set())
            c1 = len(parts & r1)
            c2 = len(parts & r2)
            prefer_smaller = t1 if c1 < c2 else (t2 if c2 < c1 else None)
            majority = t1 if c1 > c2 else (t2 if c2 > c1 else None)
            out: list[dict] = []
            for uid in uids:
                if uid in r1:
                    out.append(
                        {
                            "tournament_id": tid,
                            "match_id": mid,
                            "user_id": uid,
                            "team_id": t1,
                        }
                    )
                elif uid in r2:
                    out.append(
                        {
                            "tournament_id": tid,
                            "match_id": mid,
                            "user_id": uid,
                            "team_id": t2,
                        }
                    )
                elif prefer_smaller is not None and uid in parts:
                    out.append(
                        {
                            "tournament_id": tid,
                            "match_id": mid,
                            "user_id": uid,
                            "team_id": prefer_smaller,
                        }
                    )
                elif majority is not None and uid in parts:
                    out.append(
                        {
                            "tournament_id": tid,
                            "match_id": mid,
                            "user_id": uid,
                            "team_id": majority,
                        }
                    )
            return out

        results: list[dict] = []
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
            futs = {ex.submit(_do_one, mid): mid for mid in batch}
            for fut in as_completed(futs):
                try:
                    part = fut.result()
                    if part:
                        results.extend(part)
                except Exception:
                    continue

        enriched = (
            pl.DataFrame(results)
            if results
            else pl.DataFrame(
                {
                    "tournament_id": [],
                    "match_id": [],
                    "user_id": [],
                    "team_id": [],
                }
            )
        )
        if not enriched.is_empty():
            enriched = enriched.unique(
                subset=["tournament_id", "match_id", "user_id"]
            )
        if args.to_db:
            rows = (
                [
                    {
                        "tournament_id": int(r["tournament_id"]),
                        "match_id": int(r["match_id"]),
                        "player_id": int(r["user_id"]),
                        "team_id": int(r["team_id"]),
                    }
                    for r in enriched.iter_rows(named=True)
                ]
                if not enriched.is_empty()
                else []
            )
            table = RM.PlayerAppearanceTeam.__table__
            stmt = pg_insert(table).values(rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=[
                    table.c.tournament_id,
                    table.c.match_id,
                    table.c.player_id,
                ],
                set_={"team_id": stmt.excluded.team_id},
            )
            with rankings_create_engine(args.db_url).begin() as conn:
                if rows:
                    conn.execute(stmt)
            print(f"Upserted {len(rows)} assignments into DB")
        else:
            # Choose extension/format
            ext = ".feather" if args.format == "feather" else ".parquet"
            out_path = out_dir / f"batch_{i//int(args.batch_size):05d}{ext}"
            if args.format == "feather":
                enriched.write_ipc(str(out_path))
            else:
                enriched.write_parquet(str(out_path))
            print(f"Wrote {enriched.height} assignments -> {out_path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
