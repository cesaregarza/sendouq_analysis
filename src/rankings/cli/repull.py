from __future__ import annotations

import argparse
import logging
import sys
from typing import Iterable

import requests
from sqlalchemy import delete, select

from rankings.cli import db_import as import_cli
from rankings.scraping.api import scrape_tournament
from rankings.scraping.calendar_api import fetch_tournament_players
from rankings.sql import create_all as rankings_create_all
from rankings.sql import create_engine as rankings_create_engine
from rankings.sql import models as RM

logger = logging.getLogger(__name__)


def _scrape_with_players(tournament_id: int, session: requests.Session) -> dict:
    """Fetch tournament payload and enrich it with player matches when available."""
    payload = scrape_tournament(tournament_id, session=session)
    try:
        players_payload = fetch_tournament_players(tournament_id)
    except requests.RequestException as exc:  # keep going on partial fetch
        logger.warning(
            "Players route fetch failed for %s: %s", tournament_id, exc
        )
    else:
        if players_payload:
            payload["player_matches"] = players_payload
    return payload


def _import_payloads(engine, payloads: Iterable[dict]) -> int:
    """Import payloads into the rankings schema."""
    payload_list = list(payloads)
    if not payload_list:
        return 0
    return import_cli.import_file(engine, payload_list)


def _row_to_dict(row) -> dict:
    mapping = row._mapping if hasattr(row, "_mapping") else row
    return dict(mapping)


def _snapshot_for_rollback(
    engine, tournament_ids: list[int]
) -> dict[str, list[dict]]:
    if not tournament_ids:
        return {
            "tournaments": [],
            "stages": [],
            "groups": [],
            "rounds": [],
            "teams": [],
            "matches": [],
            "roster_entries": [],
            "player_appearances": [],
            "player_appearance_teams": [],
        }

    tids = [int(t) for t in tournament_ids]
    snapshot: dict[str, list[dict]] = {}

    with engine.connect() as conn:
        tournament_table = RM.Tournament.__table__
        stage_table = RM.Stage.__table__
        group_table = RM.Group.__table__
        round_table = RM.Round.__table__
        team_table = RM.TournamentTeam.__table__
        match_table = RM.Match.__table__
        roster_table = RM.RosterEntry.__table__
        appearance_table = RM.PlayerAppearance.__table__
        appearance_team_table = RM.PlayerAppearanceTeam.__table__

        snapshot["tournaments"] = [
            _row_to_dict(row)
            for row in conn.execute(
                select(tournament_table).where(
                    tournament_table.c.tournament_id.in_(tids)
                )
            )
        ]

        stage_rows = list(
            conn.execute(
                select(stage_table).where(stage_table.c.tournament_id.in_(tids))
            )
        )
        snapshot["stages"] = [_row_to_dict(row) for row in stage_rows]
        stage_ids = [row[stage_table.c.stage_id] for row in stage_rows]

        if stage_ids:
            group_rows = list(
                conn.execute(
                    select(group_table).where(
                        group_table.c.stage_id.in_(stage_ids)
                    )
                )
            )
            snapshot["groups"] = [_row_to_dict(row) for row in group_rows]
            group_ids = [row[group_table.c.group_id] for row in group_rows]

            round_rows = list(
                conn.execute(
                    select(round_table).where(
                        round_table.c.stage_id.in_(stage_ids)
                    )
                )
            )
            snapshot["rounds"] = [_row_to_dict(row) for row in round_rows]
        else:
            snapshot["groups"] = []
            snapshot["rounds"] = []

        snapshot["teams"] = [
            _row_to_dict(row)
            for row in conn.execute(
                select(team_table).where(team_table.c.tournament_id.in_(tids))
            )
        ]

        snapshot["matches"] = [
            _row_to_dict(row)
            for row in conn.execute(
                select(match_table).where(match_table.c.tournament_id.in_(tids))
            )
        ]

        snapshot["roster_entries"] = [
            _row_to_dict(row)
            for row in conn.execute(
                select(roster_table).where(
                    roster_table.c.tournament_id.in_(tids)
                )
            )
        ]

        snapshot["player_appearances"] = [
            _row_to_dict(row)
            for row in conn.execute(
                select(appearance_table).where(
                    appearance_table.c.tournament_id.in_(tids)
                )
            )
        ]

        snapshot["player_appearance_teams"] = [
            _row_to_dict(row)
            for row in conn.execute(
                select(appearance_team_table).where(
                    appearance_team_table.c.tournament_id.in_(tids)
                )
            )
        ]

    return snapshot


def _delete_current_state(conn, tournament_ids: list[int]) -> None:
    if not tournament_ids:
        return

    tids = [int(t) for t in tournament_ids]
    tournament_table = RM.Tournament.__table__
    stage_table = RM.Stage.__table__
    group_table = RM.Group.__table__
    round_table = RM.Round.__table__
    team_table = RM.TournamentTeam.__table__
    match_table = RM.Match.__table__
    roster_table = RM.RosterEntry.__table__
    appearance_table = RM.PlayerAppearance.__table__
    appearance_team_table = RM.PlayerAppearanceTeam.__table__

    stage_rows = list(
        conn.execute(
            select(stage_table.c.stage_id).where(
                stage_table.c.tournament_id.in_(tids)
            )
        )
    )
    stage_ids = [row[0] for row in stage_rows]

    group_ids: list[int] = []
    round_ids: list[int] = []
    if stage_ids:
        group_rows = list(
            conn.execute(
                select(group_table.c.group_id).where(
                    group_table.c.stage_id.in_(stage_ids)
                )
            )
        )
        group_ids = [row[0] for row in group_rows]

        round_rows = list(
            conn.execute(
                select(round_table.c.round_id).where(
                    round_table.c.stage_id.in_(stage_ids)
                )
            )
        )
        round_ids = [row[0] for row in round_rows]

    conn.execute(
        delete(appearance_table).where(
            appearance_table.c.tournament_id.in_(tids)
        )
    )
    conn.execute(
        delete(appearance_team_table).where(
            appearance_team_table.c.tournament_id.in_(tids)
        )
    )
    conn.execute(
        delete(roster_table).where(roster_table.c.tournament_id.in_(tids))
    )
    conn.execute(
        delete(match_table).where(match_table.c.tournament_id.in_(tids))
    )
    if round_ids:
        conn.execute(
            delete(round_table).where(round_table.c.round_id.in_(round_ids))
        )
    if group_ids:
        conn.execute(
            delete(group_table).where(group_table.c.group_id.in_(group_ids))
        )
    if stage_ids:
        conn.execute(
            delete(stage_table).where(stage_table.c.stage_id.in_(stage_ids))
        )
    conn.execute(delete(team_table).where(team_table.c.tournament_id.in_(tids)))
    conn.execute(
        delete(tournament_table).where(
            tournament_table.c.tournament_id.in_(tids)
        )
    )


def _restore_snapshot(
    engine, snapshot: dict[str, list[dict]], tournament_ids: list[int]
) -> None:
    if not tournament_ids:
        return

    with engine.begin() as conn:
        _delete_current_state(conn, tournament_ids)

        def _insert_rows(table, rows):
            if rows:
                conn.execute(table.insert(), rows)

        tournament_table = RM.Tournament.__table__
        stage_table = RM.Stage.__table__
        group_table = RM.Group.__table__
        round_table = RM.Round.__table__
        team_table = RM.TournamentTeam.__table__
        match_table = RM.Match.__table__
        roster_table = RM.RosterEntry.__table__
        appearance_table = RM.PlayerAppearance.__table__
        appearance_team_table = RM.PlayerAppearanceTeam.__table__

        _insert_rows(tournament_table, snapshot.get("tournaments", []))
        _insert_rows(stage_table, snapshot.get("stages", []))
        _insert_rows(group_table, snapshot.get("groups", []))
        _insert_rows(round_table, snapshot.get("rounds", []))
        _insert_rows(team_table, snapshot.get("teams", []))
        _insert_rows(match_table, snapshot.get("matches", []))
        _insert_rows(roster_table, snapshot.get("roster_entries", []))
        _insert_rows(appearance_table, snapshot.get("player_appearances", []))
        _insert_rows(
            appearance_team_table, snapshot.get("player_appearance_teams", [])
        )


def import_with_rollback(
    engine,
    payloads: Iterable[dict],
    tournament_ids: list[int],
    *,
    log: logging.Logger,
) -> int:
    snapshot = _snapshot_for_rollback(engine, tournament_ids)
    try:
        return _import_payloads(engine, payloads)
    except Exception as exc:
        log.error(
            "Import failed for tournaments %s; attempting rollback: %s",
            tournament_ids,
            exc,
        )
        try:
            _restore_snapshot(engine, snapshot, tournament_ids)
        except Exception as restore_exc:  # pragma: no cover - best effort
            log.error(
                "Rollback failed for tournaments %s: %s",
                tournament_ids,
                restore_exc,
            )
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-scrape tournaments from Sendou and import them into the "
            "rankings database."
        )
    )
    parser.add_argument(
        "tournament_ids",
        nargs="+",
        type=int,
        help="One or more tournament IDs to refresh",
    )
    parser.add_argument(
        "--db-url",
        dest="db_url",
        default=None,
        help=(
            "Override database URL. Defaults to env RANKINGS_DATABASE_URL or "
            "DATABASE_URL"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level for stdout output",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    session = requests.Session()
    engine = rankings_create_engine(args.db_url)
    rankings_create_all(engine)

    payloads = []
    for tid in args.tournament_ids:
        try:
            payload = _scrape_with_players(tid, session=session)
        except requests.RequestException as exc:
            logger.error("Failed to scrape tournament %s: %s", tid, exc)
            return 1
        payloads.append(payload)
        logger.info("Fetched tournament %s", tid)

    try:
        inserted = import_with_rollback(
            engine, payloads, [int(t) for t in args.tournament_ids], log=logger
        )
    except Exception:
        return 1

    logger.info(
        "Ingestion complete for %d tournament(s); import attempted %d row groups",
        len(payloads),
        inserted,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
