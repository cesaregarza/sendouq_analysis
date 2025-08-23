"""
Tournament data parsing utilities for Sendou.ink exports.

This module handles the JSON data downloaded from https://sendou.ink/to/[tournament_id]
and normalizes it into structured Polars DataFrames for analysis.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import polars as pl


def parse_tournaments_data(
    tournaments: list[dict],
) -> Dict[str, Optional[pl.DataFrame]]:
    """Parse a Sendou tournament JSON list into a set of polars DataFrames.

    The raw data contains a list of tournaments. Each entry has a
    `tournament` key with nested `data` (stages, groups, rounds and
    matches) and `ctx` (teams, participants, and tournament metadata).
    This function extracts and normalizes this information into separate tables.

    Parameters
    ----------
    tournaments : list of dict
        List of tournament entries from Sendou.ink JSON export.

    Returns
    -------
    dict[str, Optional[pl.DataFrame]]
        A dictionary containing seven tables: ``tournaments``, ``stages``,
        ``groups``, ``rounds``, ``teams``, ``players`` and ``matches``.
        Tables without any rows are returned as ``None``.

    Notes
    -----
    * The ``tournaments`` table contains comprehensive metadata about each
      tournament including organization, settings, staff, and configuration.
    * Match outcomes are deduced from the ``opponent1`` and ``opponent2``
      sub-dictionaries. If neither opponent has a ``result`` of
      ``"win"``, the ``winner_team_id`` and ``loser_team_id`` columns
      will be set to ``None``.
    * For matches where a team receives a bye (``opponent2`` is ``None``),
      the second team's columns will be ``None`` and the match will
      effectively be a walkover.
    * Additional fields from the tournament metadata are carried into
      the tables wherever they help to disambiguate the data (for
      example, ``tournament_id``, ``stage_id`` and ``group_id``).

    Examples
    --------
    >>> import json
    >>> with open("tournament_data.json") as f:
    ...     data = json.load(f)
    >>> tables = parse_tournaments_data(data)
    >>> tournaments_df = tables['tournaments']
    >>> matches_df = tables['matches']
    >>> teams_df = tables['teams']
    >>> players_df = tables['players']
    """
    # Prepare containers for each table
    tournament_rows: List[Dict[str, object]] = []
    stage_rows: List[Dict[str, object]] = []
    group_rows: List[Dict[str, object]] = []
    round_rows: List[Dict[str, object]] = []
    team_rows: List[Dict[str, object]] = []
    player_rows: List[Dict[str, object]] = []
    match_rows: List[Dict[str, object]] = []

    # Track seen match IDs and tournament IDs to avoid duplicates
    seen_match_ids: set = set()
    seen_tournament_ids: set = set()

    # Iterate through each tournament entry
    for entry in tournaments:
        tournament = entry.get("tournament", {})
        data = tournament.get("data", {})
        ctx = tournament.get("ctx", {})

        # Assign a unique identifier for the tournament
        tournament_id = ctx.get("id")

        # ---------- Tournament Metadata ----------
        row = {
            "tournament_id": tournament_id,
            "event_id": ctx.get("eventId"),
            "name": ctx.get("name"),
            "description": ctx.get("description"),
            "start_time": ctx.get("startTime"),
            "is_finalized": ctx.get("isFinalized", False),
            "parent_tournament_id": ctx.get("parentTournamentId"),
            "discord_url": ctx.get("discordUrl"),
            "logo_url": ctx.get("logoUrl"),
            "logo_validated_at": ctx.get("logoValidatedAt"),
            "logo_src": ctx.get("logoSrc"),
            "map_picking_style": ctx.get("mapPickingStyle"),
            "rules": ctx.get("rules"),
        }

        # Tags (convert comma-separated string to list)
        tags_str = ctx.get("tags")
        if tags_str:
            row["tags"] = [
                tag.strip() for tag in tags_str.split(",") if tag.strip()
            ]
        else:
            row["tags"] = None

        # Cast accounts (keep as list)
        cast_accounts = ctx.get("castTwitchAccounts", []) or []
        row["cast_twitch_accounts"] = cast_accounts if cast_accounts else None

        # Organization info
        org = ctx.get("organization")
        if org:
            row["org_id"] = org.get("id")
            row["org_name"] = org.get("name")
            row["org_slug"] = org.get("slug")
            row["org_description"] = org.get("description")
            row["org_url"] = org.get("url")
            row["org_logo_url"] = org.get("logoUrl")
            row["org_socials"] = (
                str(org.get("socials")) if org.get("socials") else None
            )
        else:
            row.update(
                {
                    "org_id": None,
                    "org_name": None,
                    "org_slug": None,
                    "org_description": None,
                    "org_url": None,
                    "org_logo_url": None,
                    "org_socials": None,
                }
            )

        # Author info
        author = ctx.get("author")
        if author:
            row["author_id"] = author.get("id")
            row["author_username"] = author.get("username")
            row["author_discord_id"] = author.get("discordId")
        else:
            row["author_id"] = None
            row["author_username"] = None
            row["author_discord_id"] = None

        # Staff info (serialized list)
        staff = ctx.get("staff", []) or []
        row["staff"] = str(staff) if staff else None
        row["staff_count"] = len(staff)

        # Settings (serialized)
        settings = ctx.get("settings", {})
        row["settings"] = str(settings) if settings else None

        # Extract specific settings
        if settings:
            row["settings_is_league"] = settings.get("isLeague", False)
            row["settings_is_ranked"] = settings.get("isRanked", False)
            row["settings_enable_no_screen"] = settings.get(
                "enableNoScreen", False
            )
            row["settings_autonomous_subs"] = settings.get(
                "autonomousSubs", False
            )
            row["settings_registration_cap"] = settings.get("regClosesAt")
            row["settings_min_members_per_team"] = settings.get(
                "minMembersPerTeam"
            )
            row["settings_timezone"] = settings.get("timezone")
        else:
            row.update(
                {
                    "settings_is_league": None,
                    "settings_is_ranked": None,
                    "settings_enable_no_screen": None,
                    "settings_autonomous_subs": None,
                    "settings_registration_cap": None,
                    "settings_min_members_per_team": None,
                    "settings_timezone": None,
                }
            )

        # Sub counts
        sub_counts = ctx.get("subCounts", {})
        if sub_counts and isinstance(sub_counts, dict):
            row["sub_count_plus_one"] = sub_counts.get("+1", 0)
            row["sub_count_plus_two"] = sub_counts.get("+2", 0)
        else:
            row["sub_count_plus_one"] = 0
            row["sub_count_plus_two"] = 0

        # Casted matches info
        casted_info = ctx.get("castedMatchesInfo", []) or []
        row["casted_matches_count"] = len(casted_info)
        row["casted_matches_info"] = str(casted_info) if casted_info else None

        # Map pool info
        tie_breaker_pool = ctx.get("tieBreakerMapPool", []) or []
        to_set_pool = ctx.get("toSetMapPool", []) or []
        row["tie_breaker_map_pool"] = (
            str(tie_breaker_pool) if tie_breaker_pool else None
        )
        row["to_set_map_pool"] = str(to_set_pool) if to_set_pool else None

        # Bracket progression overrides
        bracket_overrides = ctx.get("bracketProgressionOverrides", []) or []
        row["bracket_progression_overrides"] = (
            str(bracket_overrides) if bracket_overrides else None
        )

        # Participated users count
        participated_users = ctx.get("participatedUsers", []) or []
        row["participated_users_count"] = len(participated_users)

        # Team and match counts
        row["team_count"] = len(ctx.get("teams", []))
        row["match_count"] = len(data.get("match", []))
        row["stage_count"] = len(data.get("stage", []))
        row["group_count"] = len(data.get("group", []))
        row["round_count"] = len(data.get("round", []))

        tournament_rows.append(row)

        # ---------- Stages ----------
        for stage in data.get("stage", []) or []:
            row: Dict[str, object] = {
                "tournament_id": tournament_id,
                "stage_id": stage.get("id"),
                "stage_name": stage.get("name"),
                "stage_number": stage.get("number"),
                "stage_type": stage.get("type"),
            }
            # Flatten stage settings
            settings: Dict[str, object] = stage.get("settings", {}) or {}
            for key, value in settings.items():
                row[f"setting_{key}"] = value
            stage_rows.append(row)

        # ---------- Groups ----------
        for group in data.get("group", []) or []:
            group_rows.append(
                {
                    "tournament_id": tournament_id,
                    "stage_id": group.get("stage_id"),
                    "group_id": group.get("id"),
                    "group_number": group.get("number"),
                }
            )

        # ---------- Rounds ----------
        for rnd in data.get("round", []) or []:
            maps = rnd.get("maps", {}) or {}
            round_rows.append(
                {
                    "tournament_id": tournament_id,
                    "stage_id": rnd.get("stage_id"),
                    "group_id": rnd.get("group_id"),
                    "round_id": rnd.get("id"),
                    "round_number": rnd.get("number"),
                    "maps_count": maps.get("count"),
                    "maps_type": maps.get("type"),
                }
            )

        # ---------- Teams & Players ----------
        for team in ctx.get("teams", []) or []:
            team_id = team.get("id")
            team_rows.append(
                {
                    "tournament_id": tournament_id,
                    "team_id": team_id,
                    "team_name": team.get("name"),
                    "seed": team.get("seed"),
                    "prefers_not_to_host": bool(team.get("prefersNotToHost")),
                    "no_screen": bool(team.get("noScreen")),
                    "dropped_out": bool(team.get("droppedOut")),
                    "invite_code": team.get("inviteCode"),
                    "created_at": team.get("createdAt"),
                }
            )

            # Parse roster - use activeRosterUserIds if available, otherwise all members
            active_roster_ids = team.get("activeRosterUserIds")
            members = team.get("members", []) or []

            # Create a set to track which user IDs we've already added (deduplication)
            added_user_ids = set()

            # If activeRosterUserIds is specified, only include those members
            if active_roster_ids:
                # Convert to set for efficient lookup
                active_ids_set = set(active_roster_ids)
                members_to_add = [
                    m for m in members if m.get("userId") in active_ids_set
                ]
            else:
                # Include all members if no active roster specified
                members_to_add = members

            # Add members, deduplicating by user_id
            for member in members_to_add:
                user_id = member.get("userId")
                # Skip if we've already added this user to this team
                if user_id in added_user_ids:
                    continue
                added_user_ids.add(user_id)

                player_rows.append(
                    {
                        "tournament_id": tournament_id,
                        "team_id": team_id,
                        "user_id": user_id,
                        "username": member.get("username"),
                        "discord_id": member.get("discordId"),
                        "in_game_name": member.get("inGameName"),
                        "country": member.get("country"),
                        "twitch": member.get("twitch"),
                        "is_owner": bool(member.get("isOwner")),
                        "roster_created_at": member.get("createdAt"),
                    }
                )

        # ---------- Matches ----------
        for match in data.get("match", []) or []:
            match_id = match.get("id")

            # Skip duplicate match IDs
            if match_id in seen_match_ids:
                continue
            seen_match_ids.add(match_id)

            # Basic identifiers and meta
            row = {
                "tournament_id": tournament_id,
                "stage_id": match.get("stage_id"),
                "group_id": match.get("group_id"),
                "round_id": match.get("round_id"),
                "match_id": match_id,
                "match_number": match.get("number"),
                "status": match.get("status"),
                "last_game_finished_at": match.get("lastGameFinishedAt"),
                "match_created_at": match.get("createdAt"),
            }
            # Opponents may be missing (e.g. byes)
            opp1 = match.get("opponent1", {}) or {}
            opp2 = match.get("opponent2") or {}

            # Normalize opponent one fields
            row.update(
                {
                    "team1_id": opp1.get("id"),
                    "team1_position": opp1.get("position"),
                    "team1_score": opp1.get("score"),
                    "team1_result": opp1.get("result"),
                }
            )
            # Normalize opponent two fields (may be None if a bye)
            row.update(
                {
                    "team2_id": opp2.get("id") if opp2 else None,
                    "team2_position": opp2.get("position") if opp2 else None,
                    "team2_score": opp2.get("score") if opp2 else None,
                    "team2_result": opp2.get("result") if opp2 else None,
                }
            )

            # Determine winner and loser when possible
            team1_res = row.get("team1_result")
            team2_res = row.get("team2_result")
            winner_id: Optional[int] = None
            loser_id: Optional[int] = None
            if team1_res == "win":
                winner_id = row.get("team1_id")
                loser_id = row.get("team2_id")
            elif team2_res == "win":
                winner_id = row.get("team2_id")
                loser_id = row.get("team1_id")

            # Compute score difference and total games when both scores are present
            team1_score = row.get("team1_score")
            team2_score = row.get("team2_score")
            score_diff: Optional[int] = None
            total_games: Optional[int] = None
            if team1_score is not None and team2_score is not None:
                total_games = team1_score + team2_score
                if winner_id == row.get("team1_id"):
                    score_diff = team1_score - team2_score
                elif winner_id == row.get("team2_id"):
                    score_diff = team2_score - team1_score

            is_bye = (opp2 is None) or (row["status"] in {"bye", "forfeit"})

            row["winner_team_id"] = winner_id
            row["loser_team_id"] = loser_id
            row["score_diff"] = score_diff
            row["total_games"] = total_games
            row["is_bye"] = is_bye

            match_rows.append(row)

    # Convert to polars DataFrames with proper schema inference
    # Use infer_schema_length=None to scan all rows for consistent schema
    tournament_df = (
        pl.DataFrame(tournament_rows, infer_schema_length=None)
        if tournament_rows
        else None
    )
    stage_df = (
        pl.DataFrame(stage_rows, infer_schema_length=None)
        if stage_rows
        else None
    )
    group_df = (
        pl.DataFrame(group_rows, infer_schema_length=None)
        if group_rows
        else None
    )
    round_df = (
        pl.DataFrame(round_rows, infer_schema_length=None)
        if round_rows
        else None
    )
    team_df = (
        pl.DataFrame(team_rows, infer_schema_length=None) if team_rows else None
    )
    player_df = (
        pl.DataFrame(player_rows, infer_schema_length=None)
        if player_rows
        else None
    )
    match_df = (
        pl.DataFrame(match_rows, infer_schema_length=None)
        if match_rows
        else None
    )

    return {
        "tournaments": tournament_df,
        "stages": stage_df,
        "groups": group_df,
        "rounds": round_df,
        "teams": team_df,
        "players": player_df,
        "matches": match_df,
    }
