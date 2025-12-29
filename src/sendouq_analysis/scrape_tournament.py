import json
import os
import traceback

import requests
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from rankings.core.logging import get_logger, log_timing
from rankings.scraping.api import validate_tournament_data
from rankings.scraping.turbo_stream import extract_tournament_data
from sendouq_analysis.sql.tourney_models import (  # ParticipatedUser  # Uncomment if this model exists
    Base,
    BracketProgression,
    BracketProgressionOverride,
    BracketProgressionSource,
    CastedMatch,
    CastedMatchesInfo,
    CastTwitchAccount,
    CheckIn,
    Group,
    LockedMatch,
    Map,
    MapPool,
    Match,
    Organization,
    Round,
    Staff,
    Stage,
    StageSettings,
    Stream,
    SubCount,
    Team,
    TeamMember,
    Tournament,
    TournamentSettings,
    User,
)


def build_url(tournament_id: int, legacy: bool = False) -> str:
    """Construct the URL for a given tournament ID.

    Args:
        tournament_id: Tournament ID to scrape
        legacy: If True, use the legacy ?_data endpoint format
    """
    if legacy:
        return f"https://sendou.ink/to/{tournament_id}?_data=features%2Ftournament%2Froutes%2Fto.%24id"
    return f"https://sendou.ink/to/{tournament_id}/results.data"


def _try_turbo_stream_endpoint(
    tournament_id: int, session: requests.Session, logger
) -> dict | None:
    """Try fetching from the new turbo-stream .data endpoint."""
    url = build_url(tournament_id, legacy=False)
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        raw_data = response.json()
        data = extract_tournament_data(raw_data)
        if data and validate_tournament_data(data):
            logger.debug(f"Fetched tournament {tournament_id} via turbo-stream")
            return data
    except Exception as e:
        logger.debug(f"Turbo-stream endpoint failed for {tournament_id}: {e}")
    return None


def _try_legacy_endpoint(
    tournament_id: int, session: requests.Session, logger
) -> dict | None:
    """Try fetching from the legacy ?_data endpoint (plain JSON)."""
    url = build_url(tournament_id, legacy=True)
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data and validate_tournament_data(data):
            logger.debug(f"Fetched tournament {tournament_id} via legacy endpoint")
            return data
    except Exception as e:
        logger.debug(f"Legacy endpoint failed for {tournament_id}: {e}")
    return None


def scrape_tournament(tournament_id: int, session=requests.Session()) -> dict:
    """Scrape tournament data from the API."""
    logger = get_logger(__name__)
    logger.debug(f"Scraping tournament {tournament_id}")

    with log_timing(logger, f"tournament {tournament_id} scraping"):
        # Try turbo-stream endpoint first (new format)
        data = _try_turbo_stream_endpoint(tournament_id, session, logger)
        if data is not None:
            logger.info(f"Successfully scraped tournament {tournament_id}")
            return data

        # Fall back to legacy endpoint
        data = _try_legacy_endpoint(tournament_id, session, logger)
        if data is not None:
            logger.info(f"Successfully scraped tournament {tournament_id}")
            return data

    raise ValueError(
        f"Could not extract tournament data for {tournament_id} "
        "(tried both turbo-stream and legacy endpoints)"
    )


def parse_staff_user(user_data: dict, db_session) -> User:
    """Parse user data and return a User instance."""
    if not user_data:
        return None
    user = db_session.query(User).get(user_data["id"])
    if not user:
        user = User(
            id=user_data["id"],
            username=user_data.get("username"),
            discord_id=user_data.get("discordId"),
            discord_avatar=user_data.get("discordAvatar"),
            custom_url=user_data.get("customUrl"),
            chat_name_color=user_data.get("chatNameColor"),
        )
        db_session.add(user)
    return user


def parse_user(user_data: dict, db_session) -> User:
    """Parse user data and return a User instance."""
    if not user_data:
        return None
    user = db_session.query(User).get(user_data["userId"])
    if not user:
        user = User(
            id=user_data["userId"],
            username=user_data.get("username"),
            discord_id=user_data.get("discordId"),
            discord_avatar=user_data.get("discordAvatar"),
            custom_url=user_data.get("customUrl"),
            chat_name_color=user_data.get("chatNameColor"),
        )
        db_session.add(user)
    return user


def parse_organization(org_data: dict, db_session) -> Organization:
    """Parse organization data and return an Organization instance."""
    if not org_data:
        return None
    org = db_session.query(Organization).get(org_data["id"])
    if not org:
        org = Organization(
            id=org_data["id"],
            name=org_data.get("name"),
            slug=org_data.get("slug"),
            avatar_url=org_data.get("avatarUrl"),
        )
        db_session.add(org)
    return org


def parse_tournament_settings(
    settings_data: dict, db_session
) -> TournamentSettings:
    """Parse tournament settings and return a TournamentSettings instance."""
    if not settings_data:
        return None
    settings = TournamentSettings(
        bracket_progression_overrides=json.dumps(
            settings_data.get("bracketProgressionOverrides", [])
        ),
        teams_per_group=settings_data.get("teamsPerGroup"),
        third_place_match=settings_data.get("thirdPlaceMatch", False),
        is_ranked=settings_data.get("isRanked", False),
        enable_no_screen_toggle=settings_data.get(
            "enableNoScreenToggle", False
        ),
        autonomous_subs=settings_data.get("autonomousSubs", False),
        reg_closes_at=settings_data.get("regClosesAt"),
        auto_check_in_all=settings_data.get("autoCheckInAll", False),
        deadlines=settings_data.get("deadlines"),
        is_invitational=settings_data.get("isInvitational", False),
        swiss_group_count=settings_data.get("swiss", {}).get("groupCount"),
        swiss_round_count=settings_data.get("swiss", {}).get("roundCount"),
    )
    db_session.add(settings)
    db_session.flush()
    return settings


def parse_staff(staff_data: list, tournament_id: int, db_session):
    """Parse staff data and add Staff instances to the session."""
    for staff_member in staff_data:
        user = parse_staff_user(staff_member, db_session)
        if user:
            staff = Staff(
                tournament_id=tournament_id,
                user_id=user.id,
                role=staff_member.get("role"),
            )
            db_session.add(staff)


def parse_team_members(members_data: list, team_id: int, db_session):
    """Parse team members and add TeamMember instances to the session."""
    for member_data in members_data:
        user = parse_user(member_data, db_session)
        if user:
            team_member = TeamMember(
                team_id=team_id,
                user_id=user.id,
                username=member_data.get("username"),
                discord_id=member_data.get("discordId"),
                discord_avatar=member_data.get("discordAvatar"),
                custom_url=member_data.get("customUrl"),
                country=member_data.get("country"),
                twitch=member_data.get("twitch"),
                is_owner=bool(member_data.get("isOwner", False)),
                created_at=member_data.get("createdAt"),
                in_game_name=member_data.get("inGameName"),
            )
            db_session.add(team_member)


def parse_check_ins(check_ins_data: list, team_id: int, db_session):
    """Parse check-ins and add CheckIn instances to the session."""
    for check_in_data in check_ins_data:
        check_in = CheckIn(
            team_id=team_id,
            bracket_idx=check_in_data.get("bracketIdx"),
            checked_in_at=check_in_data.get("checkedInAt"),
            is_check_out=bool(check_in_data.get("isCheckOut", False)),
        )
        db_session.add(check_in)


def parse_map_pool(map_pool_data: list, team_id: int, db_session):
    """Parse map pool data and add MapPool instances to the session."""
    for map_pool_entry in map_pool_data:
        map_pool = MapPool(
            team_id=team_id,
            stage_id=map_pool_entry.get("stageId"),
            mode=map_pool_entry.get("mode"),
        )
        db_session.add(map_pool)


def parse_sub_counts(sub_counts_data: list, tournament_id: int, db_session):
    """Parse sub counts and add SubCount instances to the session."""
    for sub_count in sub_counts_data:
        sub = SubCount(
            tournament_id=tournament_id,
            visibility=sub_count.get("visibility"),
            count=sub_count.get("count"),
        )
        db_session.add(sub)


def parse_bracket_progression_overrides(
    overrides_data: list, tournament_id: int, db_session
):
    """Parse bracket progression overrides and add BracketProgressionOverride instances."""
    for override in overrides_data:
        bracket_override = BracketProgressionOverride(
            tournament_id=tournament_id, override_data=json.dumps(override)
        )
        db_session.add(bracket_override)


def parse_cast_twitch_accounts(
    cast_twitch_accounts_data, tournament_id: int, db_session
):
    """Parse cast Twitch accounts and add CastTwitchAccount instances."""
    if cast_twitch_accounts_data:
        for twitch_account in cast_twitch_accounts_data:
            cast_account = CastTwitchAccount(
                tournament_id=tournament_id, twitch_account=twitch_account
            )
            db_session.add(cast_account)


def parse_casted_matches_info(
    casted_matches_info_data: dict, tournament_id: int, db_session
):
    """Parse casted matches info and related entities."""
    if not casted_matches_info_data:
        return None

    # Parse casted matches
    for casted_match in casted_matches_info_data.get("castedMatches", []):
        casted_match_obj = CastedMatch(
            tournament_id=tournament_id,
            twitch_account=casted_match.get("twitchAccount"),
            match_id=casted_match.get("matchId"),
        )
        db_session.add(casted_match_obj)

    # Parse locked matches
    for locked_match_id in casted_matches_info_data.get("lockedMatches", []):
        locked_match = LockedMatch(
            tournament_id=tournament_id, match_id=locked_match_id
        )
        db_session.add(locked_match)


def parse_bracket_progression(
    bracket_progression_data: list, tournament_id: int, db_session
):
    """Parse bracket progression and related sources."""
    for progression in bracket_progression_data:
        bracket_prog = BracketProgression(
            tournament_id=tournament_id,
            type=progression.get("type"),
            name=progression.get("name"),
        )
        db_session.add(bracket_prog)
        db_session.flush()  # To get the bracket_prog.id

        for source in progression.get("sources", []):
            progression_source = BracketProgressionSource(
                bracket_progression_id=bracket_prog.id,
                bracket_idx=source.get("bracketIdx"),
                placements=json.dumps(source.get("placements", [])),
            )
            db_session.add(progression_source)


def parse_tie_breaker_map_pool(
    tie_breaker_map_pool_data: list, tournament_id: int, db_session
):
    """Parse tie-breaker map pool data and add MapPool instances."""
    for map_pool_entry in tie_breaker_map_pool_data:
        map_pool = MapPool(
            tournament_id=tournament_id,
            stage_id=map_pool_entry.get("stageId"),
            mode=map_pool_entry.get("mode"),
            is_tie_breaker=True,  # Assuming a boolean flag to indicate tie-breaker maps
        )
        db_session.add(map_pool)


def parse_to_set_map_pool(
    to_set_map_pool_data: list, tournament_id: int, db_session
):
    """Parse to-set map pool data and add MapPool instances."""
    for map_pool_entry in to_set_map_pool_data:
        map_pool = MapPool(
            tournament_id=tournament_id,
            stage_id=map_pool_entry.get("stageId"),
            mode=map_pool_entry.get("mode"),
            is_to_set=True,  # Assuming a boolean flag to indicate to-set maps
        )
        db_session.add(map_pool)


# Uncomment and define this function if ParticipatedUser model exists
# def parse_participated_users(participated_users_data: list, tournament_id: int, db_session):
#     """Parse participated users and associate them with the tournament."""
#     for user_id in participated_users_data:
#         # Assuming there's a ParticipatedUser model to handle this relationship
#         participated_user = ParticipatedUser(
#             tournament_id=tournament_id,
#             user_id=user_id
#         )
#         db_session.add(participated_user)


def parse_teams(teams_data: list, tournament_id: int, db_session):
    """Parse teams data and add Team and TeamMember instances to the session."""
    for team_data in teams_data:
        team = Team(
            id=team_data["id"],
            tournament_id=tournament_id,
            name=team_data.get("name"),
            seed=team_data.get("seed"),
            prefers_not_to_host=bool(team_data.get("prefersNotToHost", False)),
            no_screen=bool(team_data.get("noScreen", False)),
            dropped_out=bool(team_data.get("droppedOut", False)),
            invite_code=team_data.get("inviteCode"),
            created_at=team_data.get("createdAt"),
            active_roster_user_ids=json.dumps(
                team_data.get("activeRosterUserIds", [])
            ),
            starting_bracket_idx=team_data.get("startingBracketIdx"),
            pickup_avatar_url=team_data.get("pickupAvatarUrl"),
            avg_seeding_skill_ordinal=team_data.get("avgSeedingSkillOrdinal"),
        )
        db_session.add(team)

        # Parse team members
        members = team_data.get("members", [])
        parse_team_members(members, team.id, db_session)

        # Parse check-ins if available
        check_ins = team_data.get("checkIns", [])
        if check_ins:
            parse_check_ins(check_ins, team.id, db_session)

        # Parse map pools if available
        map_pools = team_data.get("mapPool", [])
        if map_pools:
            parse_map_pool(map_pools, team.id, db_session)


def scrape_tournaments(
    batch_size: int,
    start_value: int,
    end_value: int = -1,
    max_failures: int = 10,
):
    """Scrape tournaments from the API and store them in the database."""
    # Initialize database connection
    db_path = "data/tournaments.db"
    if not os.path.exists("data"):
        os.makedirs("data")

    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)  # Ensure all tables are created
    Session = sessionmaker(bind=engine)
    db_session = Session()

    data_batch = []
    current_ids = []
    failures = 0
    current_id = start_value

    with tqdm(desc="Scraping Tournaments", unit="tournament") as pbar:
        session_requests = requests.Session()

        while True:
            if end_value != -1 and current_id > end_value:
                break

            try:
                tournament_json = scrape_tournament(
                    current_id, session=session_requests
                )
                data_batch.append(tournament_json)
                current_ids.append(current_id)
                failures = 0
            except Exception as e:
                print(f"Failed to scrape tournament {current_id}:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print("Traceback:")
                print(traceback.format_exc())
                failures += 1
                if end_value == -1 and failures >= max_failures:
                    print(
                        f"Maximum failures ({max_failures}) reached. Stopping scraping."
                    )
                    break
            finally:
                current_id += 1
                pbar.update(1)

            # Process batch if size reached
            if len(data_batch) >= batch_size:
                process_tournament_batch(data_batch, current_ids, db_session)
                data_batch = []
                current_ids = []

        # Process remaining data
        if data_batch:
            process_tournament_batch(data_batch, current_ids, db_session)

    db_session.close()
    print("Scraping completed.")


def process_tournament_batch(data_batch: list, current_ids: list, db_session):
    """Process a batch of tournament data."""
    for tournament_data, tournament_id in zip(data_batch, current_ids):
        try:
            tournament_obj = tournament_data.get("tournament")
            if not tournament_obj:
                print(
                    f"Skipping tournament {tournament_id} - missing tournament object"
                )
                continue

            tournament_ctx = tournament_obj.get("ctx", {})
            tournament_data_content = tournament_obj.get("data", {})
            if not tournament_ctx:
                print(f"Skipping tournament {tournament_id} - missing context")
                continue

            # Parse core entities
            settings = parse_tournament_settings(
                tournament_ctx.get("settings"), db_session
            )
            author = parse_staff_user(tournament_ctx.get("author"), db_session)
            org = parse_organization(
                tournament_ctx.get("organization"), db_session
            )

            # Create tournament - use tournament_id from URL, not from data
            tournament = Tournament(
                id=tournament_id,  # Use the ID from the URL
                name=tournament_ctx.get("name"),
                description=tournament_ctx.get("description"),
                start_time=tournament_ctx.get("startTime"),
                discord_url=tournament_ctx.get("discordUrl"),
                tags=json.dumps(tournament_ctx.get("tags"))
                if tournament_ctx.get("tags")
                else None,
                rules=tournament_ctx.get("rules"),
                map_picking_style=tournament_ctx.get("mapPickingStyle"),
                logo_src=tournament_ctx.get("logoSrc"),
                logo_url=tournament_ctx.get("logoUrl"),
                logo_validated_at=tournament_ctx.get("logoValidatedAt"),
                event_id=tournament_ctx.get("eventId"),
                is_finalized=bool(tournament_ctx.get("isFinalized", False)),
                settings_id=settings.id if settings else None,
                organization_id=org.id if org else None,
                author_id=author.id if author else None,
            )
            db_session.add(tournament)
            db_session.flush()  # Ensure tournament has an ID before creating related records

            # Parse related entities
            if tournament_ctx.get("staff"):
                parse_staff(
                    tournament_ctx.get("staff"), tournament.id, db_session
                )
            if tournament_ctx.get("teams"):
                parse_teams(
                    tournament_ctx.get("teams"), tournament.id, db_session
                )
            if tournament_data_content.get("stage"):
                parse_stages(
                    tournament_data_content.get("stage"),
                    tournament.id,
                    db_session,
                )
            if tournament_ctx.get("castedMatchesInfo"):
                parse_casted_matches_info(
                    tournament_ctx.get("castedMatchesInfo"),
                    tournament.id,
                    db_session,
                )
            if tournament_ctx.get("castTwitchAccounts"):
                parse_cast_twitch_accounts(
                    tournament_ctx.get("castTwitchAccounts"),
                    tournament.id,
                    db_session,
                )
            if tournament_ctx.get("subCounts"):
                parse_sub_counts(
                    tournament_ctx.get("subCounts"), tournament.id, db_session
                )
            if tournament_ctx.get("bracketProgressionOverrides"):
                parse_bracket_progression_overrides(
                    tournament_ctx.get("bracketProgressionOverrides"),
                    tournament.id,
                    db_session,
                )
            if tournament_ctx.get("bracketProgression"):
                parse_bracket_progression(
                    tournament_ctx.get("bracketProgression"),
                    tournament.id,
                    db_session,
                )
            if tournament_ctx.get("tieBreakerMapPool"):
                parse_tie_breaker_map_pool(
                    tournament_ctx.get("tieBreakerMapPool"),
                    tournament.id,
                    db_session,
                )
            if tournament_ctx.get("toSetMapPool"):
                parse_to_set_map_pool(
                    tournament_ctx.get("toSetMapPool"),
                    tournament.id,
                    db_session,
                )
            # if tournament_ctx.get('participatedUsers'):
            #     parse_participated_users(tournament_ctx.get('participatedUsers'), tournament.id, db_session)

            db_session.commit()
        except Exception as e:
            print(f"Error processing tournament {tournament_id}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("Traceback:")
            print(traceback.format_exc())
            db_session.rollback()
            continue


def parse_stages(stages_data: list, tournament_id: int, db_session):
    """Parse stages and related data."""
    for stage_data in stages_data:
        # Parse stage settings
        stage_settings = None
        if stage_data.get("settings"):
            stage_settings = StageSettings(
                grand_final=stage_data["settings"].get("grandFinal"),
                matches_child_count=stage_data["settings"].get(
                    "matchesChildCount"
                ),
                size=stage_data["settings"].get("size"),
                consolation_final=stage_data["settings"].get(
                    "consolationFinal", False
                ),
                group_count=stage_data["settings"].get("groupCount"),
                round_robin_mode=stage_data["settings"].get("roundRobinMode"),
                swiss_group_count=stage_data["settings"]
                .get("swiss", {})
                .get("groupCount"),
                swiss_round_count=stage_data["settings"]
                .get("swiss", {})
                .get("roundCount"),
            )
            db_session.add(stage_settings)
            db_session.flush()

        # Create stage
        stage = Stage(
            id=stage_data["id"],
            tournament_id=tournament_id,
            name=stage_data.get("name"),
            number=stage_data.get("number"),
            type=stage_data.get("type"),
            created_at=stage_data.get("createdAt"),
            settings_id=stage_settings.id if stage_settings else None,
        )
        db_session.add(stage)

        # Parse groups, rounds, and matches
        for group_data in stage_data.get("groups", []):
            group = Group(
                id=group_data["id"],
                stage_id=stage.id,
                number=group_data.get("number"),
            )
            db_session.add(group)

            # Parse rounds
            for round_data in group_data.get("rounds", []):
                # Create map if exists
                map_id = None
                if round_data.get("maps"):
                    map_obj = Map(
                        count=round_data["maps"].get("count"),
                        type=round_data["maps"].get("type"),
                        pick_ban=round_data["maps"].get("pickBan"),
                    )
                    db_session.add(map_obj)
                    db_session.flush()
                    map_id = map_obj.id

                # Create round
                round_obj = Round(
                    id=round_data["id"],
                    group_id=group.id,
                    stage_id=stage.id,
                    number=round_data.get("number"),
                    maps_id=map_id,
                )
                db_session.add(round_obj)

                # Parse matches
                for match_data in round_data.get("matches", []):
                    match = Match(
                        id=match_data["id"],
                        group_id=group.id,
                        stage_id=stage.id,
                        round_id=round_obj.id,
                        number=match_data.get("number"),
                        status=match_data.get("status"),
                        last_game_finished_at=match_data.get(
                            "lastGameFinishedAt"
                        ),
                        created_at=match_data.get("createdAt"),
                        opponent1_id=match_data.get("opponent1", {}).get("id")
                        if match_data.get("opponent1")
                        else None,
                        opponent1_position=match_data.get("opponent1", {}).get(
                            "position"
                        )
                        if match_data.get("opponent1")
                        else None,
                        opponent1_score=match_data.get("opponent1", {}).get(
                            "score"
                        )
                        if match_data.get("opponent1")
                        else None,
                        opponent1_result=match_data.get("opponent1", {}).get(
                            "result"
                        )
                        if match_data.get("opponent1")
                        else None,
                        opponent1_total_points=match_data.get(
                            "opponent1", {}
                        ).get("totalPoints")
                        if match_data.get("opponent1")
                        else None,
                        opponent2_id=match_data.get("opponent2", {}).get("id")
                        if match_data.get("opponent2")
                        else None,
                        opponent2_position=match_data.get("opponent2", {}).get(
                            "position"
                        )
                        if match_data.get("opponent2")
                        else None,
                        opponent2_score=match_data.get("opponent2", {}).get(
                            "score"
                        )
                        if match_data.get("opponent2")
                        else None,
                        opponent2_result=match_data.get("opponent2", {}).get(
                            "result"
                        )
                        if match_data.get("opponent2")
                        else None,
                        opponent2_total_points=match_data.get(
                            "opponent2", {}
                        ).get("totalPoints")
                        if match_data.get("opponent2")
                        else None,
                    )
                    db_session.add(match)


if __name__ == "__main__":
    # Example usage:
    # Scrape tournaments with IDs from 1000 to 2000 in batches of 50
    scrape_tournaments(
        batch_size=50, start_value=1000, end_value=2000, max_failures=10
    )
