from sqlalchemy import Boolean, Column, Float, Integer, String, Table, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


# Users Table
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String, nullable=False)
    discord_id = Column(String, nullable=True)
    discord_avatar = Column(String, nullable=True)
    custom_url = Column(String, nullable=True)
    chat_name_color = Column(String, nullable=True)


# Organizations Table
class Organization(Base):
    __tablename__ = "organizations"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    slug = Column(String, nullable=False, unique=True)
    avatar_url = Column(String, nullable=True)


# Tournament Settings Table
class TournamentSettings(Base):
    __tablename__ = "tournament_settings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    bracket_progression_overrides = Column(Text, nullable=True)
    teams_per_group = Column(Integer, nullable=True)
    third_place_match = Column(Boolean, nullable=False, default=False)
    is_ranked = Column(Boolean, nullable=False, default=False)
    enable_no_screen_toggle = Column(Boolean, nullable=False, default=False)
    autonomous_subs = Column(Boolean, nullable=False, default=False)
    reg_closes_at = Column(Integer, nullable=True)
    auto_check_in_all = Column(Boolean, nullable=False, default=False)
    deadlines = Column(Text, nullable=True)
    is_invitational = Column(Boolean, nullable=False, default=False)
    swiss_group_count = Column(Integer, nullable=True)
    swiss_round_count = Column(Integer, nullable=True)


# Tournaments Table
class Tournament(Base):
    __tablename__ = "tournaments"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    start_time = Column(Integer, nullable=False)
    discord_url = Column(String, nullable=True)
    tags = Column(Text, nullable=True)
    rules = Column(Text, nullable=True)
    map_picking_style = Column(String, nullable=True)
    logo_src = Column(String, nullable=True)
    logo_url = Column(String, nullable=True)
    logo_validated_at = Column(Integer, nullable=True)
    event_id = Column(Integer, nullable=True)
    is_finalized = Column(Boolean, nullable=False, default=False)
    settings_id = Column(Integer, nullable=True)
    organization_id = Column(Integer, nullable=True)
    author_id = Column(Integer, nullable=True)


# Staff Table
class Staff(Base):
    __tablename__ = "staff"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tournament_id = Column(Integer, nullable=False)
    user_id = Column(Integer, nullable=False)
    role = Column(String, nullable=False)


# Teams Table
class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True)
    tournament_id = Column(Integer, nullable=False)
    name = Column(String, nullable=False)
    seed = Column(Integer, nullable=True)
    prefers_not_to_host = Column(Boolean, nullable=False, default=False)
    no_screen = Column(Boolean, nullable=False, default=False)
    dropped_out = Column(Boolean, nullable=False, default=False)
    invite_code = Column(String, nullable=True)
    created_at = Column(Integer, nullable=False)
    active_roster_user_ids = Column(Text, nullable=True)
    starting_bracket_idx = Column(Integer, nullable=True)
    pickup_avatar_url = Column(String, nullable=True)
    avg_seeding_skill_ordinal = Column(Float, nullable=True)


# Team Members Table
class TeamMember(Base):
    __tablename__ = "team_members"

    id = Column(Integer, primary_key=True, autoincrement=True)
    team_id = Column(Integer, nullable=False)
    user_id = Column(Integer, nullable=False)
    username = Column(String, nullable=False)
    discord_id = Column(String, nullable=True)
    discord_avatar = Column(String, nullable=True)
    custom_url = Column(String, nullable=True)
    country = Column(String, nullable=True)
    twitch = Column(String, nullable=True)
    is_owner = Column(Boolean, nullable=False, default=False)
    created_at = Column(Integer, nullable=False)
    in_game_name = Column(String, nullable=True)


# Check-Ins Table
class CheckIn(Base):
    __tablename__ = "check_ins"

    id = Column(Integer, primary_key=True, autoincrement=True)
    team_id = Column(Integer, nullable=False)
    bracket_idx = Column(Integer, nullable=True)
    checked_in_at = Column(Integer, nullable=False)
    is_check_out = Column(Boolean, nullable=False, default=False)


# Stages Table
class Stage(Base):
    __tablename__ = "stages"

    id = Column(Integer, primary_key=True)
    tournament_id = Column(Integer, nullable=False)
    name = Column(String, nullable=False)
    number = Column(Integer, nullable=False)
    type = Column(String, nullable=False)
    created_at = Column(Integer, nullable=True)
    settings_id = Column(Integer, nullable=True)


# Stage Settings Table
class StageSettings(Base):
    __tablename__ = "stage_settings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    grand_final = Column(String, nullable=True)
    matches_child_count = Column(Integer, nullable=True)
    size = Column(Integer, nullable=True)
    consolation_final = Column(Boolean, nullable=True, default=False)
    group_count = Column(Integer, nullable=True)
    round_robin_mode = Column(String, nullable=True)
    swiss_group_count = Column(Integer, nullable=True)
    swiss_round_count = Column(Integer, nullable=True)


# Groups Table
class Group(Base):
    __tablename__ = "groups"

    id = Column(Integer, primary_key=True)
    stage_id = Column(Integer, nullable=False)
    number = Column(Integer, nullable=False)


# Rounds Table
class Round(Base):
    __tablename__ = "rounds"

    id = Column(Integer, primary_key=True)
    group_id = Column(Integer, nullable=False)
    stage_id = Column(Integer, nullable=False)
    number = Column(Integer, nullable=False)
    maps_id = Column(Integer, nullable=True)


# Maps Table
class Map(Base):
    __tablename__ = "maps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    count = Column(Integer, nullable=False)
    type = Column(String, nullable=False)
    pick_ban = Column(String, nullable=True)


# Matches Table
class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True)
    group_id = Column(Integer, nullable=False)
    stage_id = Column(Integer, nullable=False)
    round_id = Column(Integer, nullable=False)
    number = Column(Integer, nullable=False)
    status = Column(Integer, nullable=False)
    last_game_finished_at = Column(Integer, nullable=True)
    created_at = Column(Integer, nullable=True)
    opponent1_id = Column(Integer, nullable=True)
    opponent1_position = Column(Integer, nullable=True)
    opponent1_score = Column(Integer, nullable=True)
    opponent1_result = Column(String, nullable=True)
    opponent1_total_points = Column(Integer, nullable=True)
    opponent2_id = Column(Integer, nullable=True)
    opponent2_position = Column(Integer, nullable=True)
    opponent2_score = Column(Integer, nullable=True)
    opponent2_result = Column(String, nullable=True)
    opponent2_total_points = Column(Integer, nullable=True)


# Streams Table
class Stream(Base):
    __tablename__ = "streams"

    tournament_id = Column(Integer, primary_key=True)
    streams_count = Column(Integer, nullable=False)


# Bracket Progression Table
class BracketProgression(Base):
    __tablename__ = "bracket_progression"

    id = Column(Integer, primary_key=True, autoincrement=True)
    settings_id = Column(Integer, nullable=False)
    type = Column(String, nullable=False)
    name = Column(String, nullable=False)


# Bracket Progression Sources Table
class BracketProgressionSource(Base):
    __tablename__ = "bracket_progression_sources"

    id = Column(Integer, primary_key=True, autoincrement=True)
    bracket_progression_id = Column(Integer, nullable=False)
    bracket_idx = Column(Integer, nullable=False)
    placement = Column(Integer, nullable=False)


# Bracket Progression Overrides Table
class BracketProgressionOverride(Base):
    __tablename__ = "bracket_progression_overrides"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tournament_id = Column(Integer, nullable=False)
    override_data = Column(Text, nullable=True)


# Casted Matches Info Table
class CastedMatchesInfo(Base):
    __tablename__ = "casted_matches_info"

    tournament_id = Column(Integer, primary_key=True)
    locked_matches = Column(Text, nullable=True)


# Casted Matches Table
class CastedMatch(Base):
    __tablename__ = "casted_matches"

    id = Column(Integer, primary_key=True, autoincrement=True)
    casted_matches_info_id = Column(Integer, nullable=False)
    twitch_account = Column(String, nullable=False)
    match_id = Column(Integer, nullable=False)


# Locked Matches Table
class LockedMatch(Base):
    __tablename__ = "locked_matches"

    casted_matches_info_id = Column(Integer, primary_key=True)
    match_id = Column(Integer, primary_key=True)


# Sub Counts Table
class SubCount(Base):
    __tablename__ = "sub_counts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tournament_id = Column(Integer, nullable=False)
    visibility = Column(String, nullable=False)
    count = Column(Integer, nullable=False)


# Map Pools Table
class MapPool(Base):
    __tablename__ = "map_pools"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tournament_id = Column(Integer, nullable=False)
    stage_id = Column(Integer, nullable=False)
    team_id = Column(Integer, nullable=True)
    mode = Column(String, nullable=False)
    type = Column(String, nullable=False)


# Cast Twitch Accounts Table
class CastTwitchAccount(Base):
    __tablename__ = "cast_twitch_accounts"

    tournament_id = Column(Integer, primary_key=True)
    twitch_account = Column(String, nullable=True)
