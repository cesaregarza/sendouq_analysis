from __future__ import annotations

from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

from .constants import SCHEMA
from .engine import Base


class Player(Base):
    __tablename__ = "players"
    __table_args__ = ({"schema": SCHEMA},)

    player_id = Column(BigInteger, primary_key=True, autoincrement=True)
    player_uuid = Column(UUID(as_uuid=True), unique=True, nullable=True)
    display_name = Column(String, nullable=False)
    discord_id = Column(String, nullable=True)
    country = Column(String(2), nullable=True)
    created_at = Column(String, nullable=True)  # ISO8601 or server default


class Tournament(Base):
    __tablename__ = "tournaments"
    __table_args__ = (
        Index("ix_tournaments_tags_gin", "tags", postgresql_using="gin"),
        Index("ix_tournaments_meta_gin", "meta", postgresql_using="gin"),
        {"schema": SCHEMA},
    )

    tournament_id = Column(BigInteger, primary_key=True, autoincrement=True)
    # Internal UUID to decouple from provider IDs; populated by importer
    tournament_uuid = Column(UUID(as_uuid=True), unique=True, nullable=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    start_time_ms = Column(BigInteger, nullable=True)
    is_finalized = Column(Boolean, nullable=True)
    is_ranked = Column(Boolean, nullable=True)
    format_hint = Column(String, nullable=True)  # e.g., 'swiss', 'double_elim'
    map_picking_style = Column(String, nullable=True)
    rules = Column(Text, nullable=True)
    tags = Column(JSONB, nullable=True)  # list of strings
    meta = Column(JSONB, nullable=True)  # provider-specific blob
    # Parser-derived counts for parity with filter_ranked_tournaments
    team_count = Column(Integer, nullable=True)
    match_count = Column(Integer, nullable=True)
    stage_count = Column(Integer, nullable=True)
    group_count = Column(Integer, nullable=True)
    round_count = Column(Integer, nullable=True)
    participated_users_count = Column(Integer, nullable=True)


class Stage(Base):
    __tablename__ = "stages"
    __table_args__ = (
        Index("ix_stages_tournament_id", "tournament_id"),
        Index("ix_stages_settings_gin", "settings", postgresql_using="gin"),
        {"schema": SCHEMA},
    )

    stage_id = Column(BigInteger, primary_key=True, autoincrement=True)
    stage_uuid = Column(UUID(as_uuid=True), unique=True, nullable=True)
    tournament_id = Column(
        BigInteger,
        ForeignKey(f"{SCHEMA}.tournaments.tournament_id"),
        nullable=False,
    )
    name = Column(String, nullable=True)
    number = Column(Integer, nullable=True)
    type = Column(String, nullable=True)
    settings = Column(JSONB, nullable=True)


class Group(Base):
    __tablename__ = "groups"
    __table_args__ = (
        Index("ix_groups_stage_id", "stage_id"),
        {"schema": SCHEMA},
    )

    group_id = Column(BigInteger, primary_key=True, autoincrement=True)
    group_uuid = Column(UUID(as_uuid=True), unique=True, nullable=True)
    stage_id = Column(
        BigInteger, ForeignKey(f"{SCHEMA}.stages.stage_id"), nullable=False
    )
    number = Column(Integer, nullable=True)


class Round(Base):
    __tablename__ = "rounds"
    __table_args__ = (
        Index("ix_rounds_stage_id", "stage_id"),
        Index("ix_rounds_group_id", "group_id"),
        {"schema": SCHEMA},
    )

    round_id = Column(BigInteger, primary_key=True, autoincrement=True)
    round_uuid = Column(UUID(as_uuid=True), unique=True, nullable=True)
    stage_id = Column(
        BigInteger, ForeignKey(f"{SCHEMA}.stages.stage_id"), nullable=False
    )
    group_id = Column(
        BigInteger, ForeignKey(f"{SCHEMA}.groups.group_id"), nullable=True
    )
    number = Column(Integer, nullable=True)
    maps_count = Column(Integer, nullable=True)
    maps_type = Column(String, nullable=True)


class TournamentTeam(Base):
    __tablename__ = "tournament_teams"
    __table_args__ = (
        Index("ix_teams_tournament_id", "tournament_id"),
        {"schema": SCHEMA},
    )

    team_id = Column(BigInteger, primary_key=True, autoincrement=True)
    team_uuid = Column(UUID(as_uuid=True), unique=True, nullable=True)
    tournament_id = Column(
        BigInteger,
        ForeignKey(f"{SCHEMA}.tournaments.tournament_id"),
        nullable=False,
    )
    name = Column(String, nullable=True)
    seed = Column(Integer, nullable=True)
    prefers_not_to_host = Column(Boolean, nullable=True)
    no_screen = Column(Boolean, nullable=True)
    dropped_out = Column(Boolean, nullable=True)
    created_at_ms = Column(BigInteger, nullable=True)


class RosterEntry(Base):
    __tablename__ = "roster_entries"
    __table_args__ = (
        UniqueConstraint(
            "tournament_id", "team_id", "player_id", name="uq_roster_unique"
        ),
        Index("ix_roster_tournament_team", "tournament_id", "team_id"),
        Index("ix_roster_tournament_player", "tournament_id", "player_id"),
        {"schema": SCHEMA},
    )

    roster_id = Column(BigInteger, primary_key=True, autoincrement=True)
    tournament_id = Column(
        BigInteger,
        ForeignKey(f"{SCHEMA}.tournaments.tournament_id"),
        nullable=False,
    )
    team_id = Column(
        BigInteger,
        ForeignKey(f"{SCHEMA}.tournament_teams.team_id"),
        nullable=False,
    )
    player_id = Column(
        BigInteger, ForeignKey(f"{SCHEMA}.players.player_id"), nullable=False
    )
    is_owner = Column(Boolean, nullable=True)
    joined_at_ms = Column(BigInteger, nullable=True)
    left_at_ms = Column(BigInteger, nullable=True)


class Match(Base):
    __tablename__ = "matches"
    __table_args__ = (
        Index("ix_matches_tournament", "tournament_id"),
        Index("ix_matches_winner", "winner_team_id"),
        Index("ix_matches_loser", "loser_team_id"),
        {"schema": SCHEMA},
    )

    match_id = Column(BigInteger, primary_key=True, autoincrement=True)
    match_uuid = Column(UUID(as_uuid=True), unique=True, nullable=True)
    tournament_id = Column(
        BigInteger,
        ForeignKey(f"{SCHEMA}.tournaments.tournament_id"),
        nullable=False,
    )
    stage_id = Column(
        BigInteger, ForeignKey(f"{SCHEMA}.stages.stage_id"), nullable=True
    )
    group_id = Column(
        BigInteger, ForeignKey(f"{SCHEMA}.groups.group_id"), nullable=True
    )
    round_id = Column(
        BigInteger, ForeignKey(f"{SCHEMA}.rounds.round_id"), nullable=True
    )

    number = Column(Integer, nullable=True)
    status = Column(String, nullable=True)
    created_at_ms = Column(BigInteger, nullable=True)
    last_game_finished_at_ms = Column(BigInteger, nullable=True)

    team1_id = Column(
        BigInteger,
        ForeignKey(f"{SCHEMA}.tournament_teams.team_id"),
        nullable=True,
    )
    team1_position = Column(Integer, nullable=True)
    team1_score = Column(Integer, nullable=True)

    team2_id = Column(
        BigInteger,
        ForeignKey(f"{SCHEMA}.tournament_teams.team_id"),
        nullable=True,
    )
    team2_position = Column(Integer, nullable=True)
    team2_score = Column(Integer, nullable=True)

    winner_team_id = Column(
        BigInteger,
        ForeignKey(f"{SCHEMA}.tournament_teams.team_id"),
        nullable=True,
    )
    loser_team_id = Column(
        BigInteger,
        ForeignKey(f"{SCHEMA}.tournament_teams.team_id"),
        nullable=True,
    )

    is_bye = Column(Boolean, nullable=True)


class ExternalID(Base):
    __tablename__ = "external_ids"
    __table_args__ = (
        UniqueConstraint(
            "provider", "entity_type", "external_id", name="uq_external_id"
        ),
        Index("ix_external_entity", "entity_type", "internal_id"),
        Index("ix_external_entity_uuid", "entity_type", "internal_uuid"),
        CheckConstraint(
            "entity_type in ('player','tournament','team','match','stage','group','round')",
            name="ck_entity_type",
        ),
        {"schema": SCHEMA},
    )

    alias_id = Column(BigInteger, primary_key=True, autoincrement=True)
    entity_type = Column(String, nullable=False)
    internal_id = Column(BigInteger, nullable=False)
    # Optional universal UUID linking to internal entity when available
    internal_uuid = Column(UUID(as_uuid=True), nullable=True)
    provider = Column(
        String, nullable=False
    )  # 'sendou','startgg','battlefy','challonge','manual','sheet'
    external_id = Column(String, nullable=False)


class PlayerRanking(Base):
    __tablename__ = "player_rankings"
    __table_args__ = (
        UniqueConstraint(
            "player_id",
            "calculated_at_ms",
            "build_version",
            name="uq_player_rankings_run",
        ),
        Index("ix_player_rankings_player", "player_id"),
        Index("ix_player_rankings_calculated", "calculated_at_ms"),
        Index(
            "ix_player_rankings_build_ts", "build_version", "calculated_at_ms"
        ),
        {"schema": SCHEMA},
    )

    ranking_id = Column(BigInteger, primary_key=True, autoincrement=True)
    player_id = Column(
        BigInteger, ForeignKey(f"{SCHEMA}.players.player_id"), nullable=False
    )
    # Metrics
    player_rank = Column(Float, nullable=True)
    score = Column(Float, nullable=True)
    win_pr = Column(Float, nullable=True)
    loss_pr = Column(Float, nullable=True)
    exposure = Column(Float, nullable=True)

    # Run metadata
    calculated_at_ms = Column(BigInteger, nullable=False)
    build_version = Column(String(64), nullable=False)
