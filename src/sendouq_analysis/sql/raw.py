from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Double,
    PrimaryKeyConstraint,
    Text,
)
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Match(Base):
    __tablename__ = "match"
    __table_args__ = {"schema": "sendouq_analysis"}
    match_id = Column(BigInteger, primary_key=True)
    alpha_team_id = Column(BigInteger)
    bravo_team_id = Column(BigInteger)
    created_at = Column(BigInteger)
    reported_at = Column(BigInteger)
    reported_by_user_id = Column(BigInteger)
    winner_id = Column(Text)
    winner = Column(Text)


class GroupMemento(Base):
    __tablename__ = "group_memento"
    __table_args__ = (
        PrimaryKeyConstraint("match_id", "group_id"),
        {"schema": "sendouq_analysis"},
    )
    match_id = Column(BigInteger, primary_key=True)
    group_id = Column(Text, primary_key=True)
    tier_name = Column(Text)
    tier_is_plus = Column(Boolean)
    skill_difference_calculated = Column(Boolean)
    skill_difference_matches_count = Column(Double)
    skill_difference_matches_count_needed = Column(Double)
    skill_difference_new_sp = Column(Double)
    skill_difference_old_sp = Column(Double)


class UserMemento(Base):
    __tablename__ = "user_memento"
    __table_args__ = (
        PrimaryKeyConstraint("user_id", "match_id"),
        {"schema": "sendouq_analysis"},
    )
    user_id = Column(BigInteger, primary_key=True)
    skill_ordinal = Column(Double)
    skill_tier_name = Column(Text)
    skill_tier_is_plus = Column(Boolean)
    skill_approximate = Column(Boolean)
    match_id = Column(BigInteger, primary_key=True)
    group_id = Column(BigInteger)
    skill_difference_calculated = Column(Boolean)
    skill_difference_sp_diff = Column(Double)
    skill_difference_matches_count = Column(Double)
    skill_difference_matches_count_needed = Column(Double)
    skill_difference_new_sp = Column(Double)
    plus_tier = Column(Double)
    skill = Column(Text)


class Map(Base):
    __tablename__ = "map"
    __table_args__ = (
        PrimaryKeyConstraint("id", "match_id"),
        {"schema": "sendouq_analysis"},
    )
    id = Column(BigInteger, primary_key=True)
    mode = Column(Text)
    stage_id = Column(BigInteger)
    source = Column(Text)
    winner_group_id = Column(Double)
    match_id = Column(BigInteger, primary_key=True)


class Group(Base):
    __tablename__ = "group"
    __table_args__ = (
        PrimaryKeyConstraint("user_id", "group_id"),
        {"schema": "sendouq_analysis"},
    )

    user_id = Column(BigInteger, primary_key=True)
    discord_name = Column(Text)
    discord_id = Column(Text)
    custom_url = Column(Text)
    role = Column(Text)
    in_game_name = Column(Text)
    vc = Column(Text)
    languages = Column(Text)
    private_note = Column(Text)
    chat_name_color = Column(Text)
    group_id = Column(BigInteger, primary_key=True)
    team = Column(Text)
    plus_tier = Column(Double)


class MapPreferences(Base):
    __tablename__ = "map_preferences"
    __table_args__ = (
        PrimaryKeyConstraint("match_id", "user_id"),
        {"schema": "sendouq_analysis"},
    )
    match_id = Column(BigInteger, primary_key=True)
    user_id = Column(BigInteger, primary_key=True)
    preference = Column(Text)
    map_index = Column(BigInteger)


class Weapons(Base):
    __tablename__ = "weapons"
    __table_args__ = (
        PrimaryKeyConstraint("user_id", "map_index", "match_id"),
        {"schema": "sendouq_analysis"},
    )
    group_match_map_id = Column(Double)
    weapon_spl_id = Column(Double)
    user_id = Column(Double, primary_key=True)
    map_index = Column(Double, primary_key=True)
    match_id = Column(BigInteger, primary_key=True)
