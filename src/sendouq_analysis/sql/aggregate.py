from sqlalchemy import BigInteger, Boolean, Column, DateTime, Double, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class PlayerDataBase(Base):
    __abstract__ = True
    user_id = Column(BigInteger, nullable=False)
    skill_ordinal = Column(Double)
    skill_tier_name = Column(Text)
    skill_tier_is_plus = Column(Boolean)
    skill_approximate = Column(Boolean)
    match_id = Column(BigInteger, nullable=False)
    group_id = Column(BigInteger, nullable=False)
    skill_difference_calculated = Column(Boolean)
    skill_difference_sp_diff = Column(Double)
    skill_difference_matches_count = Column(Double)
    skill_difference_matches_count_needed = Column(Double)
    skill_difference_new_sp = Column(Double)
    plus_tier = Column(Double)
    skill = Column(Text)
    created_at = Column(DateTime, nullable=False)
    reported_at = Column(DateTime, nullable=False)
    winner = Column(Text, nullable=False)
    discord_name = Column(Text, nullable=False)
    in_game_name = Column(Text)
    team = Column(Text, nullable=False)
    enemy_group_id = Column(BigInteger, nullable=False)
    is_winner = Column(Boolean, nullable=False)
    sp = Column(Double)
    created_at_dt = Column(DateTime, nullable=False)
    reported_at_dt = Column(DateTime, nullable=False)
    after_sp = Column(Double)
    sp_logz = Column(Double)
    after_sp_logz = Column(Double)
    sp_diff_logz = Column(Double)
    sp_logz_sum = Column(Double, nullable=False)
    sp_logz_std = Column(Double)
    sp_diff_logz_sum = Column(Double, nullable=False)
    sp_diff_logz_std = Column(Double)
    enemy_sp_logz_sum = Column(Double, nullable=False)
    enemy_sp_logz_std = Column(Double)
    enemy_sp_diff_logz_sum = Column(Double, nullable=False)
    enemy_sp_diff_logz_std = Column(Double)
    teammate_sp_logz_diff = Column(Double)
    enemy_sp_logz_diff = Column(Double)
    count_7D = Column(Double, nullable=False)
    won_matches_7D = Column(Double, nullable=False)
    won_matches_prior_7D = Column(Double, nullable=False)
    count_24H = Column(Double, nullable=False)
    won_matches_24H = Column(Double, nullable=False)
    won_matches_prior_24H = Column(Double, nullable=False)
    after_sp_std_7 = Column(Double)
    after_sp_logz_std_7 = Column(Double)
    teammate_sp_logz_diff_std_7 = Column(Double)
    enemy_sp_logz_diff_std_7 = Column(Double)
    has_calculated_7 = Column(Boolean, nullable=False)
    cumulative_matches = Column(BigInteger, nullable=False)


class PlayerPast(PlayerDataBase):
    __tablename__ = "player_past"
    __table_args__ = {"schema": "sendouq_aggregate"}


class PlayerCurrent(PlayerDataBase):
    __tablename__ = "player_current"
    __table_args__ = {"schema": "sendouq_aggregate"}


class MatchAggregate(Base):
    __tablename__ = "match_aggregate"
    match_id = Column(BigInteger, primary_key=True, nullable=False)
    alpha_team_id = Column(BigInteger, nullable=False)
    bravo_team_id = Column(BigInteger, nullable=False)
    created_at = Column(DateTime, nullable=False)
    reported_at = Column(DateTime, nullable=False)
    reported_by_user_id = Column(BigInteger, nullable=False)
    winner_id = Column(Text, nullable=False)
    winner = Column(Text, nullable=False)
    season = Column(BigInteger, nullable=False)
    time_slot = Column(Double, nullable=False)
