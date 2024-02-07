from sqlalchemy import (
    TIMESTAMP,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Double,
    Index,
    Integer,
    Text,
)
from sqlalchemy.ext.declarative import declarative_base

from sendouq_analysis.constants.table_names import (
    AGGREGATE_SCHEMA,
    AGGREGATE_SEASON_CURRENT,
    AGGREGATE_SEASON_PAST,
)

Base = declarative_base()


class SeasonData(Base):
    __tablename__ = AGGREGATE_SEASON_PAST
    __table_args__ = {"schema": AGGREGATE_SCHEMA}
    season = Column(Integer, primary_key=True)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    start_match_id = Column(BigInteger)
    end_match_id = Column(BigInteger)
    num_matches = Column(Integer)
    lognorm_shape = Column(Double)
    lognorm_location = Column(Double)
    lognorm_scale = Column(Double)


class CurrentSeason(Base):
    __tablename__ = AGGREGATE_SEASON_CURRENT
    __table_args__ = {"schema": AGGREGATE_SCHEMA}
    season = Column(Integer, primary_key=True)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    start_match_id = Column(BigInteger)
    latest_match_id = Column(BigInteger)


class PlayerStats(Base):
    __tablename__ = "player_stats"
    __table_args__ = (
        Index("idx_season", "season"),
        Index("idx_user_id", "user_id"),
        Index("idx_match_id", "match_id"),
        Index("idx_group_id", "group_id"),
        Index("idx_sp", "sp"),
        {"schema": AGGREGATE_SCHEMA},
    )
    id = Column(Integer, primary_key=True)
    season = Column(Integer, nullable=False, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    skill_ordinal = Column(Double)
    skill_tier_name = Column(Text)
    skill_tier_is_plus = Column(Boolean)
    skill_approximate = Column(Boolean)
    match_id = Column(BigInteger, nullable=False, index=True)
    group_id = Column(BigInteger, nullable=False, index=True)
    skill_difference_calculated = Column(Boolean)
    skill_difference_sp_diff = Column(Double)
    skill_difference_matches_count = Column(Double)
    skill_difference_matches_count_needed = Column(Double)
    skill_difference_new_sp = Column(Double)
    plus_tier = Column(Double)
    skill = Column(Text)
    created_at = Column(TIMESTAMP, nullable=False)
    reported_at = Column(TIMESTAMP, nullable=False)
    winner = Column(Text, nullable=False)
    discord_name = Column(Text, nullable=False)
    in_game_name = Column(Text)
    team = Column(Text, nullable=False)
    enemy_group_id = Column(BigInteger, nullable=False)
    is_winner = Column(Boolean, nullable=False)
    sp = Column(Double, index=True)
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
    cumulative_matches = Column(Integer, nullable=False)
