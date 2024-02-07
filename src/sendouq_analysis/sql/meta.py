from sqlalchemy import BigInteger, Column, DateTime, Double, Integer
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
