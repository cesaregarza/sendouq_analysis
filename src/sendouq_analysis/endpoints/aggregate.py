from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

import requests
from sqlalchemy.exc import ProgrammingError

from sendouq_analysis.compute import get_matches_metadata
from sendouq_analysis.constants.columns import AGGREGATE, MATCHES
from sendouq_analysis.ingest import create_engine, load_tables
from sendouq_analysis.sql.meta import CurrentSeason, SeasonData
from sendouq_analysis.sql.write import dataframe_to_sql
from sendouq_analysis.transforms import build_match_df, build_player_df
from sendouq_analysis.utils import delete_droplet, get_droplet_id, setup_logging

if TYPE_CHECKING:
    import pandas as pd
    import sqlalchemy as db

logger = logging.getLogger(__name__)


def create_new_aggregate() -> None:
    """Runs the aggregation process"""
    # Set up logging for output to docker logs, remove for production
    setup_logging()

    logger.info("Starting aggregation process")
    logger.info("Creating an engine to connect to the database")
    engine = create_engine()
    logger.info(
        "Pulling metadata from the database, if schema does not exist, "
        "compute all aggregates"
    )


def create_new_aggregate(engine: db.engine.Engine) -> None:
    """Runs the aggregation process

    Args:
        engine (db.engine.Engine): Engine for the database
    """
    logger.info("Starting aggregation process")
    logger.info("Pulling all raw data from the database")
    (
        match_df,
        group_memento_df,
        user_memento_df,
        map_df,
        group_df,
        map_preferences_df,
        weapons_df,
    ) = load_tables()

    logger.info("Building match dataframe")
    match_df = build_match_df(match_df)
    aggregates = get_matches_metadata(match_df)
    current_season = aggregates[MATCHES.SEASON].max()
    past_seasons_df = aggregates.query(f"{MATCHES.SEASON} < @current_season")
    current_season_df = aggregates.query(f"{MATCHES.SEASON} == @current_season")

    logger.info("Writing aggregates to the database")
    write_aggregates(engine, past_seasons_df, current_season_df)
    logger.info("Aggregation process complete")


def write_aggregates(
    engine: db.engine.Engine,
    past_seasons_df: pd.DataFrame,
    current_season_df: pd.DataFrame,
) -> None:
    """Writes the aggregated data to the database

    Args:
        engine (db.engine.Engine): Engine for the database
        past_seasons_df (pd.DataFrame): DataFrame of past season aggregates
        current_season_df (pd.DataFrame): DataFrame of current season aggregates
    """
    logger.info("Writing past season aggregates to the database")
    dataframe_to_sql(past_seasons_df, SeasonData, engine)
    logger.info("Writing current season aggregates to the database")
    dataframe_to_sql(current_season_df, CurrentSeason, engine)
    logger.info("Aggregates written to the database")
