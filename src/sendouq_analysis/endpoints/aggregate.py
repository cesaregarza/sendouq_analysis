from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

import requests
from sqlalchemy import inspect
from sqlalchemy.exc import ProgrammingError

from sendouq_analysis.compute import get_matches_metadata
from sendouq_analysis.constants.columns import AGGREGATE, MATCHES
from sendouq_analysis.ingest import create_engine, load_tables
from sendouq_analysis.sql.meta import CurrentSeason, PlayerStats, SeasonData
from sendouq_analysis.sql.read_write import dataframe_to_sql
from sendouq_analysis.transforms import build_match_df, build_player_df
from sendouq_analysis.utils import delete_droplet, get_droplet_id, setup_logging

if TYPE_CHECKING:
    import pandas as pd
    import sqlalchemy as db

logger = logging.getLogger(__name__)


def new_aggregation() -> None:
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
    try:
        create_new_aggregate(engine)
    except ProgrammingError as e:
        logger.error("Error creating new aggregate, exiting")
        logger.error(e)
        raise e


def create_new_aggregate(engine: db.engine.Engine | None = None) -> None:
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
    ) = load_tables(engine)

    logger.info("Building match dataframe")
    match_df = build_match_df(match_df)
    aggregates = get_matches_metadata(match_df)
    current_season = aggregates[MATCHES.SEASON].max()
    past_seasons_df = aggregates.query(f"{MATCHES.SEASON} < @current_season")
    current_season_df = (
        aggregates.query(f"{MATCHES.SEASON} == @current_season")
        .rename(columns={AGGREGATE.END_MATCH_ID: AGGREGATE.LATEST_MATCH_ID})
        .drop(columns=[AGGREGATE.NUM_MATCHES])
    )

    logger.info("Writing aggregates to the database")
    write_aggregates(
        engine,
        past_seasons_df,
        current_season_df,
        match_df,
        user_memento_df,
        group_df,
    )
    logger.info("Aggregation process complete")


def write_aggregates(
    engine: db.engine.Engine,
    past_seasons_df: pd.DataFrame,
    current_season_df: pd.DataFrame,
    match_df: pd.DataFrame,
    user_memento_df: pd.DataFrame,
    group_df: pd.DataFrame,
) -> None:
    """Writes the aggregated data to the database

    Args:
        engine (db.engine.Engine): Engine for the database
        past_seasons_df (pd.DataFrame): DataFrame of past season aggregates
        current_season_df (pd.DataFrame): DataFrame of current season aggregates
        match_df (pd.DataFrame): DataFrame of matches
        user_memento_df (pd.DataFrame): DataFrame of user mementos
        group_df (pd.DataFrame): DataFrame of groups
    """
    past_seasons_cutoffs = (
        past_seasons_df.sort_values(by=MATCHES.SEASON, ascending=True)
        .loc[:, [AGGREGATE.START_MATCH_ID, AGGREGATE.END_MATCH_ID]]
        .values.tolist()
    )
    # Delete all player stats data
    inspector = inspect(engine)
    table_name = PlayerStats.__tablename__
    schema = PlayerStats.__table_args__[-1]["schema"]
    logger.info("Deleting all player stats data")
    if inspector.has_table(table_name, schema=schema):
        PlayerStats.__table__.drop(engine)
    # Write player stats data
    logger.info("Writing player stats data to the database")
    PlayerStats.__table__.create(engine)

    for i, (start, end) in enumerate(past_seasons_cutoffs):
        if i == 0:
            logger.info("Season 0 has no lognormal parameters")
            continue

        logger.info(f"Building player stats for season %s", i)
        season_df = match_df.loc[
            (match_df[MATCHES.MATCH_ID] >= start)
            & (match_df[MATCHES.MATCH_ID] <= end)
        ]

        player_df, lognorm_params = build_player_df(
            season_df,
            user_memento_df.loc[
                user_memento_df[MATCHES.MATCH_ID].isin(
                    season_df[MATCHES.MATCH_ID]
                )
            ],
            group_df,
            return_lognorm_params=True,
        )
        player_df[MATCHES.SEASON] = i
        logger.info("Writing player stats to the database")
        player_df.to_sql(
            PlayerStats.__tablename__,
            engine,
            if_exists="append",
            index=False,
            schema=PlayerStats.__table_args__[-1]["schema"],
        )
        row_idx = past_seasons_df.index[i]
        past_seasons_df.loc[
            row_idx,
            [
                AGGREGATE.LOGNORMAL_SHAPE,
                AGGREGATE.LOGNORMAL_LOCATION,
                AGGREGATE.LOGNORMAL_SCALE,
            ],
        ] = lognorm_params

    logger.info("Writing past season aggregates to the database")
    dataframe_to_sql(past_seasons_df, SeasonData, engine, replace=True)
    logger.info("Writing current season aggregates to the database")
    dataframe_to_sql(current_season_df, CurrentSeason, engine, replace=True)
    logger.info("Aggregates written to the database")
