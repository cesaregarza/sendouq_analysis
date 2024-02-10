from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

import requests
from sqlalchemy import inspect
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import sessionmaker

from sendouq_analysis.compute import get_matches_metadata
from sendouq_analysis.constants.columns import AGGREGATE, MATCHES
from sendouq_analysis.ingest import create_engine, load_tables
from sendouq_analysis.sql.meta import (
    CurrentSeason,
    LatestPlayerStats,
    PlayerStats,
    SeasonData,
)
from sendouq_analysis.sql.raw import (
    Group,
    GroupMemento,
    Map,
    MapPreferences,
    Match,
    UserMemento,
    Weapons,
)
from sendouq_analysis.sql.read_write import dataframe_to_sql, read_table
from sendouq_analysis.sql.statements import create_latest_player_stats
from sendouq_analysis.transforms import build_match_df, build_player_df
from sendouq_analysis.utils import delete_droplet, get_droplet_id, setup_logging

if TYPE_CHECKING:
    import pandas as pd
    import sqlalchemy as db

logger = logging.getLogger(__name__)


def run() -> None:
    """Runs the aggregation process"""
    # Set up logging for output to docker logs, remove for production
    setup_logging()

    if os.getenv("NEW_AGGREGATION") == "True":
        new_aggregation()
    else:
        update_aggregation()


def new_aggregation() -> None:
    """Runs the aggregation process"""

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
    seasons_cutoffs = (
        past_seasons_df.sort_values(by=MATCHES.SEASON, ascending=True)
        .loc[:, [AGGREGATE.START_MATCH_ID, AGGREGATE.END_MATCH_ID]]
        .values.tolist()
    )
    # Add current season cutoffs
    seasons_cutoffs = [
        *seasons_cutoffs,
        (seasons_cutoffs[-1][1] + 1, int(1e9)),
    ]
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

    for i, (start, end) in enumerate(seasons_cutoffs):
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
        del player_df
        if end == int(1e9):
            logger.info("Current season, skipping aggregate writing")
            continue

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
    logger.info("Dropping existing latest player stats table")
    if inspector.has_table(LatestPlayerStats.__tablename__, schema=schema):
        LatestPlayerStats.__table__.drop(engine)
    logger.info("Creating latest player stats table")

    Session = sessionmaker(bind=engine)
    session = Session()
    session.execute(create_latest_player_stats)
    session.commit()
    session.close()

    logger.info("Aggregates written to the database")


def update_aggregation() -> None:
    logger.info("Starting update aggregation process")
    logger.info("Creating an engine to connect to the database")
    engine = create_engine()
    logger.info("Pulling metadata from the database")
    try:
        update_existing_aggregate(engine)
    except ProgrammingError as e:
        logger.error("Error updating existing aggregate, exiting")
        logger.error(e)
        raise e


def update_existing_aggregate(engine: db.engine.Engine | None = None) -> None:
    """Updates the existing aggregate data

    Args:
        engine (db.engine.Engine): Engine for the database
    """
    logger.info("Pulling all raw data from the database")

    current_season = read_table(CurrentSeason, engine)
    latest_match = current_season[AGGREGATE.LATEST_MATCH_ID].iloc[0]
    logger.info(f"Latest match ID: {latest_match}")
    logger.info("Pulling all rows after the latest match")
    Session = sessionmaker(bind=engine)
    session = Session()

    match_statement = session.query(Match).filter(Match.match_id > latest_match)
    match_df = pd.read_sql(match_statement.statement, engine)
