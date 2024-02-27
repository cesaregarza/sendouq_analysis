import os

import pandas as pd
import sqlalchemy as db

from sendouq_dashboard.constants import ENV_VARS, TABLE_NAMES

DATABASE_URL_FORMAT = (
    "postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
)


def create_engine() -> db.engine.Engine:
    """Creates an engine for the database

    Returns:
        db.engine.Engine: Engine for the database
    """
    user = os.getenv(ENV_VARS.POSTGRES_USER)
    password = os.getenv(ENV_VARS.POSTGRES_PASSWORD)
    host = os.getenv(ENV_VARS.POSTGRES_HOST)
    port = os.getenv(ENV_VARS.POSTGRES_PORT)
    database = os.getenv(ENV_VARS.POSTGRES_DB)
    return db.create_engine(
        DATABASE_URL_FORMAT.format(
            user=user,
            password=password,
            host=host,
            port=port,
            database=database,
        )
    )


def load_raw_tables() -> (
    tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]
):
    """Loads all tables from the database

    Returns:
        tuple:
            - pd.DataFrame: The match data
            - pd.DataFrame: The group memento data
            - pd.DataFrame: The user memento data
            - pd.DataFrame: The map data
            - pd.DataFrame: The group data
            - pd.DataFrame: The map preferences data
            - pd.DataFrame: The weapons data
    """
    engine = create_engine()
    return tuple(
        [
            pd.read_sql_table(table_name, engine, schema=TABLE_NAMES.RAW_SCHEMA)
            for table_name in TABLE_NAMES.ALL_TABLES
        ]
    )


def load_player_stats() -> pd.DataFrame:
    """Loads the player stats table from the database

    Returns:
        pd.DataFrame: The player stats table
    """
    engine = create_engine()
    return pd.read_sql_table(
        TABLE_NAMES.PLAYER_STATS, engine, schema=TABLE_NAMES.AGGREGATE_SCHEMA
    )


def load_latest_player_stats() -> pd.DataFrame:
    """Loads the latest player stats table from the database

    Returns:
        pd.DataFrame: The latest player stats table
    """
    engine = create_engine()
    return pd.read_sql_table(
        TABLE_NAMES.LATEST_PLAYER_STATS,
        engine,
        schema=TABLE_NAMES.AGGREGATE_SCHEMA,
    )


def load_season_data() -> pd.DataFrame:
    """Loads the season data table from the database

    Returns:
        pd.DataFrame: The season data table
    """
    engine = create_engine()
    return pd.read_sql_table(
        TABLE_NAMES.SEASON_DATA, engine, schema=TABLE_NAMES.AGGREGATE_SCHEMA
    )


def load_current_season() -> pd.DataFrame:
    """Loads the current season table from the database

    Returns:
        pd.DataFrame: The current season table
    """
    engine = create_engine()
    return pd.read_sql_table(
        TABLE_NAMES.CURRENT_SEASON, engine, schema=TABLE_NAMES.AGGREGATE_SCHEMA
    )
