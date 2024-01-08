import os

import pandas as pd
import sqlalchemy as db

from sendouq_analysis.constants import ENV_VARS, TABLE_NAMES

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


def load_latest_match_number(engine: db.engine.Engine) -> int:
    """Loads the latest match number from the database

    Args:
        engine (db.engine.Engine): Engine for the database

    Returns:
        int: The latest match number
    """
    query = f"SELECT MAX(match_number) FROM {TABLE_NAMES.MATCHES}"
    return pd.read_sql(query, engine).iloc[0, 0]


def load_tables() -> (
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
        [pd.read_sql_table(table_name, engine) for table_name in TABLE_NAMES]
    )


def write_tables(
    match_df: pd.DataFrame,
    group_memento_df: pd.DataFrame,
    user_memento_df: pd.DataFrame,
    map_df: pd.DataFrame,
    group_df: pd.DataFrame,
    map_preferences_df: pd.DataFrame,
    weapons_df: pd.DataFrame,
    engine: db.engine.Engine,
) -> None:
    """Writes all tables to the database

    Args:
        match_df (pd.DataFrame): The match data
        group_memento_df (pd.DataFrame): The group memento data
        user_memento_df (pd.DataFrame): The user memento data
        map_df (pd.DataFrame): The map data
        group_df (pd.DataFrame): The group data
        map_preferences_df (pd.DataFrame): The map preferences data
        weapons_df (pd.DataFrame): The weapons data
        engine (db.engine.Engine): Engine for the database
    """
    match_df.to_sql(
        TABLE_NAMES.MATCHES, engine, if_exists="append", index=False
    )
    group_memento_df.to_sql(
        TABLE_NAMES.GROUP_MEMENTOS, engine, if_exists="append", index=False
    )
    user_memento_df.to_sql(
        TABLE_NAMES.USER_MEMENTOS, engine, if_exists="append", index=False
    )
    map_df.to_sql(TABLE_NAMES.MAPS, engine, if_exists="append", index=False)
    group_df.to_sql(TABLE_NAMES.GROUPS, engine, if_exists="append", index=False)
    map_preferences_df.to_sql(
        TABLE_NAMES.MAP_PREFERENCES, engine, if_exists="append", index=False
    )
    weapons_df.to_sql(
        TABLE_NAMES.WEAPONS, engine, if_exists="append", index=False
    )
