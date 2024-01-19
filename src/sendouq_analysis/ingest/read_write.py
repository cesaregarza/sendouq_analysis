import logging
import os

import pandas as pd
import sqlalchemy as db
from sqlalchemy.exc import ProgrammingError

from sendouq_analysis.constants import ENV_VARS, TABLE_NAMES

DATABASE_URL_FORMAT = (
    "postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
)

logger = logging.getLogger(__name__)


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


def create_schema(engine: db.engine.Engine) -> None:
    """Creates the schema for the database

    Args:
        engine (db.engine.Engine): Engine for the database
    """
    logger.warning("Creating schema")
    with engine.connect() as conn:
        conn.execute(
            db.text(
                f"CREATE SCHEMA IF NOT EXISTS {TABLE_NAMES.SCHEMA} "
                f"AUTHORIZATION {os.getenv(ENV_VARS.POSTGRES_USER)}"
            )
        )


def get_table_columns(
    connection: db.engine.Connection, table_name: str
) -> list[str]:
    """Gets the columns of a table

    Args:
        connection (db.engine.Connection): Connection to the database
        table_name (str): Name of the table

    Returns:
        list[str]: List of column names
    """
    query = f"SELECT * FROM {TABLE_NAMES.SCHEMA}.{table_name} LIMIT 0"
    return list(connection.execute(db.text(query)).keys())


def add_column_if_not_exists(
    conn: db.engine.Connection,
    table_name: str,
    column_name: str,
    df: pd.DataFrame,
) -> None:
    """Adds a column to a database table if it does not already exist.

    Args:
        conn (db.engine.Connection): Connection to the database
        table_name (str): Name of the table to modify
        column_name (str): Name of the column to add
        df (pd.DataFrame): DataFrame containing the column data
    """
    # Infer the data type of the column in pandas DataFrame
    dtype = df[column_name].dtype

    # Convert pandas dtype to SQL type
    if pd.api.types.is_integer_dtype(dtype):
        sql_dtype = "INTEGER"
    elif pd.api.types.is_float_dtype(dtype):
        sql_dtype = "FLOAT"
    elif pd.api.types.is_string_dtype(dtype):
        sql_dtype = "TEXT"
    elif pd.api.types.is_object_dtype(dtype):
        sql_dtype = "TEXT"
    elif pd.api.types.is_bool_dtype(dtype):
        sql_dtype = "BOOLEAN"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        sql_dtype = "TIMESTAMP"
    else:
        sql_dtype = "TEXT"

    # Check if the column already exists in the table
    column_exists_query = f"""
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = '{TABLE_NAMES.SCHEMA}'
            AND table_name = '{table_name}'
            AND column_name = '{column_name}'
        );
    """
    column_exists = conn.execute(db.text(column_exists_query)).scalar()

    # If the column does not exist, add it to the table
    if not column_exists:
        add_column_query = (
            f"ALTER TABLE {TABLE_NAMES.SCHEMA}.{table_name} ADD COLUMN "
            f"{column_name} {sql_dtype};"
        )
        conn.execute(db.text(add_column_query))
        logger.info(f"Added column '{column_name}' to table '{table_name}'.")


def load_latest_match_number(engine: db.engine.Engine) -> int:
    """Loads the latest match number from the database

    Args:
        engine (db.engine.Engine): Engine for the database

    Returns:
        int: The latest match number
    """
    query = (
        f"SELECT MAX(match_id) FROM {TABLE_NAMES.SCHEMA}.{TABLE_NAMES.MATCH}"
    )
    try:
        return pd.read_sql(query, engine).iloc[0, 0]
    # If the table is empty or does not exist, return 1
    except (TypeError, IndexError, ProgrammingError):
        return 0


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
        [
            pd.read_sql_table(table_name, engine)
            for table_name in TABLE_NAMES.ALL_TABLES
        ]
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
    """Writes all tables to the database after ensuring all columns exist.

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
    logger.warning("Writing tables to database")
    name_dataframe: list[tuple[str, pd.DataFrame]] = [
        (TABLE_NAMES.MATCH, match_df),
        (TABLE_NAMES.GROUP_MEMENTO, group_memento_df),
        (TABLE_NAMES.USER_MEMENTO, user_memento_df),
        (TABLE_NAMES.MAP, map_df),
        (TABLE_NAMES.GROUP, group_df),
        (TABLE_NAMES.MAP_PREFERENCES, map_preferences_df),
        (TABLE_NAMES.WEAPONS, weapons_df),
    ]
    with engine.begin() as connection:
        for table_name, df in name_dataframe:
            if df.empty:
                logger.info(f"Skipping empty dataframe for table {table_name}")
                continue
            existing_columns = get_table_columns(connection, table_name)
            missing_columns = set(df.columns) - set(existing_columns)
            for column in missing_columns:
                logger.info(
                    f"Adding missing column {column} to table {table_name}"
                )
                add_column_if_not_exists(connection, table_name, column, df)
            df.to_sql(
                table_name,
                con=connection,
                if_exists="append",
                index=False,
                schema=TABLE_NAMES.SCHEMA,
            )
    logger.info("Finished writing tables to database")
