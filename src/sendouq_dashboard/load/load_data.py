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
