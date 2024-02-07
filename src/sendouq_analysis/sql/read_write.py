from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from sqlalchemy import inspect
from sqlalchemy.orm import sessionmaker

if TYPE_CHECKING:
    import sqlalchemy as db

    from sendouq_analysis.sql.meta import Base


def dataframe_to_sql(
    df: pd.DataFrame,
    table: Base,
    engine: db.engine.Engine,
    replace: bool = False,
) -> None:
    """Writes a DataFrame to a SQL table using the defined schema.

    Args:
        df (pd.DataFrame): The DataFrame to write to the database
        table (Base): The SQLAlchemy Base class for the table
        engine (db.engine.Engine): The engine for the database
        replace (bool): Whether to replace the table if it exists. Defaults to
            False.
    """
    Session = sessionmaker(bind=engine)
    session = Session()

    # Create table if it doesn't exist
    inspector = inspect(engine)
    table_name = table.__tablename__
    try:
        schema = table.__table_args__["schema"]
    except TypeError:
        try:
            schema = table.__table_args__[-1]["schema"]
        except TypeError:
            schema = None

    if not inspector.has_table(table_name, schema=schema):
        table.__table__.create(engine)
    elif replace:
        table.__table__.drop(engine)
        table.__table__.create(engine)

    data = df.to_dict(orient="records")
    for record in data:
        row = table(**record)
        session.add(row)

    try:
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def read_table(table: Base, engine: db.engine.Engine) -> pd.DataFrame:
    """Reads a table from the database into a DataFrame.

    Args:
        table (Base): The SQLAlchemy Base class for the table
        engine (db.engine.Engine): The engine for the database

    Returns:
        pd.DataFrame: The DataFrame containing the table data
    """
    try:
        return pd.read_sql_table(
            table.__tablename__, engine, schema=table.__table_args__["schema"]
        )
    except TypeError:
        return pd.read_sql_table(
            table.__tablename__,
            engine,
            schema=table.__table_args__[-1]["schema"],
        )
