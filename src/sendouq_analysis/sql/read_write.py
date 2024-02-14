from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from sqlalchemy import inspect
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import sessionmaker

if TYPE_CHECKING:
    from typing import Callable, Iterable, TypeAlias

    import sqlalchemy as db
    from pandas.io.sql import SQLTable
    from sqlalchemy.orm import DeclarativeBase

    MethodSignature: TypeAlias = Callable[
        [
            SQLTable,
            db.engine.Engine | db.engine.Connection,
            list[str],
            Iterable,
        ],
        None,
    ]


def insert_on_conflict_do_nothing(
    table: SQLTable,
    conn: db.engine.Engine | db.engine.Connection,
    keys: list[str],
    data_iter: Iterable,
) -> None:
    """Execute SQL statement inserting data. If a conflict occurs, do nothing.

    Args:
        table (pandas.io.sql.SQLTable): The SQL table to insert data into.
        conn (sqlalchemy.engine.Engine or sqlalchemy.engine.Connection): The
            database connection.
        keys (list of str): The column names.
        data_iter (Iterable): An iterable that iterates over the values to be
        inserted.
    """
    stmt = insert(table.table).values(list(data_iter))
    stmt = stmt.on_conflict_do_nothing()
    conn.execute(stmt)


def insert_on_conflict_do_update(
    base_table: DeclarativeBase,
) -> MethodSignature:
    """
    Returns a function that executes an SQL statement to insert data into a table.
    If a conflict occurs, it updates the existing row.

    Args:
        base_table (DeclarativeBase): The SQLAlchemy DeclarativeBase object representing the base table.

    Returns:
        MethodSignature: A function that takes the following parameters:
            - table (SQLTable): The SQLAlchemy Table object representing the table.
            - conn (db.engine.Engine | db.engine.Connection): The database connection.
            - keys (list[str]): The column names.
            - data_iter (Iterable): An iterable that iterates over the values to be inserted.

    Example usage:
        insert_function = insert_on_conflict_do_update(Base)
        insert_function(table, conn, keys, data_iter)
    """

    def fxn(
        table: SQLTable,
        conn: db.engine.Engine | db.engine.Connection,
        keys: list[str],
        data_iter: Iterable,
    ) -> None:
        """
        Execute SQL statement inserting data. If a conflict occurs, update the
        existing row.

        Args:
            table (SQLTable): The SQLAlchemy Table object representing the table.
            conn (sqlalchemy.engine.Engine or sqlalchemy.engine.Connection): The
                database connection.
            keys (list of str): The column names.
            data_iter (Iterable): An iterable that iterates over the values to be
                inserted.
        """
        subtable = table.table

        stmt = insert(subtable).values(list(data_iter))

        primary_keys = [k.name for k in inspect(base_table).primary_key]
        update_dict = {c.name: c for c in stmt.excluded if not c.primary_key}

        if not update_dict:
            insert_on_conflict_do_nothing(table, conn, keys, data_iter)
            return

        update_stmt = stmt.on_conflict_do_update(
            index_elements=primary_keys,
            set_=update_dict,
        )
        conn.execute(update_stmt)

    return fxn


def dataframe_to_sql(
    df: pd.DataFrame,
    table: DeclarativeBase,
    engine: db.engine.Engine,
    replace: bool = False,
) -> None:
    """Writes a DataFrame to a SQL table using the defined schema.

    Args:
        df (pd.DataFrame): The DataFrame to write to the database
        table (DeclarativeBase): The SQLAlchemy Base class for the table
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


def read_table(
    table: DeclarativeBase, engine: db.engine.Engine
) -> pd.DataFrame:
    """Reads a table from the database into a DataFrame.

    Args:
        table (DeclarativeBase): The SQLAlchemy Base class for the table
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
