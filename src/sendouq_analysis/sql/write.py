from typing import TYPE_CHECKING

from sqlalchemy.orm import sessionmaker

if TYPE_CHECKING:
    import pandas as pd
    import sqlalchemy as db

    from sendouq_analysis.sql.meta import Base


def dataframe_to_sql(
    df: pd.DataFrame, table: Base, engine: db.engine.Engine
) -> None:
    """Writes a DataFrame to a SQL table using the defined schema.

    Args:
        df (pd.DataFrame): The DataFrame to write to the database
        table (Base): The SQLAlchemy Base class for the table
        engine (db.engine.Engine): The engine for the database
    """
    Session = sessionmaker(bind=engine)
    session = Session()

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
