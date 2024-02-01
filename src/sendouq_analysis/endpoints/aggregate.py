import logging
import os
import sys

import requests
from sqlalchemy.exc import ProgrammingError

logger = logging.getLogger(__name__)


def create_new_aggregate() -> None:
    """Runs the aggregation process"""
    # Set up logging for output to docker logs, remove for production
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)

    logger.info("Starting aggregation process")
    logger.info("Creating an engine to connect to the database")
    engine = create_engine()
    logger.info("Aggregating data")
    aggregate_data(engine)
    logger.info("Aggregation process complete")
