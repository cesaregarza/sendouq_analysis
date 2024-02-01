import logging
import os
import sys

import requests
from sqlalchemy.exc import ProgrammingError

from sendouq_analysis.ingest import create_engine
from sendouq_analysis.utils import delete_droplet, get_droplet_id, setup_logging

logger = logging.getLogger(__name__)


def create_new_aggregate() -> None:
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
