import logging
import os
import sys

import requests
from sqlalchemy.exc import ProgrammingError

from sendouq_analysis.ingest import (
    create_engine,
    create_schema,
    load_latest_match_number,
    parse_all,
    scrape_matches,
    write_tables,
)
from sendouq_analysis.utils import delete_droplet, get_droplet_id, setup_logging

logger = logging.getLogger(__name__)


def update_database() -> None:
    """Runs the scraping and parsing process"""
    # Set up logging for output to docker logs, remove for production
    if os.environ.get("ENV") != "prod":
        setup_logging()

    logger.info("Starting scraping and parsing process")
    logger.info("Creating an engine to connect to the database")
    engine = create_engine()
    logger.info("Loading latest match number")
    next_match = load_latest_match_number(engine) + 1
    logger.info("Scraping matches starting from match number %s", next_match)
    chunk_size = 100
    count = 0
    condition = True
    while condition:
        matches = scrape_matches(next_match + count, chunk_size=chunk_size)
        if len(matches) == 0:
            logger.info("Chunk is empty, ending scrape")
            break
        elif len(matches) < chunk_size:
            logger.info("Scraped all matches, ending scrape")
            condition = False
        parsed_matches = parse_all(matches, disable_tqdm=True)
        try:
            write_tables(*parsed_matches, engine)
        except ProgrammingError:
            logger.warning(
                "Error writing to database, attempting to create schema"
            )
            create_schema(engine)
            write_tables(*parsed_matches, engine)

        count += chunk_size
    logger.info("Finished scraping and parsing process")
    logger.info("Shutting down droplet")
    do_api_token = os.environ["DO_API_TOKEN"]
    if do_api_token is None:
        logger.warning(
            "No DigitalOcean API token found, assuming not running on "
            "DigitalOcean droplet."
        )
        return

    # Get droplet id from the droplet name, "sendouq_scraper"
    droplet_id = get_droplet_id(do_api_token, "sendouq-scraper")
    delete_droplet(do_api_token, droplet_id)


if __name__ == "__main__":
    update_database()
