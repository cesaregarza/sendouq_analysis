import logging
import sys

from sendouq_analysis.ingest import (
    create_engine,
    load_latest_match_number,
    parse_all,
    scrape_matches,
    write_tables,
)

logger = logging.getLogger(__name__)


def update_database() -> None:
    """Runs the scraping and parsing process"""
    logger.info("Starting scraping and parsing process")
    logger.info("Creating an engine to connect to the database")
    engine = create_engine()
    logger.info("Loading latest match number")
    latest_match = load_latest_match_number(engine)
    logger.info("Scraping matches up to match number %s", latest_match)
    chunk_size = 100
    condition = True
    while condition:
        matches = scrape_matches(latest_match, chunk_size=chunk_size)
        if len(matches) == 0:
            logger.info("Chunk is empty, ending scrape")
            break
        elif len(matches) < chunk_size:
            logger.info("Scraped all matches, ending scrape")
            condition = False
        parsed_matches = parse_all(matches)
        write_tables(*parsed_matches, engine)
    logger.info("Finished scraping and parsing process")


if __name__ == "__main__":
    # Set up logging for output to docker logs
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    update_database()
