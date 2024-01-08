import logging

from sendouq_analysis.ingest import (
    load_latest_match_number,
    parse_all,
    scrape_matches,
    write_tables,
)

logger = logging.getLogger(__name__)


def update_database() -> None:
    """Runs the scraping and parsing process"""
    logger.info("Starting scraping and parsing process")
    logger.info("Loading latest match number")
    latest_match = load_latest_match_number()
    logger.info("Scraping matches up to match number %s", latest_match)
    matches = scrape_matches(latest_match)
    logger.info("Parsing all %s matches", len(matches))
    parsed_matches = parse_all(matches)
    logger.info("Writing all data to database")
    write_tables(*parsed_matches)
    logger.info("Finished scraping and parsing process")


if __name__ == "__main__":
    update_database()
