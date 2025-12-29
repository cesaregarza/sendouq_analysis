#!/usr/bin/env python3
"""Main entrypoint for the rankings module automatic scraper.

Scrapes tournaments, runs PageRank algorithm, and saves results.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import polars as pl
import requests
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from rankings.algorithms.exposure_log_odds import ExposureLogOddsEngine
from rankings.core import ExposureLogOddsConfig
from rankings.core.logging import get_logger, log_timing
from rankings.scraping.api import validate_tournament_data
from rankings.scraping.turbo_stream import extract_tournament_data

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

logger = get_logger(__name__)


def get_database_url() -> str:
    """Get database URL from environment or fall back to SQLite."""
    # Check for DATABASE_URL first (standard for many platforms)
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]

    # Check for PostgreSQL components
    if all(
        os.environ.get(k)
        for k in ["POSTGRES_HOST", "POSTGRES_DB", "POSTGRES_USER"]
    ):
        host = os.environ["POSTGRES_HOST"]
        port = os.environ.get("POSTGRES_PORT", "5432")
        db = os.environ["POSTGRES_DB"]
        user = os.environ["POSTGRES_USER"]
        password = os.environ.get("POSTGRES_PASSWORD", "")
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"

    # Fall back to SQLite
    db_path = os.environ.get("RANKINGS_DB_PATH", "data/tournaments.db")
    return f"sqlite:///{db_path}"


class RankingScraper:
    """Automated tournament scraper and ranking calculator."""

    def __init__(self, database_url: str | None = None) -> None:
        """Initialize the scraper with database connection.

        Args:
            database_url: Database connection URL. If None, uses environment variables.
        """
        self.database_url = database_url or get_database_url()

        # Parse URL to determine database type
        parsed = urlparse(self.database_url)
        self.db_type = (
            parsed.scheme.split("+")[0]
            if "+" in parsed.scheme
            else parsed.scheme
        )

        # Create engine with appropriate settings
        if self.db_type == "sqlite":
            self.engine = create_engine(self.database_url, echo=False)
        else:
            # For PostgreSQL/MySQL, add connection pool settings
            self.engine = create_engine(
                self.database_url,
                echo=False,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
            )

        self.Session = sessionmaker(bind=self.engine)

        # Configure requests session with environment settings
        self.session = requests.Session()
        self.api_base_url = os.environ.get(
            "SENDOU_API_BASE_URL", "https://sendou.ink"
        )
        self.api_timeout = int(os.environ.get("API_TIMEOUT", "30"))
        self.api_retry_count = int(os.environ.get("API_RETRY_COUNT", "3"))
        self.rate_limit_delay = float(
            os.environ.get("API_RATE_LIMIT_DELAY", "0.5")
        )

    def get_latest_tournament_id(self) -> int | None:
        """Get the latest tournament ID from the database.

        Returns:
            Latest tournament ID or None if no tournaments exist.
        """
        with self.Session() as db_session:
            result = db_session.execute(
                text("SELECT MAX(id) as max_id FROM tournaments")
            ).fetchone()
            if result and result[0]:
                return result[0]
        return None

    def _try_turbo_stream_endpoint(self, tournament_id: int) -> dict | None:
        """Try fetching from the new turbo-stream .data endpoint."""
        url = f"{self.api_base_url}/to/{tournament_id}/results.data"
        try:
            response = self.session.get(url, timeout=self.api_timeout)
            response.raise_for_status()
            raw_data = response.json()
            data = extract_tournament_data(raw_data)
            if data and validate_tournament_data(data):
                logger.debug(f"Fetched tournament {tournament_id} via turbo-stream")
                return data
        except Exception as e:
            logger.debug(f"Turbo-stream endpoint failed for {tournament_id}: {e}")
        return None

    def _try_legacy_endpoint(self, tournament_id: int) -> dict | None:
        """Try fetching from the legacy ?_data endpoint (plain JSON)."""
        url = f"{self.api_base_url}/to/{tournament_id}?_data=features%2Ftournament%2Froutes%2Fto.%24id"
        try:
            response = self.session.get(url, timeout=self.api_timeout)
            response.raise_for_status()
            data = response.json()
            if data and validate_tournament_data(data):
                logger.debug(f"Fetched tournament {tournament_id} via legacy endpoint")
                return data
        except Exception as e:
            logger.debug(f"Legacy endpoint failed for {tournament_id}: {e}")
        return None

    def scrape_tournament(self, tournament_id: int) -> dict:
        """Scrape tournament data from the API with retries.

        Tries the new turbo-stream .data endpoint first, then falls back
        to the legacy ?_data endpoint for resilience.

        Args:
            tournament_id: Tournament ID to scrape.

        Returns:
            Tournament data as dictionary.

        Raises:
            Exception: If scraping fails after all retry attempts.
        """
        logger.info(f"Scraping tournament {tournament_id}")

        for attempt in range(self.api_retry_count):
            with log_timing(logger, f"tournament {tournament_id} scraping"):
                # Try turbo-stream endpoint first (new format)
                data = self._try_turbo_stream_endpoint(tournament_id)
                if data is not None:
                    logger.info(f"Successfully scraped tournament {tournament_id}")
                    return data

                # Fall back to legacy endpoint
                data = self._try_legacy_endpoint(tournament_id)
                if data is not None:
                    logger.info(f"Successfully scraped tournament {tournament_id}")
                    return data

            # Both failed, retry with backoff
            if attempt == self.api_retry_count - 1:
                logger.error(
                    f"Failed to scrape tournament {tournament_id} after {self.api_retry_count} attempts"
                )
                raise RuntimeError(
                    f"Failed to fetch tournament {tournament_id} after {self.api_retry_count} attempts"
                )
            logger.warning(
                f"Attempt {attempt + 1} failed for tournament {tournament_id}"
            )
            time.sleep(self.rate_limit_delay * (attempt + 1))

    def load_match_data(self, days_back: int = 90) -> pl.DataFrame:
        """Load match data from database for the specified time window.

        Args:
            days_back: Number of days to look back from current date.

        Returns:
            DataFrame containing match data.
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)

        query = """
        SELECT 
            m.id as match_id,
            m.stage_id,
            m.round_id,
            m.opponent1_id as team1_id,
            m.opponent2_id as team2_id,
            m.opponent1_score as team1_score,
            m.opponent2_score as team2_score,
            m.opponent1_result as team1_result,
            m.opponent2_result as team2_result,
            m.created_at,
            t.id as tournament_id,
            t.start_time as tournament_start_time
        FROM matches m
        JOIN tournaments t ON m.stage_id IN (
            SELECT id FROM stages WHERE tournament_id = t.id
        )
        WHERE m.status = 'completed'
        AND t.start_time >= ?
        ORDER BY t.start_time DESC
        """

        with self.Session() as db_session:
            result = db_session.execute(text(query), [cutoff_date.isoformat()])
            columns = result.keys()
            data = [dict(zip(columns, row)) for row in result.fetchall()]

        if not data:
            logger.warning("No match data found in database")
            return pl.DataFrame()

        return pl.DataFrame(data)

    def load_player_data(self) -> pl.DataFrame:
        """Load player data from database.

        Returns:
            DataFrame containing player data.
        """
        query = """
        SELECT DISTINCT
            tm.user_id as id,
            tm.username,
            tm.discord_id,
            tm.country,
            MAX(t.start_time) as last_seen
        FROM team_members tm
        JOIN teams te ON tm.team_id = te.id
        JOIN tournaments t ON te.tournament_id = t.id
        GROUP BY tm.user_id
        """

        with self.Session() as db_session:
            result = db_session.execute(text(query))
            columns = result.keys()
            data = [dict(zip(columns, row)) for row in result.fetchall()]

        if not data:
            logger.warning("No player data found in database")
            return pl.DataFrame()

        return pl.DataFrame(data)

    def run_pagerank(
        self, matches: pl.DataFrame, players: pl.DataFrame
    ) -> pl.DataFrame:
        """Run ExposureLogOdds PageRank algorithm on the data.

        Args:
            matches: DataFrame containing match data.
            players: DataFrame containing player data.

        Returns:
            DataFrame containing player rankings.
        """
        logger.info("Running ExposureLogOdds PageRank algorithm")

        config = ExposureLogOddsConfig(
            lambda_mode="auto",
            apply_log_transform=True,
            engine={
                "min_exposure": float(
                    os.environ.get("PAGERANK_MIN_EXPOSURE", "5")
                ),
                "score_decay_rate": float(
                    os.environ.get("PAGERANK_DECAY_RATE", "0.01")
                ),
                "score_decay_delay_days": int(
                    os.environ.get("PAGERANK_DECAY_DELAY_DAYS", "30")
                ),
                "beta": 0.5,
                "lambda_smooth": 1e-4,
            },
            pagerank={
                "alpha": float(os.environ.get("PAGERANK_ALPHA", "0.85")),
                "tol": float(os.environ.get("PAGERANK_TOLERANCE", "1e-8")),
                "max_iter": int(
                    os.environ.get("PAGERANK_MAX_ITERATIONS", "100")
                ),
            },
        )

        engine = ExposureLogOddsEngine(config=config)

        with log_timing(logger, "pagerank calculation"):
            rankings = engine.rank_players(matches, players)

        logger.info(f"Calculated rankings for {len(rankings)} players")
        return rankings

    def save_rankings(
        self, rankings: pl.DataFrame, timestamp: datetime | None = None
    ) -> None:
        """Save ranking results to database.

        Args:
            rankings: DataFrame containing player rankings.
            timestamp: Timestamp for the rankings. Defaults to current time.
        """
        if timestamp is None:
            timestamp = datetime.now()

        logger.info(f"Saving {len(rankings)} rankings to database")

        # Create rankings table if it doesn't exist
        create_table_query = """
        CREATE TABLE IF NOT EXISTS rankings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            rank REAL NOT NULL,
            score REAL NOT NULL,
            win_pr REAL,
            loss_pr REAL,
            exposure REAL,
            calculated_at TIMESTAMP NOT NULL,
            FOREIGN KEY (player_id) REFERENCES users(id)
        )
        """

        with self.Session() as db_session:
            db_session.execute(text(create_table_query))
            db_session.commit()

            # Insert rankings
            for row in rankings.iter_rows(named=True):
                insert_query = """
                INSERT INTO rankings (player_id, rank, score, win_pr, loss_pr, exposure, calculated_at)
                VALUES (:player_id, :rank, :score, :win_pr, :loss_pr, :exposure, :calculated_at)
                """
                db_session.execute(
                    text(insert_query),
                    {
                        "player_id": row["id"],
                        "rank": row["player_rank"],
                        "score": row.get("score", row["player_rank"]),
                        "win_pr": row.get("win_pr"),
                        "loss_pr": row.get("loss_pr"),
                        "exposure": row.get("exposure"),
                        "calculated_at": timestamp.isoformat(),
                    },
                )
            db_session.commit()

        logger.info("Rankings saved successfully")

    def check_for_new_tournaments(
        self, max_failures: int = 5
    ) -> list[tuple[int, dict]]:
        """Check for new tournaments since the last known ID.

        Args:
            max_failures: Maximum consecutive failures before stopping.

        Returns:
            List of tuples containing (tournament_id, tournament_data).
        """
        latest_id = self.get_latest_tournament_id()
        if latest_id is None:
            logger.warning(
                "No tournaments found in database, starting from ID 1"
            )
            latest_id = 0

        logger.info(f"Checking for tournaments after ID {latest_id}")

        new_tournaments = []
        consecutive_failures = 0
        current_id = latest_id + 1

        while consecutive_failures < max_failures:
            try:
                data = self.scrape_tournament(current_id)
                if data and data.get("tournament"):
                    new_tournaments.append((current_id, data))
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
            except Exception as e:
                logger.debug(f"Failed to fetch tournament {current_id}: {e}")
                consecutive_failures += 1

            current_id += 1
            time.sleep(0.5)  # Rate limiting

        logger.info(f"Found {len(new_tournaments)} new tournaments")
        return new_tournaments

    def process_tournament(
        self, tournament_id: int, tournament_data: dict
    ) -> None:
        """Process and save a single tournament to the database.

        Args:
            tournament_id: Tournament ID.
            tournament_data: Tournament data dictionary.
        """
        # This would use the existing parsing logic from scrape_tournament.py
        # For now, we'll just log it
        logger.info(f"Processing tournament {tournament_id}")
        # TODO: Implement tournament parsing and saving logic

    def run(self, check_new: bool = True, recalculate: bool = True) -> None:
        """Main execution flow.

        Args:
            check_new: Whether to check for new tournaments.
            recalculate: Whether to recalculate rankings.
        """
        logger.info("Starting ranking scraper")

        try:
            # Check for new tournaments
            if check_new:
                new_tournaments = self.check_for_new_tournaments()
                for tid, data in new_tournaments:
                    self.process_tournament(tid, data)

            # Run PageRank calculation
            if recalculate:
                logger.info("Loading match and player data")
                matches = self.load_match_data(days_back=90)
                players = self.load_player_data()

                if not matches.is_empty() and not players.is_empty():
                    rankings = self.run_pagerank(matches, players)
                    self.save_rankings(rankings)
                    logger.info("PageRank calculation completed successfully")
                else:
                    logger.warning("Insufficient data for PageRank calculation")

        except Exception as e:
            logger.error(f"Error in ranking scraper: {e}")
            logger.error(traceback.format_exc())
            raise

        logger.info("Ranking scraper completed")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Rankings module automatic scraper"
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Database URL (overrides environment variables)",
    )
    parser.add_argument(
        "--no-scrape",
        action="store_true",
        default=not os.environ.get("ENABLE_TOURNAMENT_SCRAPING", "true").lower()
        == "true",
        help="Skip checking for new tournaments",
    )
    parser.add_argument(
        "--no-calculate",
        action="store_true",
        default=not os.environ.get("ENABLE_RANKING_CALCULATION", "true").lower()
        == "true",
        help="Skip PageRank calculation",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        default=os.environ.get("RUN_ONCE", "false").lower() == "true",
        help="Run once and exit (default: run continuously)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=int(os.environ.get("SCRAPER_RUN_INTERVAL", "3600")),
        help="Interval between runs in seconds (default: 3600)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=os.environ.get("DRY_RUN", "false").lower() == "true",
        help="Run without saving to database",
    )

    args = parser.parse_args()

    # Log configuration
    logger.info(
        f"Starting rankings scraper in {'DRY RUN' if args.dry_run else 'NORMAL'} mode"
    )
    logger.info(f"Environment: {os.environ.get('CI_ENVIRONMENT', 'local')}")
    logger.info(f"Deployment ID: {os.environ.get('DEPLOYMENT_ID', 'local')}")

    scraper = RankingScraper(database_url=args.database_url)

    if args.once:
        scraper.run(
            check_new=not args.no_scrape, recalculate=not args.no_calculate
        )
    else:
        logger.info(
            f"Running continuously with interval of {args.interval} seconds"
        )
        while True:
            try:
                scraper.run(
                    check_new=not args.no_scrape,
                    recalculate=not args.no_calculate,
                )
            except Exception as e:
                logger.error(f"Error in continuous run: {e}")

            logger.info(f"Sleeping for {args.interval} seconds")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
