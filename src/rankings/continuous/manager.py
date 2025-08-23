"""Main continuous scraping manager."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

from rankings.continuous.state import (
    TournamentMetadata,
    TournamentState,
    TournamentStateTracker,
)
from rankings.continuous.strategies import ScrapingPrioritizer, ScrapingStrategy
from rankings.scraping.api import build_tournament_url, scrape_tournament
from rankings.scraping.storage import save_tournament_batch

logger = logging.getLogger(__name__)


class ContinuousScraper:
    """
    Manages continuous tournament scraping.

    Handles the complete lifecycle of continuous scraping including:
    - State tracking and persistence
    - Intelligent scheduling and prioritization
    - Rate limiting and error handling
    - Tournament lifecycle management
    """

    def __init__(
        self,
        output_dir: str = "data/tournaments",
        state_file: str = "data/tournament_state.json",
        strategy: ScrapingStrategy | None = None,
        session: requests.Session | None = None,
    ):
        """
        Initialize the continuous scraper.

        Args:
            output_dir: Directory to save tournament data
            state_file: File to persist tournament state
            strategy: Scraping strategy (uses defaults if None)
            session: Requests session for HTTP operations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.state_tracker = TournamentStateTracker(state_file)
        self.strategy = strategy or ScrapingStrategy()
        self.prioritizer = ScrapingPrioritizer(
            self.strategy, self.state_tracker
        )

        self.session = session or requests.Session()
        self.request_count = 0
        self.last_request_time = None
        self.hourly_request_times: list[datetime] = []

    def run_once(self) -> dict[str, int]:
        """
        Run a single scraping cycle.

        Returns:
            Summary of the scraping cycle
        """
        logger.info("Starting scraping cycle")

        # Clean up stale tournaments
        self._cleanup_stale()

        # Get tournaments to scrape
        to_scrape = self.prioritizer.get_tournaments_to_scrape(
            max_tournaments=self.strategy.burst_size
        )

        if not to_scrape:
            logger.info("No tournaments to scrape in this cycle")
            return {"scraped": 0, "failed": 0, "discovered": 0}

        logger.info(f"Scraping {len(to_scrape)} tournaments")

        # Scrape tournaments
        results = self._scrape_batch(to_scrape)

        # Save state after each cycle
        self.state_tracker.save_state()

        # Log summary
        logger.info(
            f"Cycle complete: scraped={results['scraped']}, "
            f"failed={results['failed']}, discovered={results['discovered']}"
        )

        return results

    def run_continuous(
        self, interval_minutes: int = 60, max_cycles: int | None = None
    ) -> None:
        """
        Run continuous scraping.

        Args:
            interval_minutes: Minutes between scraping cycles
            max_cycles: Maximum cycles to run (None for infinite)
        """
        logger.info(
            f"Starting continuous scraping with {interval_minutes} minute intervals"
        )

        cycles = 0
        while max_cycles is None or cycles < max_cycles:
            try:
                # Run a scraping cycle
                cycle_start = time.time()
                results = self.run_once()
                cycle_duration = time.time() - cycle_start

                logger.info(
                    f"Cycle {cycles + 1} completed in {cycle_duration:.1f}s: "
                    f"{results}"
                )

                # Clean up old completed tournaments periodically
                if cycles % 24 == 0:  # Once per day if hourly
                    removed = self.state_tracker.cleanup_old_completed(
                        self.strategy.cleanup_completed_days
                    )
                    if removed:
                        logger.info(f"Cleaned up {removed} old tournaments")

                cycles += 1

                # Wait for next cycle
                if max_cycles is None or cycles < max_cycles:
                    sleep_seconds = max(
                        0, interval_minutes * 60 - cycle_duration
                    )
                    if sleep_seconds > 0:
                        logger.info(
                            f"Sleeping {sleep_seconds:.0f}s until next cycle"
                        )
                        time.sleep(sleep_seconds)

            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in cycle {cycles + 1}: {e}", exc_info=True)
                # Sleep before retrying
                time.sleep(60)
                cycles += 1

        logger.info(f"Continuous scraping stopped after {cycles} cycles")

    def _scrape_batch(self, tournament_ids: list[int]) -> dict[str, int]:
        """
        Scrape a batch of tournaments.

        Args:
            tournament_ids: List of tournament IDs to scrape

        Returns:
            Summary of results
        """
        scraped = 0
        failed = 0
        discovered = 0
        batch_data = []

        for tid in tournament_ids:
            # Rate limiting
            self._rate_limit()

            try:
                # Attempt to scrape
                data = self._scrape_tournament_with_state(tid)

                if data:
                    batch_data.append(data)
                    scraped += 1

                    # Check if this is a newly discovered tournament
                    meta = self.state_tracker.get_tournament(tid)
                    if meta and meta.scrape_count == 1:
                        discovered += 1
                else:
                    failed += 1

            except Exception as e:
                logger.error(f"Error scraping tournament {tid}: {e}")
                failed += 1

        # Save batch to file
        if batch_data:
            self._save_batch(batch_data)

        return {"scraped": scraped, "failed": failed, "discovered": discovered}

    def _scrape_tournament_with_state(self, tournament_id: int) -> dict | None:
        """
        Scrape a tournament and update its state.

        Args:
            tournament_id: Tournament ID to scrape

        Returns:
            Tournament data if successful, None otherwise
        """
        logger.debug(f"Scraping tournament {tournament_id}")

        try:
            # Make the request
            url = build_tournament_url(tournament_id)
            response = self.session.get(url, timeout=30)

            # Track request for rate limiting
            self.request_count += 1
            self.last_request_time = datetime.utcnow()
            self.hourly_request_times.append(self.last_request_time)

            # Handle 404
            if response.status_code == 404:
                self._handle_404(tournament_id)
                return None

            response.raise_for_status()
            data = response.json()

            # Determine tournament state from data
            state = self._determine_state(data)

            # Extract scheduled date if available
            scheduled_date = self._extract_scheduled_date(data)

            # Get Last-Modified header
            last_modified = None
            if "Last-Modified" in response.headers:
                try:
                    last_modified = datetime.strptime(
                        response.headers["Last-Modified"],
                        "%a, %d %b %Y %H:%M:%S %Z",
                    )
                except ValueError:
                    pass

            # Update state tracker
            self.state_tracker.update_tournament(
                tournament_id=tournament_id,
                state=state,
                scheduled_date=scheduled_date,
                last_modified=last_modified,
                is_404=False,
            )

            logger.info(
                f"Successfully scraped tournament {tournament_id} "
                f"(state: {state.value})"
            )

            return data

        except requests.RequestException as e:
            logger.warning(
                f"Request failed for tournament {tournament_id}: {e}"
            )
            self.state_tracker.update_tournament(
                tournament_id=tournament_id, error=str(e)
            )
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON for tournament {tournament_id}: {e}")
            self.state_tracker.update_tournament(
                tournament_id=tournament_id, error=f"Invalid JSON: {e}"
            )
            return None

    def _handle_404(self, tournament_id: int) -> None:
        """
        Handle a 404 response for a tournament.

        Args:
            tournament_id: Tournament ID that returned 404
        """
        logger.debug(f"Tournament {tournament_id} returned 404")

        # Update state tracker
        self.state_tracker.update_tournament(
            tournament_id=tournament_id, is_404=True
        )

        # Check if we should mark as deleted
        if self.prioritizer.should_mark_deleted(tournament_id):
            self.state_tracker.update_tournament(
                tournament_id=tournament_id, state=TournamentState.DELETED
            )
            logger.info(f"Marked tournament {tournament_id} as deleted")

    def _determine_state(self, data: Dict) -> TournamentState:
        """
        Determine tournament state from its data.

        Args:
            data: Tournament data from API

        Returns:
            Tournament state
        """
        # Check for completion indicator
        if data.get("status") == "COMPLETED":
            return TournamentState.COMPLETED

        # Check if brackets exist and matches are being played
        if data.get("brackets") and any(
            bracket.get("matches", []) for bracket in data.get("brackets", [])
        ):
            # Check if there are unfinished matches
            has_unfinished = False
            for bracket in data.get("brackets", []):
                for match in bracket.get("matches", []):
                    if not match.get("winnerId"):
                        has_unfinished = True
                        break

            if has_unfinished:
                return TournamentState.IN_PROGRESS
            else:
                return TournamentState.COMPLETED

        # If it has a future start time, it's scheduled
        start_time = data.get("startTime")
        if start_time:
            try:
                start_dt = datetime.fromisoformat(
                    start_time.replace("Z", "+00:00")
                )
                if start_dt > datetime.utcnow():
                    return TournamentState.SCHEDULED
            except (ValueError, AttributeError):
                pass

        # Default to scheduled if we can't determine
        return TournamentState.SCHEDULED

    def _extract_scheduled_date(self, data: dict) -> datetime | None:
        """
        Extract scheduled start date from tournament data.

        Args:
            data: Tournament data

        Returns:
            Scheduled datetime if found
        """
        start_time = data.get("startTime")
        if start_time:
            try:
                return datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass
        return None

    def _cleanup_stale(self) -> None:
        """Clean up stale scheduled tournaments."""
        stale_ids = self.prioritizer.get_stale_scheduled()
        if stale_ids:
            self.state_tracker.mark_stale(stale_ids)
            logger.info(f"Marked {len(stale_ids)} tournaments as stale")

    def _save_batch(self, batch_data: list[dict]) -> None:
        """
        Save a batch of tournament data.

        Args:
            batch_data: List of tournament data dictionaries
        """
        if not batch_data:
            return

        # Generate filename with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"tournaments_continuous_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(batch_data, f, indent=2)

        logger.info(f"Saved {len(batch_data)} tournaments to {filename}")

    def _rate_limit(self) -> None:
        """Apply rate limiting to avoid overwhelming the API."""
        # Clean up old hourly requests
        now = datetime.utcnow()
        one_hour_ago = now - timedelta(hours=1)
        self.hourly_request_times = [
            t for t in self.hourly_request_times if t > one_hour_ago
        ]

        # Check hourly limit
        if (
            len(self.hourly_request_times)
            >= self.strategy.max_requests_per_hour
        ):
            # Wait until the oldest request is more than an hour old
            wait_until = self.hourly_request_times[0] + timedelta(hours=1)
            wait_seconds = (wait_until - now).total_seconds()
            if wait_seconds > 0:
                logger.warning(
                    f"Rate limit reached, waiting {wait_seconds:.0f}s"
                )
                time.sleep(wait_seconds)

        # Apply minimum delay between requests
        if self.last_request_time:
            elapsed = (now - self.last_request_time).total_seconds()
            min_delay = 0.5  # Minimum 0.5 seconds between requests
            if elapsed < min_delay:
                time.sleep(min_delay - elapsed)

    def get_status(self) -> Dict:
        """
        Get current scraper status.

        Returns:
            Status information
        """
        states_count = {}
        for state in TournamentState:
            count = len(self.state_tracker.get_tournaments_by_state(state))
            if count > 0:
                states_count[state.value] = count

        return {
            "total_tournaments": len(self.state_tracker.tournaments),
            "states": states_count,
            "highest_id": self.state_tracker.get_highest_known_id(),
            "requests_this_hour": len(self.hourly_request_times),
            "active_tournaments": len(
                self.state_tracker.get_active_tournaments()
            ),
        }
