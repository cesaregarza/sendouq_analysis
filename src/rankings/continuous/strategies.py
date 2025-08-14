"""
Scraping strategies for different tournament states and scenarios.

Defines when and how often to scrape tournaments based on their state
and other factors.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Set

from rankings.continuous.state import TournamentMetadata, TournamentState

logger = logging.getLogger(__name__)


@dataclass
class ScrapingStrategy:
    """
    Configuration for the continuous scraping strategy.

    Defines intervals and limits for different tournament states.
    """

    # Scraping intervals (in minutes)
    completed_interval: int = 1440  # 24 hours - rarely changes
    in_progress_interval: int = 15  # 15 minutes - actively updating
    scheduled_interval: int = 60  # 1 hour - check if started
    unknown_interval: int = 60  # 1 hour - discovery rate

    # Retry strategies
    max_consecutive_404s: int = 5  # Before marking as deleted
    max_consecutive_404s_known: int = 10  # For previously seen tournaments

    # Discovery
    lookahead_ids: int = 20  # How many IDs ahead to probe
    discovery_batch_size: int = 5  # IDs to probe at once

    # Cleanup
    stale_scheduled_days: int = 7  # Mark scheduled as stale after this
    cleanup_completed_days: int = 30  # Remove old completed from tracking

    # Rate limiting
    max_requests_per_hour: int = 500
    burst_size: int = 20  # Max requests in quick succession
    burst_cooldown: int = 60  # Seconds to wait after burst


class ScrapingPrioritizer:
    """
    Determines which tournaments to scrape and in what order.

    Prioritizes based on:
    - Tournament state (in-progress > scheduled > completed)
    - Time since last check
    - Discovery of new tournament IDs
    """

    def __init__(
        self,
        strategy: ScrapingStrategy,
        state_tracker: "TournamentStateTracker",
    ):
        """
        Initialize the prioritizer.

        Args:
            strategy: Scraping strategy configuration
            state_tracker: Tournament state tracker
        """
        self.strategy = strategy
        self.state_tracker = state_tracker

    def get_tournaments_to_scrape(
        self, max_tournaments: Optional[int] = None
    ) -> List[int]:
        """
        Get prioritized list of tournaments to scrape.

        Args:
            max_tournaments: Maximum number to return

        Returns:
            Prioritized list of tournament IDs
        """
        now = datetime.utcnow()
        to_scrape = []

        # 1. In-progress tournaments (highest priority)
        in_progress = self._get_due_tournaments(
            TournamentState.IN_PROGRESS, self.strategy.in_progress_interval
        )
        to_scrape.extend(in_progress)

        # 2. Scheduled tournaments (might have started)
        scheduled = self._get_due_tournaments(
            TournamentState.SCHEDULED, self.strategy.scheduled_interval
        )
        to_scrape.extend(scheduled)

        # 3. Discovery - probe for new tournament IDs
        discovery = self._get_discovery_ids()
        to_scrape.extend(discovery)

        # 4. Completed tournaments (lowest priority, just checking for updates)
        if len(to_scrape) < (max_tournaments or self.strategy.burst_size):
            completed = self._get_due_tournaments(
                TournamentState.COMPLETED,
                self.strategy.completed_interval,
                limit=5,  # Only check a few completed per cycle
            )
            to_scrape.extend(completed)

        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for tid in to_scrape:
            if tid not in seen:
                seen.add(tid)
                unique.append(tid)

        if max_tournaments:
            unique = unique[:max_tournaments]

        return unique

    def _get_due_tournaments(
        self,
        state: TournamentState,
        interval_minutes: int,
        limit: Optional[int] = None,
    ) -> List[int]:
        """
        Get tournaments of a specific state that are due for scraping.

        Args:
            state: Tournament state to filter
            interval_minutes: Minimum minutes since last check
            limit: Maximum number to return

        Returns:
            List of tournament IDs due for scraping
        """
        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=interval_minutes)

        due = []
        for tid in self.state_tracker.get_tournaments_by_state(state):
            meta = self.state_tracker.get_tournament(tid)
            if meta and (not meta.last_checked or meta.last_checked < cutoff):
                due.append(tid)

        # Sort by last_checked (oldest first)
        due.sort(
            key=lambda tid: (
                self.state_tracker.get_tournament(tid).last_checked
                or datetime.min
            )
        )

        if limit:
            due = due[:limit]

        return due

    def _get_discovery_ids(self) -> List[int]:
        """
        Get new tournament IDs to probe for discovery.

        Returns:
            List of tournament IDs to try
        """
        highest_known = self.state_tracker.get_highest_known_id()
        if highest_known == 0:
            # Start from a reasonable recent ID
            highest_known = 1900  # Adjust based on your data

        # Generate IDs to probe
        discovery_ids = []
        for i in range(
            1,
            min(self.strategy.lookahead_ids, self.strategy.discovery_batch_size)
            + 1,
        ):
            tid = highest_known + i
            # Only add if we haven't seen it or it's been a while
            meta = self.state_tracker.get_tournament(tid)
            if not meta or self._should_retry_unknown(meta):
                discovery_ids.append(tid)

        return discovery_ids[: self.strategy.discovery_batch_size]

    def _should_retry_unknown(self, meta: TournamentMetadata) -> bool:
        """
        Determine if we should retry an unknown/404 tournament.

        Args:
            meta: Tournament metadata

        Returns:
            True if we should retry
        """
        # If marked as deleted or stale, don't retry
        if meta.state in [TournamentState.DELETED, TournamentState.STALE]:
            return False

        # Use different thresholds based on whether we've seen it before
        if meta.scrape_count > 0:
            return (
                meta.consecutive_404s < self.strategy.max_consecutive_404s_known
            )
        else:
            return meta.consecutive_404s < self.strategy.max_consecutive_404s

    def should_mark_deleted(self, tournament_id: int) -> bool:
        """
        Determine if a tournament should be marked as deleted.

        Args:
            tournament_id: Tournament ID

        Returns:
            True if it should be marked as deleted
        """
        meta = self.state_tracker.get_tournament(tournament_id)
        if not meta:
            return False

        # Different thresholds for known vs unknown tournaments
        if meta.scrape_count > 0:
            # We've successfully scraped it before
            return (
                meta.consecutive_404s
                >= self.strategy.max_consecutive_404s_known
            )
        else:
            # Never successfully scraped
            return meta.consecutive_404s >= self.strategy.max_consecutive_404s

    def get_stale_scheduled(self) -> List[int]:
        """
        Get scheduled tournaments that should be marked as stale.

        Returns:
            List of tournament IDs to mark as stale
        """
        return self.state_tracker.get_stale_scheduled_tournaments(
            self.strategy.stale_scheduled_days
        )
