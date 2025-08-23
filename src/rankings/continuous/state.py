"""Tournament state tracking and lifecycle management."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class TournamentState(Enum):
    """Tournament lifecycle states."""

    SCHEDULED = "scheduled"  # Tournament created but not started
    IN_PROGRESS = "in_progress"  # Tournament currently running
    COMPLETED = "completed"  # Tournament finished
    DELETED = "deleted"  # Tournament was deleted (404)
    STALE = "stale"  # Scheduled tournament that never started
    UNKNOWN = "unknown"  # Not yet discovered


@dataclass
class TournamentMetadata:
    """Metadata for a tracked tournament."""

    tournament_id: int
    state: TournamentState
    scheduled_date: datetime | None = None
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    last_checked: datetime | None = None
    last_modified: datetime | None = None  # From API response headers
    consecutive_404s: int = 0
    scrape_count: int = 0
    error_count: int = 0
    last_error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["state"] = self.state.value
        # Convert datetime objects to ISO format strings
        for key in [
            "scheduled_date",
            "first_seen",
            "last_seen",
            "last_checked",
            "last_modified",
        ]:
            if data[key]:
                data[key] = data[key].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> TournamentMetadata:
        """Create from dictionary."""
        # Convert state string back to enum
        data["state"] = TournamentState(data["state"])
        # Convert ISO strings back to datetime objects
        for key in [
            "scheduled_date",
            "first_seen",
            "last_seen",
            "last_checked",
            "last_modified",
        ]:
            if data.get(key):
                data[key] = datetime.fromisoformat(data[key])
        return cls(**data)


class TournamentStateTracker:
    """
    Tracks tournament states and lifecycle.

    Maintains a persistent record of tournament metadata to make
    intelligent decisions about when and how often to scrape.
    """

    def __init__(self, state_file: str = "data/tournament_state.json"):
        """
        Initialize the state tracker.

        Args:
            state_file: Path to persist tournament state data
        """
        self.state_file = Path(state_file)
        self.tournaments: dict[int, TournamentMetadata] = {}
        self.load_state()

    def load_state(self) -> None:
        """Load tournament state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    self.tournaments = {
                        int(tid): TournamentMetadata.from_dict(meta)
                        for tid, meta in data.items()
                    }
                logger.info(
                    f"Loaded state for {len(self.tournaments)} tournaments"
                )
            except Exception as e:
                logger.error(f"Failed to load state file: {e}")
                self.tournaments = {}
        else:
            logger.info("No existing state file, starting fresh")

    def save_state(self) -> None:
        """Persist tournament state to disk."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            str(tid): meta.to_dict() for tid, meta in self.tournaments.items()
        }
        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Saved state for {len(self.tournaments)} tournaments")

    def get_tournament(self, tournament_id: int) -> TournamentMetadata | None:
        """Get metadata for a tournament."""
        return self.tournaments.get(tournament_id)

    def update_tournament(
        self,
        tournament_id: int,
        state: TournamentState | None = None,
        scheduled_date: datetime | None = None,
        last_modified: datetime | None = None,
        is_404: bool = False,
        error: str | None = None,
    ) -> TournamentMetadata:
        """
        Update or create tournament metadata.

        Args:
            tournament_id: Tournament ID
            state: New state (if changed)
            scheduled_date: Scheduled start date from tournament data
            last_modified: Last-Modified header from API response
            is_404: Whether the request returned 404
            error: Error message if scraping failed

        Returns:
            Updated tournament metadata
        """
        now = datetime.utcnow()

        if tournament_id not in self.tournaments:
            # New tournament
            self.tournaments[tournament_id] = TournamentMetadata(
                tournament_id=tournament_id,
                state=state or TournamentState.UNKNOWN,
                first_seen=now,
                last_seen=now,
                last_checked=now,
            )

        meta = self.tournaments[tournament_id]
        meta.last_checked = now

        if is_404:
            meta.consecutive_404s += 1
        else:
            meta.consecutive_404s = 0
            meta.last_seen = now
            meta.scrape_count += 1

        if error:
            meta.error_count += 1
            meta.last_error = error

        if state:
            meta.state = state

        if scheduled_date:
            meta.scheduled_date = scheduled_date

        if last_modified:
            meta.last_modified = last_modified

        return meta

    def get_stale_scheduled_tournaments(self, stale_days: int = 7) -> list[int]:
        """
        Get scheduled tournaments that should be marked as stale.

        Args:
            stale_days: Days after scheduled date to consider stale

        Returns:
            List of tournament IDs that should be marked stale
        """
        now = datetime.utcnow()
        cutoff = now - timedelta(days=stale_days)

        stale_ids = []
        for tid, meta in self.tournaments.items():
            if (
                meta.state == TournamentState.SCHEDULED
                and meta.scheduled_date
                and meta.scheduled_date < cutoff
            ):
                stale_ids.append(tid)

        return stale_ids

    def mark_stale(self, tournament_ids: list[int]) -> None:
        """Mark tournaments as stale."""
        for tid in tournament_ids:
            if tid in self.tournaments:
                self.tournaments[tid].state = TournamentState.STALE
                logger.info(f"Marked tournament {tid} as stale")

    def get_tournaments_by_state(self, state: TournamentState) -> list[int]:
        """Get all tournament IDs with a specific state."""
        return [
            tid for tid, meta in self.tournaments.items() if meta.state == state
        ]

    def get_active_tournaments(self) -> list[int]:
        """Get tournaments that should be actively monitored."""
        active_states = {TournamentState.SCHEDULED, TournamentState.IN_PROGRESS}
        return [
            tid
            for tid, meta in self.tournaments.items()
            if meta.state in active_states
        ]

    def should_retry_404(
        self, tournament_id: int, max_consecutive_404s: int = 5
    ) -> bool:
        """
        Determine if we should retry a tournament that returned 404.

        Args:
            tournament_id: Tournament ID
            max_consecutive_404s: Maximum 404s before assuming deleted

        Returns:
            True if we should keep trying, False if it's likely deleted
        """
        meta = self.get_tournament(tournament_id)
        if not meta:
            return True  # First time seeing this ID

        # If we've seen it successfully before, it might be temporarily unavailable
        if meta.scrape_count > 0:
            return meta.consecutive_404s < max_consecutive_404s * 2

        # Never successfully scraped - might not exist yet
        return meta.consecutive_404s < max_consecutive_404s

    def get_highest_known_id(self) -> int:
        """Get the highest tournament ID we've seen."""
        if not self.tournaments:
            return 0
        return max(self.tournaments.keys())

    def cleanup_old_completed(self, days: int = 30) -> int:
        """
        Remove old completed tournaments from tracking.

        Args:
            days: Remove completed tournaments older than this

        Returns:
            Number of tournaments removed
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        to_remove = []

        for tid, meta in self.tournaments.items():
            if (
                meta.state == TournamentState.COMPLETED
                and meta.last_checked
                and meta.last_checked < cutoff
            ):
                to_remove.append(tid)

        for tid in to_remove:
            del self.tournaments[tid]

        if to_remove:
            logger.info(
                f"Cleaned up {len(to_remove)} old completed tournaments"
            )

        return len(to_remove)
