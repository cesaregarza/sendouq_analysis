"""
Ranking confidence calculation based on player connectivity and participation.

This module implements confidence scoring that quantifies the reliability of
player rankings based on their participation patterns and graph connectivity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl


class ConfidenceTier(Enum):
    """Confidence tier classification."""

    HIGH = "high"  # Reliable rankings
    MEDIUM = "medium"  # Good rankings
    LOW = "low"  # Approximate rankings
    PROVISIONAL = "provisional"  # Insufficient data


@dataclass
class ConfidenceMetrics:
    """Detailed confidence metrics for a player."""

    player_id: str
    confidence_score: float  # 0-1 composite score
    confidence_tier: ConfidenceTier

    # Raw metrics
    tournament_count: int
    match_count: int
    unique_opponents: int
    connectivity_percent: float  # % of total player base connected to

    # Derived metrics
    information_bits: float  # Estimated bits of ranking information
    effective_comparisons: int  # Adjusted for opponent diversity
    graph_centrality: float  # How central in the player graph (0-1)

    # Uncertainty metrics
    rank_uncertainty: int  # ± range for true rank at 95% confidence
    percentile_uncertainty: float  # ± range for percentile at 95% confidence

    def to_display_string(self) -> str:
        """Format confidence for user display."""
        emoji = {
            ConfidenceTier.HIGH: "🟢",
            ConfidenceTier.MEDIUM: "🟡",
            ConfidenceTier.LOW: "🟠",
            ConfidenceTier.PROVISIONAL: "🔴",
        }[self.confidence_tier]

        message = {
            ConfidenceTier.HIGH: f"Highly reliable ranking based on {self.tournament_count} tournaments",
            ConfidenceTier.MEDIUM: f"Good ranking accuracy with {self.tournament_count} tournaments",
            ConfidenceTier.LOW: f"Approximate ranking - play more to improve accuracy",
            ConfidenceTier.PROVISIONAL: f"Provisional ranking - need {max(5 - self.tournament_count, 0)} more tournaments",
        }[self.confidence_tier]

        return f"{emoji} {message}"


class RankingConfidence:
    """
    Calculate confidence scores for player rankings.

    Confidence is based on:
    1. Tournament participation (quantity of data)
    2. Unique opponents faced (diversity of comparisons)
    3. Graph connectivity (how well-connected to player base)
    4. Match volume (statistical significance)

    The confidence score determines how much we can trust a player's ranking,
    with provisional players essentially unrankable due to insufficient data.
    """

    def __init__(
        self,
        total_players: int,
        damping_factor: float = 0.85,
        min_tournaments_high: int = 20,
        min_opponents_high: int = 150,
        min_connectivity_high: float = 3.0,
    ):
        """
        Initialize confidence calculator.

        Parameters
        ----------
        total_players : int
            Total number of players in the system
        damping_factor : float
            PageRank damping factor (for information loss calculation)
        min_tournaments_high : int
            Minimum tournaments for high confidence
        min_opponents_high : int
            Minimum unique opponents for high confidence
        min_connectivity_high : float
            Minimum connectivity % for high confidence
        """
        self.total_players = total_players
        self.damping_factor = damping_factor

        # Tier thresholds
        self.min_tournaments_high = min_tournaments_high
        self.min_opponents_high = min_opponents_high
        self.min_connectivity_high = min_connectivity_high

        # Information theory constants
        self.bits_per_match = 1.0  # Each match provides ~1 bit of information
        self.bits_needed = np.log2(
            total_players
        )  # Bits to position among N players

    def calculate_player_confidence(
        self,
        player_id: str,
        tournaments: list[str],
        matches: list[dict],
        opponents: set[str],
        player_ranks: dict[str, int] | None = None,
    ) -> ConfidenceMetrics:
        """
        Calculate confidence metrics for a single player.

        Parameters
        ----------
        player_id : str
            Player identifier
        tournaments : list[str]
            List of tournament IDs participated in
        matches : list[dict]
            List of match records
        opponents : set[str]
            Set of unique opponent IDs faced
        player_ranks : dict[str, int] | None
            Current rankings for uncertainty calculation

        Returns
        -------
        ConfidenceMetrics
            Detailed confidence metrics
        """
        num_tournaments = len(tournaments)
        num_matches = len(matches)
        num_opponents = len(opponents)
        connectivity_percent = (
            100.0 * num_opponents / self.total_players
            if self.total_players > 0
            else 0
        )

        confidence_score = self._calculate_composite_score(
            num_tournaments, num_opponents, connectivity_percent
        )

        confidence_tier = self._determine_tier(
            num_tournaments,
            num_opponents,
            connectivity_percent,
            confidence_score,
        )

        information_bits = self._calculate_information_bits(
            num_matches, num_opponents
        )
        effective_comparisons = self._calculate_effective_comparisons(
            num_matches, num_opponents
        )

        graph_centrality = min(connectivity_percent / 10.0, 1.0)

        rank_uncertainty, percentile_uncertainty = self._estimate_uncertainty(
            num_matches,
            num_opponents,
            connectivity_percent,
            player_ranks,
            player_id,
        )

        return ConfidenceMetrics(
            player_id=player_id,
            confidence_score=confidence_score,
            confidence_tier=confidence_tier,
            tournament_count=num_tournaments,
            match_count=num_matches,
            unique_opponents=num_opponents,
            connectivity_percent=connectivity_percent,
            information_bits=information_bits,
            effective_comparisons=effective_comparisons,
            graph_centrality=graph_centrality,
            rank_uncertainty=rank_uncertainty,
            percentile_uncertainty=percentile_uncertainty,
        )

    def _calculate_composite_score(
        self,
        num_tournaments: int,
        num_opponents: int,
        connectivity_percent: float,
    ) -> float:
        """Calculate 0-1 composite confidence score."""
        # Normalize each component to 0-1
        tournament_score = min(num_tournaments / self.min_tournaments_high, 1.0)
        opponent_score = min(num_opponents / self.min_opponents_high, 1.0)
        connectivity_score = min(
            connectivity_percent / self.min_connectivity_high, 1.0
        )

        # Weighted combination (tournaments and opponents equally important)
        return (
            0.4 * tournament_score
            + 0.4 * opponent_score
            + 0.2 * connectivity_score
        )

    def _determine_tier(
        self,
        num_tournaments: int,
        num_opponents: int,
        connectivity_percent: float,
        confidence_score: float,
    ) -> ConfidenceTier:
        """Determine confidence tier based on thresholds."""
        # High confidence criteria
        if (
            num_tournaments >= self.min_tournaments_high
            or (
                num_tournaments >= 15
                and num_opponents >= self.min_opponents_high
            )
            or (
                num_tournaments >= 10
                and connectivity_percent >= self.min_connectivity_high
            )
        ):
            return ConfidenceTier.HIGH

        # Medium confidence criteria
        if (
            num_tournaments >= 10
            or (num_tournaments >= 5 and num_opponents >= 100)
            or (num_tournaments >= 5 and connectivity_percent >= 2.0)
        ):
            return ConfidenceTier.MEDIUM

        # Low confidence criteria
        if num_tournaments >= 5 or (
            num_tournaments >= 3 and num_opponents >= 50
        ):
            return ConfidenceTier.LOW

        # Otherwise provisional
        return ConfidenceTier.PROVISIONAL

    def _calculate_information_bits(
        self, num_matches: int, num_opponents: int
    ) -> float:
        """Calculate estimated bits of ranking information."""
        # Each match provides information, but repeated opponents give less
        diversity_factor = min(num_opponents / max(num_matches, 1), 1.0)
        effective_matches = num_matches * diversity_factor
        return min(effective_matches * self.bits_per_match, self.bits_needed)

    def _calculate_effective_comparisons(
        self, num_matches: int, num_opponents: int
    ) -> int:
        """Calculate effective number of comparisons adjusted for diversity."""
        if num_matches == 0:
            return 0
        diversity_factor = min(num_opponents / num_matches, 1.0)
        return int(num_matches * diversity_factor)

    def _estimate_uncertainty(
        self,
        num_matches: int,
        num_opponents: int,
        connectivity_percent: float,
        player_ranks: dict[str, int] | None,
        player_id: str,
    ) -> tuple[int, float]:
        """
        Estimate ranking uncertainty based on data quantity.

        Returns
        -------
        rank_uncertainty : int
            ± range for true rank at 95% confidence
        percentile_uncertainty : float
            ± range for percentile at 95% confidence
        """
        # Base uncertainty from match count (binomial confidence interval approximation)
        if num_matches < 10:
            base_uncertainty = 0.30
        elif num_matches < 50:
            base_uncertainty = 0.14
        elif num_matches < 200:
            base_uncertainty = 0.07
        else:
            base_uncertainty = 0.05

        # Adjust for connectivity (sparse connections increase uncertainty)
        connectivity_factor = 1.0 + (1.0 - min(connectivity_percent / 5.0, 1.0))

        # Graph distance factor (more hops = more uncertainty)
        avg_path_length = np.log(self.total_players) / np.log(
            max(num_opponents, 2)
        )
        path_uncertainty = 1.0 - self.damping_factor**avg_path_length

        # Combined uncertainty
        total_uncertainty = min(
            base_uncertainty * connectivity_factor + path_uncertainty, 1.0
        )

        # Convert to rank ranges
        if player_ranks and player_id in player_ranks:
            current_rank = player_ranks[player_id]
            rank_uncertainty = int(current_rank * total_uncertainty)
        else:
            rank_uncertainty = int(self.total_players * total_uncertainty / 2)

        percentile_uncertainty = total_uncertainty * 100 / 2

        return rank_uncertainty, percentile_uncertainty

    def calculate_all_confidences(
        self, players_df: pl.DataFrame, matches_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Calculate confidence for all players.

        Parameters
        ----------
        players_df : pl.DataFrame
            Player participation data
        matches_df : pl.DataFrame
            Match records

        Returns
        -------
        pl.DataFrame
            DataFrame with player_id and confidence metrics
        """
        player_stats = {}

        for row in players_df.iter_rows(named=True):
            player_id = row.get("user_id")
            if player_id:
                if player_id not in player_stats:
                    player_stats[player_id] = {
                        "tournaments": set(),
                        "matches": [],
                        "opponents": set(),
                    }
                player_stats[player_id]["tournaments"].add(row["tournament_id"])

        for row in matches_df.iter_rows(named=True):
            winner_team = row.get("winner_team_id")
            loser_team = row.get("loser_team_id")

            if winner_team and loser_team:
                winners = players_df.filter(pl.col("team_id") == winner_team)[
                    "user_id"
                ].to_list()
                losers = players_df.filter(pl.col("team_id") == loser_team)[
                    "user_id"
                ].to_list()

                for winner in winners:
                    if winner and winner in player_stats:
                        player_stats[winner]["matches"].append(row)
                        player_stats[winner]["opponents"].update(
                            [loser for loser in losers if loser]
                        )

                for loser in losers:
                    if loser and loser in player_stats:
                        player_stats[loser]["matches"].append(row)
                        player_stats[loser]["opponents"].update(
                            [winner for winner in winners if winner]
                        )

        confidence_records = []
        for player_id, stats in player_stats.items():
            metrics = self.calculate_player_confidence(
                player_id=player_id,
                tournaments=list(stats["tournaments"]),
                matches=stats["matches"],
                opponents=stats["opponents"],
            )

            confidence_records.append(
                {
                    "player_id": player_id,
                    "confidence_score": metrics.confidence_score,
                    "confidence_tier": metrics.confidence_tier.value,
                    "tournament_count": metrics.tournament_count,
                    "unique_opponents": metrics.unique_opponents,
                    "connectivity_percent": metrics.connectivity_percent,
                    "information_bits": metrics.information_bits,
                    "rank_uncertainty": metrics.rank_uncertainty,
                    "display_string": metrics.to_display_string(),
                }
            )

        return pl.DataFrame(confidence_records)
