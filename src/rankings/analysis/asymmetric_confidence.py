"""
Asymmetric confidence scoring that considers edge quality over quantity.

Key insight: A player who beats top players is immediately rankable,
while a player with many matches against weak opponents remains uncertain.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl

from rankings.analysis.confidence import ConfidenceTier


@dataclass
class AsymmetricConfidenceMetrics:
    """Confidence metrics considering opponent quality."""

    player_id: str

    # Standard metrics
    tournament_count: int
    match_count: int
    unique_opponents: int

    # Quality metrics
    top_opponents_faced: int  # Number of top 10% players faced
    top_players_beaten: int  # Number of top 10% players beaten
    beaten_by_top: int  # Number of top 10% players who beat this player

    # Edge metrics
    incoming_edge_quality: float  # Average quality of players who beat you (0-1)
    outgoing_edge_quality: float  # Average quality of players you beat (0-1)
    incoming_edge_count: int  # Total players who beat you
    outgoing_edge_count: int  # Total players you beat

    # Asymmetric confidence scores
    upward_mobility: float  # Confidence in ability to rank high (based on wins)
    downward_pressure: float  # Confidence in ranking limit (based on losses)
    overall_confidence: float  # Combined asymmetric confidence (0-1)

    # Tier classification
    confidence_tier: ConfidenceTier
    rankability: str  # 'high', 'medium', 'low', 'unrankable'

    def is_tournament_winner(self) -> bool:
        """Check if player profile matches a tournament winner."""
        return (
            self.tournament_count <= 3
            and self.top_players_beaten >= 3
            and self.outgoing_edge_quality > 0.5
        )

    def is_isolated(self) -> bool:
        """Check if player is in an isolated subgraph."""
        return self.top_opponents_faced == 0 or (
            self.top_opponents_faced < 3 and self.match_count > 20
        )


class AsymmetricConfidenceCalculator:
    """
    Calculate confidence considering edge quality and asymmetry.

    Core principles:
    1. Beating top players provides strong upward signal
    2. Losing to weak players provides strong downward signal
    3. Few high-quality matches > many low-quality matches
    4. Tournament winners are immediately rankable
    """

    def __init__(
        self,
        total_players: int,
        top_player_percentile: float = 0.1,
        quality_weight: float = 2.0,
        tournament_winner_boost: float = 3.0,
    ):
        """
        Initialize calculator.

        Parameters
        ----------
        total_players : int
            Total number of players
        top_player_percentile : float
            Percentile for "top" players (default 0.1 = top 10%)
        quality_weight : float
            How much to weight opponent quality vs quantity
        tournament_winner_boost : float
            Confidence multiplier for tournament winners
        """
        self.total_players = total_players
        self.top_player_threshold = int(total_players * top_player_percentile)
        self.quality_weight = quality_weight
        self.tournament_winner_boost = tournament_winner_boost

    def calculate_asymmetric_confidence(
        self,
        player_id: str,
        matches: List[Dict],
        player_ranks: Dict[str, int],
        tournament_wins: Optional[Set[str]] = None,
    ) -> AsymmetricConfidenceMetrics:
        """
        Calculate asymmetric confidence for a player.

        Parameters
        ----------
        player_id : str
            Player to evaluate
        matches : List[Dict]
            Match records involving this player
        player_ranks : Dict[str, int]
            Current rankings (1 = best)
        tournament_wins : Optional[Set[str]]
            Set of tournament IDs this player won

        Returns
        -------
        AsymmetricConfidenceMetrics
            Detailed confidence metrics
        """
        # Initialize tracking
        tournaments = set()
        opponents = set()
        top_opponents = set()
        beaten_players = []
        beaten_by_players = []

        # Process matches
        for match in matches:
            tournaments.add(match.get("tournament_id"))

            # Determine if player won or lost
            if player_id in match.get("winners", []):
                # Player won
                losers = match.get("losers", [])
                for loser in losers:
                    opponents.add(loser)
                    beaten_players.append(loser)
                    if (
                        loser in player_ranks
                        and player_ranks[loser] <= self.top_player_threshold
                    ):
                        top_opponents.add(loser)

            elif player_id in match.get("losers", []):
                # Player lost
                winners = match.get("winners", [])
                for winner in winners:
                    opponents.add(winner)
                    beaten_by_players.append(winner)
                    if (
                        winner in player_ranks
                        and player_ranks[winner] <= self.top_player_threshold
                    ):
                        top_opponents.add(winner)

        # Calculate quality metrics
        top_beaten = sum(
            1
            for p in beaten_players
            if p in player_ranks
            and player_ranks[p] <= self.top_player_threshold
        )

        beaten_by_top = sum(
            1
            for p in beaten_by_players
            if p in player_ranks
            and player_ranks[p] <= self.top_player_threshold
        )

        # Calculate edge quality (average percentile of opponents)
        def calculate_edge_quality(player_list: List[str]) -> float:
            if not player_list:
                return 0.0

            percentiles = []
            for p in player_list:
                if p in player_ranks:
                    percentile = (
                        1.0 - (player_ranks[p] - 1) / self.total_players
                    )
                    percentiles.append(percentile)

            return np.mean(percentiles) if percentiles else 0.0

        incoming_quality = calculate_edge_quality(beaten_by_players)
        outgoing_quality = calculate_edge_quality(beaten_players)

        # Calculate asymmetric confidence components

        # Upward mobility: Can this player rank high?
        # High if: beat top players, high outgoing edge quality
        if beaten_players:
            quality_factor = outgoing_quality ** (1 / self.quality_weight)
            quantity_factor = min(
                len(beaten_players) / 20, 1.0
            )  # Normalize to 20 wins
            top_wins_factor = min(top_beaten / 3, 1.0)  # 3+ top wins = max

            upward_mobility = (
                0.5 * quality_factor
                + 0.2 * quantity_factor
                + 0.3 * top_wins_factor
            )
        else:
            upward_mobility = 0.0

        # Downward pressure: What limits this player's rank?
        # High if: lost to weak players, low incoming edge quality
        if beaten_by_players:
            # Invert quality for downward pressure (losing to weak players = high pressure)
            pressure_quality = 1.0 - incoming_quality
            pressure_quantity = min(
                len(beaten_by_players) / 10, 1.0
            )  # Normalize to 10 losses

            # If only lost to top players, reduce pressure
            if beaten_by_top == len(beaten_by_players) and beaten_by_top > 0:
                downward_pressure = 0.2  # Low pressure - losses are expected
            else:
                downward_pressure = (
                    0.7 * pressure_quality + 0.3 * pressure_quantity
                )
        else:
            # No losses = uncertain downward pressure
            downward_pressure = 0.5

        # Overall confidence combines both signals
        # High upward + low downward = very confident
        # Low upward + high downward = very confident (but low rank)
        # Uncertainty when signals conflict

        confidence_signal = abs(upward_mobility - 0.5) + abs(
            0.5 - downward_pressure
        )
        overall_confidence = min(confidence_signal / 1.0, 1.0)

        # Tournament winner boost
        is_tournament_winner = (
            len(tournaments) <= 3 and top_beaten >= 3 and outgoing_quality > 0.5
        )

        if is_tournament_winner:
            overall_confidence = min(
                overall_confidence * self.tournament_winner_boost, 1.0
            )

        # Determine tier
        if overall_confidence >= 0.8 or is_tournament_winner:
            tier = ConfidenceTier.HIGH
            rankability = "high"
        elif overall_confidence >= 0.6:
            tier = ConfidenceTier.MEDIUM
            rankability = "medium"
        elif overall_confidence >= 0.4:
            tier = ConfidenceTier.LOW
            rankability = "low"
        else:
            tier = ConfidenceTier.PROVISIONAL
            rankability = "unrankable" if len(opponents) < 10 else "low"

        return AsymmetricConfidenceMetrics(
            player_id=player_id,
            tournament_count=len(tournaments),
            match_count=len(matches),
            unique_opponents=len(opponents),
            top_opponents_faced=len(top_opponents),
            top_players_beaten=top_beaten,
            beaten_by_top=beaten_by_top,
            incoming_edge_quality=incoming_quality,
            outgoing_edge_quality=outgoing_quality,
            incoming_edge_count=len(set(beaten_by_players)),
            outgoing_edge_count=len(set(beaten_players)),
            upward_mobility=upward_mobility,
            downward_pressure=downward_pressure,
            overall_confidence=overall_confidence,
            confidence_tier=tier,
            rankability=rankability,
        )

    def identify_special_cases(
        self, all_metrics: List[AsymmetricConfidenceMetrics]
    ) -> Dict[str, List[str]]:
        """
        Identify special player categories.

        Returns
        -------
        Dict mapping category to player IDs
        """
        categories = {
            "tournament_winners": [],  # Few matches, beat top players
            "giant_killers": [],  # Consistently beat players ranked above them
            "gatekeepers": [],  # Many matches, beat newcomers, lose to top
            "isolated_active": [],  # Many matches but no top opponents
            "provisional_strong": [],  # Few matches but all against top players
        }

        for metrics in all_metrics:
            # Tournament winners
            if metrics.is_tournament_winner():
                categories["tournament_winners"].append(metrics.player_id)

            # Giant killers (high upward mobility, few matches)
            elif metrics.upward_mobility > 0.8 and metrics.match_count < 20:
                categories["giant_killers"].append(metrics.player_id)

            # Gatekeepers (medium mobility, high activity)
            elif (
                0.4 < metrics.upward_mobility < 0.7
                and metrics.match_count > 50
                and metrics.top_opponents_faced > 10
            ):
                categories["gatekeepers"].append(metrics.player_id)

            # Isolated active
            elif metrics.is_isolated() and metrics.match_count > 30:
                categories["isolated_active"].append(metrics.player_id)

            # Provisional but strong
            elif (
                metrics.match_count < 10 and metrics.outgoing_edge_quality > 0.7
            ):
                categories["provisional_strong"].append(metrics.player_id)

        return categories
