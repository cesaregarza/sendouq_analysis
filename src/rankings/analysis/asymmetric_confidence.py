"""
Asymmetric confidence scoring that considers edge quality over quantity.

Key insight: A player who beats top players is immediately rankable,
while a player with many matches against weak opponents remains uncertain.

Optimized version with:
- Pre-computed top player set for O(1) lookups
- Pre-computed player percentiles for fast edge quality calculation
- Single pass through matches
- Support for batch processing
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np

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
    Optimized calculation of confidence considering edge quality and asymmetry.

    Core principles:
    1. Beating top players provides strong upward signal
    2. Losing to weak players provides strong downward signal
    3. Few high-quality matches > many low-quality matches
    4. Tournament winners are immediately rankable

    Optimizations:
    - Pre-computed top player set for O(1) lookups
    - Pre-computed player percentiles for fast edge quality calculation
    - Single pass through matches
    - Support for batch processing
    """

    def __init__(
        self,
        total_players: int,
        player_ranks: Dict[str, int],
        top_player_percentile: float = 0.1,
        quality_weight: float = 2.0,
        tournament_winner_boost: float = 3.0,
    ):
        """
        Initialize optimized calculator with pre-computed data.

        Parameters
        ----------
        total_players : int
            Total number of players
        player_ranks : Dict[str, int]
            Current rankings (1 = best) - required for optimization
        top_player_percentile : float
            Percentile for "top" players (default 0.1 = top 10%)
        quality_weight : float
            How much to weight opponent quality vs quantity
        tournament_winner_boost : float
            Confidence multiplier for tournament winners
        """
        self.total_players = total_players
        self.player_ranks = player_ranks
        self.top_player_threshold = int(total_players * top_player_percentile)
        self.quality_weight = quality_weight
        self.tournament_winner_boost = tournament_winner_boost

        # Pre-compute top player set for O(1) lookups
        self.top_players = {
            p
            for p, rank in player_ranks.items()
            if rank <= self.top_player_threshold
        }

        # Pre-compute player percentiles for faster edge quality
        self.player_percentiles = {}
        for player, rank in player_ranks.items():
            self.player_percentiles[player] = 1.0 - (rank - 1) / total_players

    def calculate_asymmetric_confidence(
        self,
        player_id: str,
        matches: List[Dict],
        tournament_wins: Optional[Set[str]] = None,
    ) -> AsymmetricConfidenceMetrics:
        """
        Calculate asymmetric confidence for a player (optimized).

        Parameters
        ----------
        player_id : str
            Player to evaluate
        matches : List[Dict]
            Match records involving this player
        tournament_wins : Optional[Set[str]]
            Set of tournament IDs this player won

        Returns
        -------
        AsymmetricConfidenceMetrics
            Detailed confidence metrics
        """
        # Initialize tracking with better data structures
        tournaments = set()
        beaten_players = []
        beaten_by_players = []

        # Single pass through matches
        for match in matches:
            tournaments.add(match.get("tournament_id"))
            winners = match.get("winners", [])
            losers = match.get("losers", [])

            if player_id in winners:
                beaten_players.extend(losers)
            elif player_id in losers:
                beaten_by_players.extend(winners)

        # Deduplicate opponents
        unique_beaten = set(beaten_players)
        unique_beaten_by = set(beaten_by_players)
        all_opponents = unique_beaten | unique_beaten_by

        # Fast top player counting using pre-computed set
        # Count all occurrences (including duplicates) to match original behavior
        top_beaten = sum(1 for p in beaten_players if p in self.top_players)
        beaten_by_top = sum(
            1 for p in beaten_by_players if p in self.top_players
        )
        top_opponents = len(all_opponents & self.top_players)

        # Optimized edge quality calculation using pre-computed percentiles
        outgoing_quality = self._calculate_edge_quality_fast(beaten_players)
        incoming_quality = self._calculate_edge_quality_fast(beaten_by_players)

        # Calculate asymmetric confidence components
        # Upward mobility
        if beaten_players:
            quality_factor = outgoing_quality ** (1 / self.quality_weight)
            quantity_factor = min(len(beaten_players) / 20, 1.0)
            top_wins_factor = min(top_beaten / 3, 1.0)

            upward_mobility = (
                0.5 * quality_factor
                + 0.2 * quantity_factor
                + 0.3 * top_wins_factor
            )
        else:
            upward_mobility = 0.0

        # Downward pressure
        if beaten_by_players:
            pressure_quality = 1.0 - incoming_quality
            pressure_quantity = min(len(beaten_by_players) / 10, 1.0)

            # Special case: only lost to top players
            # Check total occurrences, not unique players
            if beaten_by_top == len(beaten_by_players) and beaten_by_top > 0:
                downward_pressure = 0.2
            else:
                downward_pressure = (
                    0.7 * pressure_quality + 0.3 * pressure_quantity
                )
        else:
            downward_pressure = 0.5

        # Overall confidence
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
            rankability = "unrankable" if len(all_opponents) < 10 else "low"

        return AsymmetricConfidenceMetrics(
            player_id=player_id,
            tournament_count=len(tournaments),
            match_count=len(matches),
            unique_opponents=len(all_opponents),
            top_opponents_faced=top_opponents,
            top_players_beaten=top_beaten,
            beaten_by_top=beaten_by_top,
            incoming_edge_quality=incoming_quality,
            outgoing_edge_quality=outgoing_quality,
            incoming_edge_count=len(unique_beaten_by),
            outgoing_edge_count=len(unique_beaten),
            upward_mobility=upward_mobility,
            downward_pressure=downward_pressure,
            overall_confidence=overall_confidence,
            confidence_tier=tier,
            rankability=rankability,
        )

    def _calculate_edge_quality_fast(self, player_list: List[str]) -> float:
        """
        Fast edge quality calculation using pre-computed percentiles.

        Parameters
        ----------
        player_list : List[str]
            List of player IDs (may contain duplicates)

        Returns
        -------
        float
            Average percentile of opponents (0-1)
        """
        if not player_list:
            return 0.0

        # Use list comprehension with pre-computed percentiles
        percentiles = [
            self.player_percentiles[p]
            for p in player_list
            if p in self.player_percentiles
        ]

        if not percentiles:
            return 0.0

        # Always use numpy mean to match original
        return np.mean(percentiles)

    def calculate_batch(
        self,
        player_matches: Dict[str, List[Dict]],
        tournament_wins: Optional[Dict[str, Set[str]]] = None,
        n_workers: int = 1,
    ) -> Dict[str, AsymmetricConfidenceMetrics]:
        """
        Calculate confidence for multiple players efficiently.

        Parameters
        ----------
        player_matches : Dict[str, List[Dict]]
            Mapping of player_id to their matches
        tournament_wins : Optional[Dict[str, Set[str]]]
            Mapping of player_id to tournament IDs they won
        n_workers : int
            Number of parallel workers (future enhancement)

        Returns
        -------
        Dict[str, AsymmetricConfidenceMetrics]
            Mapping of player_id to their metrics
        """
        results = {}

        for player_id, matches in player_matches.items():
            wins = tournament_wins.get(player_id) if tournament_wins else None
            results[player_id] = self.calculate_asymmetric_confidence(
                player_id, matches, wins
            )

        return results

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
            "tournament_winners": [],
            "giant_killers": [],
            "gatekeepers": [],
            "isolated_active": [],
            "provisional_strong": [],
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
