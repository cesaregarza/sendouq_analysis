"""Tournament seeding generation from team ratings."""

from __future__ import annotations

import math
from typing import Any

from rankings.tournament_prediction.match_predictor import MatchPredictor
from rankings.tournament_prediction.team_strength import TeamStrengthCalculator


class TournamentSeeder:
    """Generate tournament seedings from team ratings."""

    def __init__(
        self,
        team_calculator: TeamStrengthCalculator,
        match_predictor: MatchPredictor | None = None,
    ):
        """Initialize seeder with rating calculator and predictor.

        Args:
            team_calculator: Calculator for team strengths
            match_predictor: Optional predictor for confidence intervals
        """
        self.team_calculator = team_calculator
        self.match_predictor = match_predictor

    def generate_seeds(
        self,
        player_ratings: dict[int, float],
        team_rosters: dict[int, list[int]],
        exposure_weights: dict[tuple[int, int], float] | None = None,
    ) -> list[tuple[int, float, float]]:
        """Generate tournament seeds from team ratings.

        Args:
            player_ratings: Dictionary mapping player_id to log-rating
            team_rosters: Dictionary mapping team_id to list of player_ids
            exposure_weights: Optional weights as (player_id, team_id) -> weight

        Returns:
            List of (team_id, rating, confidence) sorted by rating
        """
        team_ratings = []

        for team_id, roster in team_rosters.items():
            # Get weights for this team
            team_weights = None
            if exposure_weights:
                team_weights = {
                    p: exposure_weights.get((p, team_id), 1.0) for p in roster
                }

            # Calculate team rating
            rating = self.team_calculator.team_log_rating(
                player_ratings, roster, team_weights
            )

            # Calculate confidence (gap to neighbors will be computed after sorting)
            team_ratings.append((team_id, rating, 0.0))

        # Sort by rating (descending)
        team_ratings.sort(key=lambda x: x[1], reverse=True)

        # Calculate confidence based on rating gaps
        result = []
        for i, (team_id, rating, _) in enumerate(team_ratings):
            # Confidence based on gap to neighbors
            gap_above = (
                float("inf") if i == 0 else abs(rating - team_ratings[i - 1][1])
            )
            gap_below = (
                float("inf")
                if i == len(team_ratings) - 1
                else abs(rating - team_ratings[i + 1][1])
            )

            min_gap = min(gap_above, gap_below)
            # Normalize confidence to 0-1 scale (larger gap = higher confidence)
            confidence = 1.0 - math.exp(-min_gap)

            result.append((team_id, rating, confidence))

        return result

    def apply_bracket_constraints(
        self,
        seeds: list[tuple[int, float, float]],
        constraints: dict[str, Any] | None = None,
    ) -> list[tuple[int, float, float]]:
        """Apply bracket-specific constraints to seeding.

        Args:
            seeds: Initial seeds from generate_seeds
            constraints: Dictionary of constraints (e.g., region separation)

        Returns:
            Adjusted seeds respecting constraints
        """
        # This is a placeholder for bracket-specific logic
        # Real implementation would handle:
        # - Snake seeding
        # - Region separation
        # - Avoiding early rematches
        # - Format-specific rules

        return seeds
