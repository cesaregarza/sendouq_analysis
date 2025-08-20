"""Monte Carlo simulation for tournament outcomes."""

from typing import Any, Dict, List

import numpy as np

from rankings.tournament_prediction.match_predictor import MatchPredictor
from rankings.tournament_prediction.team_strength import TeamStrengthCalculator


class MonteCarloSimulator:
    """Simulate tournament outcomes using Monte Carlo methods."""

    def __init__(
        self,
        team_calculator: TeamStrengthCalculator,
        match_predictor: MatchPredictor,
    ):
        """Initialize simulator with calculators.

        Args:
            team_calculator: Calculator for team strengths
            match_predictor: Predictor for match outcomes
        """
        self.team_calculator = team_calculator
        self.match_predictor = match_predictor

    def simulate_match(
        self, team_a_rating: float, team_b_rating: float, best_of: int = 1
    ) -> bool:
        """Simulate a single match or series.

        Args:
            team_a_rating: Team A's log-rating
            team_b_rating: Team B's log-rating
            best_of: Number of games in series (1, 3, 5, etc.)

        Returns:
            True if team A wins, False if team B wins
        """
        if best_of == 1:
            prob_a = self.match_predictor.win_probability(
                team_a_rating, team_b_rating
            )
            return np.random.random() < prob_a

        # Simulate best-of series
        wins_needed = (best_of + 1) // 2
        a_wins = 0
        b_wins = 0

        while a_wins < wins_needed and b_wins < wins_needed:
            prob_a = self.match_predictor.win_probability(
                team_a_rating, team_b_rating
            )
            if np.random.random() < prob_a:
                a_wins += 1
            else:
                b_wins += 1

        return a_wins >= wins_needed

    def simulate_tournament(
        self,
        player_ratings: Dict[int, float],
        bracket_structure: Dict[str, Any],
        team_rosters: Dict[int, List[int]],
        n_simulations: int = 10000,
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation of tournament.

        Args:
            player_ratings: Dictionary mapping player_id to log-rating
            bracket_structure: Tournament bracket structure
            team_rosters: Dictionary mapping team_id to list of player_ids
            n_simulations: Number of simulations to run

        Returns:
            Dictionary with simulation results (win probabilities, expected placements, etc.)
        """
        # Calculate team ratings once
        team_ratings = {}
        for team_id, roster in team_rosters.items():
            team_ratings[team_id] = self.team_calculator.team_log_rating(
                player_ratings, roster
            )

        # Track results across simulations
        win_counts = {team_id: 0 for team_id in team_rosters}
        top4_counts = {team_id: 0 for team_id in team_rosters}
        placement_sum = {team_id: 0 for team_id in team_rosters}

        # Run simulations
        for _ in range(n_simulations):
            # Simulate tournament (simplified - real implementation would follow bracket)
            # This is a placeholder for actual bracket simulation
            placements = self._simulate_single_tournament(
                team_ratings, bracket_structure
            )

            for team_id, placement in placements.items():
                if placement == 1:
                    win_counts[team_id] += 1
                if placement <= 4:
                    top4_counts[team_id] += 1
                placement_sum[team_id] += placement

        # Calculate probabilities and expectations
        results = {}
        for team_id in team_rosters:
            results[team_id] = {
                "win_probability": win_counts[team_id] / n_simulations,
                "top4_probability": top4_counts[team_id] / n_simulations,
                "expected_placement": placement_sum[team_id] / n_simulations,
                "rating": team_ratings[team_id],
            }

        return results

    def _simulate_single_tournament(
        self, team_ratings: Dict[int, float], bracket_structure: Dict[str, Any]
    ) -> Dict[int, int]:
        """Simulate a single tournament run.

        Args:
            team_ratings: Pre-calculated team ratings
            bracket_structure: Tournament bracket structure

        Returns:
            Dictionary mapping team_id to final placement
        """
        # Placeholder for actual bracket simulation
        # Real implementation would follow the bracket structure
        # For now, just rank by rating with some randomness

        teams = list(team_ratings.keys())
        # Add noise to ratings for simulation
        noisy_ratings = {
            team: rating + np.random.normal(0, 0.5)
            for team, rating in team_ratings.items()
        }

        # Sort by noisy rating
        sorted_teams = sorted(
            teams, key=lambda t: noisy_ratings[t], reverse=True
        )

        # Assign placements
        placements = {team: i + 1 for i, team in enumerate(sorted_teams)}
        return placements
