"""Tests for match simulation functionality."""

import pytest

from synthetic_data import (
    Match,
    MatchSimulator,
    PlayerGenerator,
    SyntheticPlayer,
    Team,
)


class TestMatchSimulator:
    """Test match simulation functionality."""

    def setup_method(self):
        """Set up test data."""
        self.player_gen = PlayerGenerator(seed=42)
        self.sim = MatchSimulator(seed=42)

        # Create teams with different skill levels
        high_skill_players = [
            SyntheticPlayer(i, f"high_{i}", 2.0, 0.1) for i in range(1, 5)
        ]
        low_skill_players = [
            SyntheticPlayer(i, f"low_{i}", -1.0, 0.2) for i in range(5, 9)
        ]

        self.strong_team = Team(1, "Strong Team", high_skill_players)
        self.weak_team = Team(2, "Weak Team", low_skill_players)

    def test_match_simulation_basic(self):
        """Test basic match simulation."""
        match = Match(
            match_id=1,
            team_a=self.strong_team,
            team_b=self.weak_team,
            round_number=1,
            stage="Test",
        )

        result = self.sim.simulate_match(match)

        assert result.winner is not None
        assert result.score_a is not None
        assert result.score_b is not None
        assert result.score_a + result.score_b >= 4  # Best of 7

    def test_match_probability_calculation(self):
        """Test match probability calculations."""
        prob_strong_wins = self.sim.calculate_match_probability(
            self.strong_team, self.weak_team
        )

        # Strong team should have high win probability
        assert 0.8 < prob_strong_wins < 1.0

        # Equal teams should have ~50% probability
        equal_team1 = Team(3, "Equal 1", self.strong_team.players[:4])
        equal_team2 = Team(4, "Equal 2", self.strong_team.players[:4])
        prob_equal = self.sim.calculate_match_probability(
            equal_team1, equal_team2
        )
        assert 0.45 < prob_equal < 0.55

    def test_upset_generation(self):
        """Test upset generation."""
        # Run many simulations to test upset rate
        upset_count = 0
        n_sims = 100

        for i in range(n_sims):
            match = Match(
                match_id=i,
                team_a=self.strong_team,
                team_b=self.weak_team,
                round_number=1,
                stage="Test",
            )

            result = self.sim.generate_upset(match, upset_probability=0.3)
            if result.winner == self.weak_team:
                upset_count += 1

        # Should have some upsets but not too many
        assert 10 < upset_count < 50

    def test_bradley_terry_integration(self):
        """Test Bradley-Terry model integration."""
        # Test that stronger teams win more often
        wins = {"strong": 0, "weak": 0}

        for i in range(100):
            match = Match(
                match_id=i,
                team_a=self.strong_team,
                team_b=self.weak_team,
                round_number=1,
                stage="Test",
            )

            # Reset simulator seed for each match
            sim = MatchSimulator(seed=42 + i)
            result = sim.simulate_match(match)

            if result.winner == self.strong_team:
                wins["strong"] += 1
            else:
                wins["weak"] += 1

        # Strong team should win significantly more
        assert wins["strong"] > wins["weak"] * 2
