"""
Tests for the synthetic_data.utils module.

Tests utility functions for synthetic data generation.
"""

import numpy as np
import pytest

from synthetic_data.circuits import (
    TournamentCircuit,
    TournamentConfig,
    TournamentType,
)
from synthetic_data.core import (
    Match,
    PlayerGenerator,
    SyntheticPlayer,
    Team,
    TournamentFormat,
    TournamentGenerator,
)
from synthetic_data.utils.match_score_fixer import (
    add_score_to_match,
    add_scores_to_circuit_results,
    add_scores_to_tournament,
)


class TestMatchScoreFixer:
    """Test match score fixing utilities."""

    def setup_method(self):
        """Set up test data."""
        self.rng = np.random.default_rng(42)
        self.player_gen = PlayerGenerator(seed=42)
        self.tournament_gen = TournamentGenerator(seed=42)

    def test_add_score_to_single_match(self):
        """Test adding score to a single match."""
        # Create teams with different skill levels
        strong_players = [
            SyntheticPlayer(i, f"strong_{i}", 2.0, 0.1) for i in range(1, 5)
        ]
        weak_players = [
            SyntheticPlayer(i, f"weak_{i}", -1.0, 0.2) for i in range(5, 9)
        ]

        strong_team = Team(1, "Strong Team", strong_players)
        weak_team = Team(2, "Weak Team", weak_players)

        # Create a match with strong team winning
        match = Match(
            match_id=1,
            team_a=strong_team,
            team_b=weak_team,
            round_number=1,
            stage="Test",
        )
        match.winner = strong_team

        # Add score
        add_score_to_match(match, self.rng)

        # Should have scores
        assert match.score_a is not None
        assert match.score_b is not None

        # Winner should have score 3
        assert match.score_a == 3

        # Should be a decisive win (skill diff > 2.0)
        assert match.score_b in [0, 1]

    def test_add_score_close_match(self):
        """Test adding score to a close match."""
        # Create teams with similar skill levels
        team1_players = [
            SyntheticPlayer(i, f"p1_{i}", 1.0, 0.1) for i in range(1, 5)
        ]
        team2_players = [
            SyntheticPlayer(i, f"p2_{i}", 0.8, 0.1) for i in range(5, 9)
        ]

        team1 = Team(1, "Team 1", team1_players)
        team2 = Team(2, "Team 2", team2_players)

        # Run multiple simulations to test randomness
        close_scores = []
        for i in range(100):
            match = Match(
                match_id=i,
                team_a=team1,
                team_b=team2,
                round_number=1,
                stage="Test",
            )
            match.winner = team1

            rng = np.random.default_rng(42 + i)
            add_score_to_match(match, rng)

            close_scores.append(match.score_b)

        # Should have mix of 3-2 and 3-1 scores
        assert 2 in close_scores  # Some 3-2 matches
        assert 1 in close_scores  # Some 3-1 matches
        assert close_scores.count(2) > close_scores.count(
            0
        )  # More close than blowouts

    def test_add_score_no_winner(self):
        """Test adding score to match without winner."""
        team1 = Team(1, "Team 1", self.player_gen.generate_players(4))
        team2 = Team(2, "Team 2", self.player_gen.generate_players(4))

        match = Match(
            match_id=1,
            team_a=team1,
            team_b=team2,
            round_number=1,
            stage="Test",
        )
        # No winner set

        add_score_to_match(match, self.rng)

        # Should not add scores
        assert match.score_a is None
        assert match.score_b is None

    def test_add_scores_to_tournament(self):
        """Test adding scores to entire tournament."""
        players = self.player_gen.generate_players(32)
        tournament = self.tournament_gen.generate_tournament(
            players=players,
            format=TournamentFormat.SINGLE_ELIMINATION,
            team_size=4,
        )

        # Simulate matches first
        from synthetic_data.core import MatchSimulator

        simulator = MatchSimulator(seed=42)

        # Simulate all matches in the tournament
        for stage in tournament.stages:
            for round_matches in stage.rounds.values():
                for match in round_matches:
                    simulator.simulate_match(match)

        # Add scores
        add_scores_to_tournament(tournament, self.rng)

        # Check all completed matches have scores
        for stage in tournament.stages:
            for round_matches in stage.rounds.values():
                for match in round_matches:
                    if match.winner is not None:
                        assert match.score_a is not None
                        assert match.score_b is not None
                        assert match.score_a + match.score_b >= 3

    def test_add_scores_to_double_elimination(self):
        """Test adding scores to double elimination tournament."""
        players = self.player_gen.generate_players(32)
        tournament = self.tournament_gen.generate_tournament(
            players=players,
            format=TournamentFormat.DOUBLE_ELIMINATION,
            team_size=4,
        )

        # Simulate matches
        from synthetic_data.core import MatchSimulator

        simulator = MatchSimulator(seed=42)

        # Simulate all matches manually
        for stage in tournament.stages:
            # Process regular rounds
            for round_matches in stage.rounds.values():
                for match in round_matches:
                    simulator.simulate_match(match)

            # Process winners bracket
            for round_matches in stage.winners_bracket.values():
                for match in round_matches:
                    simulator.simulate_match(match)

            # Process losers bracket
            for round_matches in stage.losers_bracket.values():
                for match in round_matches:
                    simulator.simulate_match(match)

            # Process grand finals
            for match in stage.grand_finals:
                simulator.simulate_match(match)

        # Add scores
        add_scores_to_tournament(tournament, self.rng)

        # Check winners bracket
        for stage in tournament.stages:
            for round_matches in stage.winners_bracket.values():
                for match in round_matches:
                    if match.winner is not None:
                        assert match.score_a is not None
                        assert match.score_b is not None

            # Check losers bracket
            for round_matches in stage.losers_bracket.values():
                for match in round_matches:
                    if match.winner is not None:
                        assert match.score_a is not None
                        assert match.score_b is not None

            # Check grand finals
            for match in stage.grand_finals:
                if match.winner is not None:
                    assert match.score_a is not None
                    assert match.score_b is not None

    def test_add_scores_to_circuit_results(self):
        """Test adding scores to full circuit results."""
        circuit = TournamentCircuit(seed=42, player_pool_size=50)

        configs = [
            TournamentConfig(
                name=f"Tournament {i}",
                tournament_type=TournamentType.OPEN,
                format=TournamentFormat.SWISS,
                n_teams=8,
                swiss_rounds=3,
            )
            for i in range(3)
        ]

        # Generate circuit (matches are automatically simulated)
        results = circuit.generate_circuit(configs)

        # Add scores
        add_scores_to_circuit_results(results, self.rng)

        # Check all tournaments have scores
        for tournament in results.tournaments:
            has_scores = False
            for stage in tournament.stages:
                for round_matches in stage.rounds.values():
                    for match in round_matches:
                        if (
                            match.winner is not None
                            and match.score_a is not None
                        ):
                            has_scores = True
                            break
            assert has_scores

    def test_score_distribution(self):
        """Test that score distributions are realistic."""
        # Generate many matches with varying skill differences
        scores = {"3-0": 0, "3-1": 0, "3-2": 0}

        for i in range(1000):
            skill_diff = self.rng.uniform(0, 4)  # 0 to 4 skill difference

            team1_players = [
                SyntheticPlayer(j, f"t1_p{j}", 2.0, 0.1) for j in range(4)
            ]
            team2_players = [
                SyntheticPlayer(j, f"t2_p{j}", 2.0 - skill_diff, 0.1)
                for j in range(4, 8)
            ]

            team1 = Team(1, "Team 1", team1_players)
            team2 = Team(2, "Team 2", team2_players)

            match = Match(
                match_id=i,
                team_a=team1,
                team_b=team2,
                round_number=1,
                stage="Test",
            )
            match.winner = team1

            rng = np.random.default_rng(i)
            add_score_to_match(match, rng)

            score_str = f"{match.score_a}-{match.score_b}"
            if score_str in scores:
                scores[score_str] += 1

        # Should have all score types
        assert scores["3-0"] > 0
        assert scores["3-1"] > 0
        assert scores["3-2"] > 0

        # Should have reasonable distribution
        total = sum(scores.values())
        assert scores["3-0"] / total > 0.2  # At least 20% blowouts
        assert scores["3-2"] / total > 0.1  # At least 10% close matches

    def test_deterministic_with_seed(self):
        """Test that score generation is deterministic with same seed."""
        team1 = Team(1, "Team 1", self.player_gen.generate_players(4))
        team2 = Team(2, "Team 2", self.player_gen.generate_players(4))

        # Create two identical matches
        match1 = Match(
            match_id=1,
            team_a=team1,
            team_b=team2,
            round_number=1,
            stage="Test",
        )
        match1.winner = team1

        match2 = Match(
            match_id=1,
            team_a=team1,
            team_b=team2,
            round_number=1,
            stage="Test",
        )
        match2.winner = team1

        # Add scores with same seed
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)

        add_score_to_match(match1, rng1)
        add_score_to_match(match2, rng2)

        # Should have same scores
        assert match1.score_a == match2.score_a
        assert match1.score_b == match2.score_b
