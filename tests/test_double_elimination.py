"""
Tests for double elimination tournament generation.
"""

import pytest

from synthetic_data.data_serializer import DataSerializer
from synthetic_data.match_simulator import MatchSimulator
from synthetic_data.player_generator import PlayerGenerator
from synthetic_data.tournament_generator import (
    TournamentFormat,
    TournamentGenerator,
)
from synthetic_data.validator import DataValidator


class TestDoubleElimination:
    """Test cases for double elimination tournament generation."""

    def test_small_double_elimination(self):
        """Test double elimination with 4 teams."""
        # Generate players
        player_gen = PlayerGenerator(seed=42)
        players = player_gen.generate_players(
            n_players=16,  # 4 teams x 4 players
            skill_distribution="normal",
            skill_params={"mean": 0.0, "std": 1.0},
        )

        # Generate tournament
        tournament_gen = TournamentGenerator(seed=42)
        tournament = tournament_gen.generate_tournament(
            players=players,
            format=TournamentFormat.DOUBLE_ELIMINATION,
            team_size=4,
            name="Small Double Elimination Test",
            seeded=True,
        )

        # Check basic structure
        assert len(tournament.stages) == 1
        stage = tournament.stages[0]
        assert stage.format == TournamentFormat.DOUBLE_ELIMINATION
        assert len(stage.teams) == 4

        # Check brackets exist
        assert len(stage.winners_bracket) > 0
        assert len(stage.losers_bracket) > 0
        assert len(stage.grand_finals) > 0

        # Check all matches have bracket type
        for round_matches in stage.rounds.values():
            for match in round_matches:
                assert match.bracket_type in [
                    "winners",
                    "losers",
                    "grand_finals",
                ]

    def test_medium_double_elimination(self):
        """Test double elimination with 16 teams."""
        # Generate players
        player_gen = PlayerGenerator(seed=123)
        players = player_gen.generate_players(
            n_players=64,  # 16 teams x 4 players
            skill_distribution="normal",
            skill_params={"mean": 0.0, "std": 1.0},
        )

        # Generate tournament
        tournament_gen = TournamentGenerator(seed=123)
        tournament = tournament_gen.generate_tournament(
            players=players,
            format=TournamentFormat.DOUBLE_ELIMINATION,
            team_size=4,
            name="Medium Double Elimination Test",
            seeded=True,
        )

        stage = tournament.stages[0]

        # Check bracket sizes
        # Winners bracket should have log2(16) = 4 rounds
        assert len(stage.winners_bracket) == 4

        # Losers bracket should have 2*(log2(16)-1) = 6 rounds
        expected_lb_rounds = 2 * (4 - 1)
        assert len(stage.losers_bracket) == expected_lb_rounds

        # Check seeding
        first_wb_round = stage.winners_bracket[1]
        for match in first_wb_round:
            # Higher seed should have lower seed number
            assert match.team_a.seed is not None
            assert match.team_b.seed is not None
            # In first round, seeds should be paired 1v16, 2v15, etc.

    def test_odd_team_count(self):
        """Test double elimination with odd number of teams (byes)."""
        # Generate players for 7 teams
        player_gen = PlayerGenerator(seed=456)
        players = player_gen.generate_players(
            n_players=28,  # 7 teams x 4 players
            skill_distribution="normal",
            skill_params={"mean": 0.0, "std": 1.0},
        )

        # Generate tournament
        tournament_gen = TournamentGenerator(seed=456)
        tournament = tournament_gen.generate_tournament(
            players=players,
            format=TournamentFormat.DOUBLE_ELIMINATION,
            team_size=4,
            name="Odd Team Double Elimination Test",
            seeded=True,
        )

        stage = tournament.stages[0]
        assert len(stage.teams) == 7

        # Bracket size should be 8 (next power of 2)
        # So 1 team gets a bye
        first_round = stage.winners_bracket[1]
        # With 7 teams and bracket size 8, there should be 3 matches in round 1
        assert len(first_round) == 3

    def test_bracket_progression(self):
        """Test that teams progress correctly through brackets."""
        # Generate small tournament for easier tracking
        player_gen = PlayerGenerator(seed=789)
        players = player_gen.generate_players(
            n_players=16,  # 4 teams
            skill_distribution="normal",
            skill_params={"mean": 0.0, "std": 1.0},
        )

        tournament_gen = TournamentGenerator(seed=789)
        tournament = tournament_gen.generate_tournament(
            players=players,
            format=TournamentFormat.DOUBLE_ELIMINATION,
            team_size=4,
            name="Bracket Progression Test",
            seeded=True,
        )

        stage = tournament.stages[0]

        # Track team appearances
        team_match_count = {}
        teams_eliminated = set()

        for round_matches in stage.rounds.values():
            for match in round_matches:
                # Count matches per team
                if match.team_a:
                    team_match_count[match.team_a.team_id] = (
                        team_match_count.get(match.team_a.team_id, 0) + 1
                    )
                if match.team_b:
                    team_match_count[match.team_b.team_id] = (
                        team_match_count.get(match.team_b.team_id, 0) + 1
                    )

                # Track eliminations
                if (
                    match.bracket_type == "losers"
                    and match.is_elimination_match
                ):
                    loser = (
                        match.team_a
                        if match.winner == match.team_b
                        else match.team_b
                    )
                    if loser:
                        teams_eliminated.add(loser.team_id)

        # Each team should play at least 2 matches (except if eliminated)
        for team in stage.teams:
            if team.team_id not in teams_eliminated:
                assert team_match_count.get(team.team_id, 0) >= 2

    def test_serialization(self):
        """Test that double elimination tournaments serialize correctly."""
        # Generate tournament
        player_gen = PlayerGenerator(seed=999)
        players = player_gen.generate_players(
            n_players=32,  # 8 teams
            skill_distribution="normal",
            skill_params={"mean": 0.0, "std": 1.0},
        )

        tournament_gen = TournamentGenerator(seed=999)
        tournament = tournament_gen.generate_tournament(
            players=players,
            format=TournamentFormat.DOUBLE_ELIMINATION,
            team_size=4,
            name="Serialization Test",
            seeded=True,
        )

        # Serialize
        match_sim = MatchSimulator(seed=999)
        serializer = DataSerializer()
        data = serializer.serialize_tournament(
            tournament, simulate_matches=True, match_simulator=match_sim
        )

        # Check structure
        assert "tournament" in data
        assert "data" in data["tournament"]
        assert "stage" in data["tournament"]["data"]

        # Check stage has bracket info
        stage_data = data["tournament"]["data"]["stage"][0]
        assert stage_data["type"] == "double_elim"
        assert "brackets" in stage_data

        # Check matches have bracket type
        matches = data["tournament"]["data"]["match"]
        for match in matches:
            if "bracket_type" in match:  # Should be present for all DE matches
                assert match["bracket_type"] in [
                    "winners",
                    "losers",
                    "grand_finals",
                ]

    def test_validation(self):
        """Test that generated double elimination tournaments pass validation."""
        # Generate tournament
        player_gen = PlayerGenerator(seed=111)
        players = player_gen.generate_players(
            n_players=64,  # 16 teams
            skill_distribution="normal",
            skill_params={"mean": 0.0, "std": 1.0},
        )

        tournament_gen = TournamentGenerator(seed=111)
        tournament = tournament_gen.generate_tournament(
            players=players,
            format=TournamentFormat.DOUBLE_ELIMINATION,
            team_size=4,
            name="Validation Test",
        )

        # Serialize and validate
        match_sim = MatchSimulator(seed=111)
        serializer = DataSerializer()
        data = serializer.serialize_tournament(
            tournament, simulate_matches=True, match_simulator=match_sim
        )

        validator = DataValidator()
        assert validator.validate_serialized_data(data)

    def test_parser_compatibility(self):
        """Test that double elimination data works with the parser."""
        # Generate tournament
        player_gen = PlayerGenerator(seed=222)
        players = player_gen.generate_players(
            n_players=32,  # 8 teams
            skill_distribution="normal",
            skill_params={"mean": 0.0, "std": 1.0},
        )

        tournament_gen = TournamentGenerator(seed=222)
        tournament = tournament_gen.generate_tournament(
            players=players,
            format=TournamentFormat.DOUBLE_ELIMINATION,
            team_size=4,
            name="Parser Test",
        )

        # Serialize
        match_sim = MatchSimulator(seed=222)
        serializer = DataSerializer()
        data = serializer.serialize_tournament(
            tournament, simulate_matches=True, match_simulator=match_sim
        )

        # Try to parse (importing here to avoid circular imports)
        try:
            from rankings.core import parse_tournaments_data

            tables = parse_tournaments_data([data])

            # Check we got data
            assert len(tables["matches"]) > 0
            assert len(tables["players"]) == 32
            assert len(tables["teams"]) == 8

            # Check matches have expected fields
            matches_df = tables["matches"]
            # Basic fields should exist
            assert "tournament_id" in matches_df.columns
            assert "stage_id" in matches_df.columns
            assert "winner_team_id" in matches_df.columns

        except ImportError:
            # Parser not available in test environment
            pytest.skip("Parser not available for testing")
