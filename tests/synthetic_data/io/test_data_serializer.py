"""Tests for data serialization functionality."""

import pytest

from rankings.core import parse_tournaments_data
from synthetic_data import (
    DataSerializer,
    MatchSimulator,
    PlayerGenerator,
    TournamentFormat,
    TournamentGenerator,
)


class TestDataSerializer:
    """Test data serialization functionality."""

    def setup_method(self):
        """Set up test data."""
        self.player_gen = PlayerGenerator(seed=42)
        self.tournament_gen = TournamentGenerator(seed=42)
        self.serializer = DataSerializer()
        self.match_sim = MatchSimulator(seed=42)

    def test_tournament_serialization(self):
        """Test basic tournament serialization."""
        players = self.player_gen.generate_players(16)
        tournament = self.tournament_gen.generate_tournament(
            players=players,
            format=TournamentFormat.SWISS,
            team_size=4,
            n_rounds=3,
        )

        data = self.serializer.serialize_tournament(
            tournament, simulate_matches=True, match_simulator=self.match_sim
        )

        # Check structure
        assert "tournament" in data
        assert "data" in data["tournament"]
        assert "ctx" in data["tournament"]

        # Check data sections
        tournament_data = data["tournament"]["data"]
        assert "stage" in tournament_data
        assert "round" in tournament_data
        assert "match" in tournament_data
        assert "group" in tournament_data

        # Check context
        ctx = data["tournament"]["ctx"]
        assert "id" in ctx
        assert "teams" in ctx
        assert len(ctx["teams"]) == 4  # 16 players / 4 per team

    def test_match_serialization(self):
        """Test match serialization details."""
        players = self.player_gen.generate_players(8)
        tournament = self.tournament_gen.generate_tournament(
            players=players,
            format=TournamentFormat.SINGLE_ELIMINATION,
            team_size=4,
        )

        data = self.serializer.serialize_tournament(
            tournament, simulate_matches=True, match_simulator=self.match_sim
        )

        matches = data["tournament"]["data"]["match"]
        assert len(matches) > 0

        # Check match structure
        match = matches[0]
        assert "id" in match
        assert "stage_id" in match
        assert "round_id" in match
        assert "opponent1" in match
        assert "opponent2" in match
        assert "status" in match

        # Check completed match has results
        if match["status"] == "completed":
            assert match["opponent1"]["result"] in ["win", "loss"]
            assert match["opponent2"]["result"] in ["win", "loss"]
            assert match["opponent1"]["score"] is not None
            assert match["opponent2"]["score"] is not None

    def test_parser_compatibility(self):
        """Test compatibility with existing parser."""
        players = self.player_gen.generate_players(32)
        tournament = self.tournament_gen.generate_tournament(
            players=players, format=TournamentFormat.ROUND_ROBIN, team_size=4
        )

        data_list = [
            self.serializer.serialize_tournament(
                tournament,
                simulate_matches=True,
                match_simulator=self.match_sim,
            )
        ]

        # Should parse without errors
        tables = parse_tournaments_data(data_list)

        assert tables["matches"] is not None
        assert tables["teams"] is not None
        assert tables["players"] is not None
        assert len(tables["matches"]) > 0
        assert len(tables["teams"]) == 8
        assert len(tables["players"]) == 32
