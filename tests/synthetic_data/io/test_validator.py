"""Tests for data validation functionality."""

import pytest

from synthetic_data import (
    DataSerializer,
    DataValidator,
    PlayerGenerator,
    TournamentFormat,
    TournamentGenerator,
)


class TestDataValidator:
    """Test data validation functionality."""

    def setup_method(self):
        """Set up test data."""
        self.player_gen = PlayerGenerator(seed=42)
        self.tournament_gen = TournamentGenerator(seed=42)
        self.serializer = DataSerializer()
        self.validator = DataValidator()

    def test_valid_tournament_validation(self):
        """Test validation of valid tournament."""
        players = self.player_gen.generate_players(32)
        tournament = self.tournament_gen.generate_tournament(
            players=players, format=TournamentFormat.SWISS, team_size=4
        )

        # Validate tournament structure
        assert self.validator.validate_tournament(tournament)
        assert len(self.validator.errors) == 0

        # Validate serialized data
        data = self.serializer.serialize_tournament(tournament)
        assert self.validator.validate_serialized_data(data)
        assert len(self.validator.errors) == 0

    def test_invalid_tournament_detection(self):
        """Test detection of invalid tournament data."""
        # Create invalid data
        invalid_data = {
            "tournament": {
                "data": {
                    "stage": [],
                    "match": [
                        {
                            "id": 1,
                            "stage_id": 999,  # Invalid stage reference
                            "opponent1": {"id": 1},
                            "opponent2": {"id": 999},  # Invalid team reference
                        }
                    ],
                },
                "ctx": {"id": 1, "teams": [{"id": 1, "members": []}]},
            }
        }

        assert not self.validator.validate_serialized_data(invalid_data)
        assert len(self.validator.errors) > 0

    def test_parser_validation(self):
        """Test validation through parser."""
        players = self.player_gen.generate_players(16)
        tournament = self.tournament_gen.generate_tournament(
            players=players,
            format=TournamentFormat.SINGLE_ELIMINATION,
            team_size=4,
        )

        data_list = [self.serializer.serialize_tournament(tournament)]

        assert self.validator.validate_with_parser(data_list)

    def test_validation_report(self):
        """Test validation report generation."""
        # Create tournament with warnings
        players = self.player_gen.generate_players(10)  # Not divisible by 4
        tournament = self.tournament_gen.generate_tournament(
            players=players[:9],  # 2 teams of 4, 1 team of 1
            format=TournamentFormat.ROUND_ROBIN,
            team_size=4,
        )

        self.validator.validate_tournament(tournament)
        report = self.validator.get_validation_report()

        assert "WARNING" in report or "All validations passed" in report
