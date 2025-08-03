"""Tests for tournament structure generation."""

import pytest

from synthetic_data import (
    PlayerGenerator,
    Team,
    TournamentFormat,
    TournamentGenerator,
)


class TestTournamentGenerator:
    """Test tournament structure generation."""

    def setup_method(self):
        """Set up test data."""
        self.player_gen = PlayerGenerator(seed=42)
        self.players = self.player_gen.generate_players(64)  # 16 teams
        self.tournament_gen = TournamentGenerator(seed=42)

    def test_swiss_generation(self):
        """Test Swiss tournament generation."""
        tournament = self.tournament_gen.generate_tournament(
            players=self.players,
            format=TournamentFormat.SWISS,
            team_size=4,
            n_rounds=5,
        )

        assert tournament.tournament_id > 0
        assert len(tournament.all_teams) == 16
        assert len(tournament.stages) == 1

        stage = tournament.stages[0]
        assert stage.format == TournamentFormat.SWISS
        assert len(stage.rounds) == 5

        # Check pairing logic
        for round_num, matches in stage.rounds.items():
            # In later rounds, it may not be possible to pair all teams
            # if they've already played each other
            if round_num <= 4:
                assert len(matches) == 8  # 16 teams = 8 matches per round
            else:
                # Later rounds might have fewer matches due to pairing constraints
                assert len(matches) >= 7  # At least 7 matches

    def test_round_robin_generation(self):
        """Test Round Robin tournament generation."""
        players = self.player_gen.generate_players(20)  # 5 teams

        tournament = self.tournament_gen.generate_tournament(
            players=players, format=TournamentFormat.ROUND_ROBIN, team_size=4
        )

        stage = tournament.stages[0]
        assert stage.format == TournamentFormat.ROUND_ROBIN

        # Each team plays every other team once
        expected_matches = 5 * 4 / 2  # n * (n-1) / 2
        total_matches = sum(len(matches) for matches in stage.rounds.values())
        assert total_matches == expected_matches

    def test_single_elimination_generation(self):
        """Test Single Elimination tournament generation."""
        tournament = self.tournament_gen.generate_tournament(
            players=self.players,
            format=TournamentFormat.SINGLE_ELIMINATION,
            team_size=4,
            seeded=True,
        )

        stage = tournament.stages[0]
        assert stage.format == TournamentFormat.SINGLE_ELIMINATION

        # Check bracket structure
        # 16 teams -> 8 matches R1, 4 matches R2, 2 matches R3, 1 match R4
        assert len(stage.rounds[1]) == 8
        assert len(stage.rounds[2]) == 4
        assert len(stage.rounds[3]) == 2
        assert len(stage.rounds[4]) == 1

    def test_group_stage_generation(self):
        """Test Group Stage tournament generation."""
        tournament = self.tournament_gen.generate_tournament(
            players=self.players,
            format=TournamentFormat.GROUP_STAGE,
            team_size=4,
            n_groups=4,
        )

        stage = tournament.stages[0]
        assert stage.format == TournamentFormat.GROUP_STAGE

        # Check groups are assigned
        groups_seen = set()
        for matches in stage.rounds.values():
            for match in matches:
                if match.group:
                    groups_seen.add(match.group)

        assert len(groups_seen) == 4

    def test_multi_stage_tournament(self):
        """Test multi-stage tournament generation."""
        tournament = self.tournament_gen.generate_multi_stage_tournament(
            players=self.players,
            stages_config=[
                {"format": "group_stage", "n_groups": 4, "advance_count": 8},
                {"format": "single_elimination", "seeded": True},
            ],
            team_size=4,
        )

        assert len(tournament.stages) == 2
        assert tournament.stages[0].format == TournamentFormat.GROUP_STAGE
        assert (
            tournament.stages[1].format == TournamentFormat.SINGLE_ELIMINATION
        )

    def test_team_formation(self):
        """Test team formation from players."""
        teams = self.tournament_gen._form_teams(self.players, team_size=4)

        assert len(teams) == 16
        assert all(len(team.players) == 4 for team in teams)
        assert all(isinstance(team, Team) for team in teams)

        # Check no player duplication
        all_player_ids = []
        for team in teams:
            all_player_ids.extend([p.user_id for p in team.players])
        assert len(all_player_ids) == len(set(all_player_ids))
