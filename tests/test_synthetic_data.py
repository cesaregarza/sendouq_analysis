"""
Comprehensive tests for synthetic tournament data generation.

Tests all components of the synthetic data generation system including
player generation, tournament structures, match simulation, and serialization.
"""

import json
from datetime import datetime

import numpy as np
import pytest

from rankings.core import parse_tournaments_data
from synthetic_data import (
    DataSerializer,
    DataValidator,
    MatchSimulator,
    PlayerGenerator,
    SyntheticPlayer,
    TournamentGenerator,
)
from synthetic_data.tournament_generator import Team, TournamentFormat


class TestPlayerGenerator:
    """Test player generation functionality."""
    
    def test_generate_players_basic(self):
        """Test basic player generation."""
        gen = PlayerGenerator(seed=42)
        players = gen.generate_players(n_players=10)
        
        assert len(players) == 10
        assert all(isinstance(p, SyntheticPlayer) for p in players)
        assert all(p.user_id > 0 for p in players)
        assert all(p.username for p in players)
        
    def test_skill_distributions(self):
        """Test different skill distributions."""
        gen = PlayerGenerator(seed=42)
        
        # Normal distribution
        normal_players = gen.generate_players(
            n_players=100,
            skill_distribution="normal",
            skill_params={"mean": 0.0, "std": 1.0}
        )
        skills = [p.true_skill for p in normal_players]
        assert -3 < np.mean(skills) < 3
        assert 0.5 < np.std(skills) < 1.5
        
        # Uniform distribution
        uniform_players = gen.generate_players(
            n_players=100,
            skill_distribution="uniform",
            skill_params={"low": -2.0, "high": 2.0}
        )
        skills = [p.true_skill for p in uniform_players]
        assert min(skills) >= -2.0
        assert max(skills) <= 2.0
        
        # Bimodal distribution
        bimodal_players = gen.generate_players(
            n_players=100,
            skill_distribution="bimodal"
        )
        assert len(bimodal_players) == 100
        
    def test_elite_players(self):
        """Test elite player generation."""
        gen = PlayerGenerator(seed=42)
        elite = gen.generate_elite_players(n_players=20, base_skill=2.0)
        
        assert len(elite) == 20
        assert all(p.true_skill >= 2.0 for p in elite)
        assert all(0.05 <= p.skill_variance <= 0.15 for p in elite)
        
    def test_player_pool_categories(self):
        """Test creating categorized player pools."""
        gen = PlayerGenerator(seed=42)
        pool = gen.create_player_pool_with_categories(
            n_elite=10,
            n_competitive=20,
            n_casual=30
        )
        
        assert len(pool["elite"]) == 10
        assert len(pool["competitive"]) == 20
        assert len(pool["casual"]) == 30
        
        # Check skill ordering
        elite_avg = np.mean([p.true_skill for p in pool["elite"]])
        comp_avg = np.mean([p.true_skill for p in pool["competitive"]])
        casual_avg = np.mean([p.true_skill for p in pool["casual"]])
        
        assert elite_avg > comp_avg > casual_avg
        
    def test_player_performance(self):
        """Test player performance variation."""
        player = SyntheticPlayer(
            user_id=1,
            username="test",
            true_skill=1.0,
            skill_variance=0.2
        )
        
        rng = np.random.default_rng(42)
        performances = [player.get_performance(rng) for _ in range(100)]
        
        assert 0.5 < np.mean(performances) < 1.5
        assert 0.1 < np.std(performances) < 0.3


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
            n_rounds=5
        )
        
        assert tournament.tournament_id > 0
        assert len(tournament.all_teams) == 16
        assert len(tournament.stages) == 1
        
        stage = tournament.stages[0]
        assert stage.format == TournamentFormat.SWISS
        assert len(stage.rounds) == 5
        
        # Check pairing logic
        for round_num, matches in stage.rounds.items():
            assert len(matches) == 8  # 16 teams = 8 matches per round
            
    def test_round_robin_generation(self):
        """Test Round Robin tournament generation."""
        players = self.player_gen.generate_players(20)  # 5 teams
        
        tournament = self.tournament_gen.generate_tournament(
            players=players,
            format=TournamentFormat.ROUND_ROBIN,
            team_size=4
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
            seeded=True
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
            n_groups=4
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
                {"format": "single_elimination", "seeded": True}
            ],
            team_size=4
        )
        
        assert len(tournament.stages) == 2
        assert tournament.stages[0].format == TournamentFormat.GROUP_STAGE
        assert tournament.stages[1].format == TournamentFormat.SINGLE_ELIMINATION
        
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
        from synthetic_data.tournament_generator import Match
        
        match = Match(
            match_id=1,
            team_a=self.strong_team,
            team_b=self.weak_team,
            round_number=1,
            stage="Test"
        )
        
        result = self.sim.simulate_match(match)
        
        assert result.winner is not None
        assert result.score_a is not None
        assert result.score_b is not None
        assert result.score_a + result.score_b >= 4  # Best of 7
        
    def test_match_probability_calculation(self):
        """Test match probability calculations."""
        prob_strong_wins = self.sim.calculate_match_probability(
            self.strong_team,
            self.weak_team
        )
        
        # Strong team should have high win probability
        assert 0.8 < prob_strong_wins < 1.0
        
        # Equal teams should have ~50% probability
        equal_team1 = Team(3, "Equal 1", self.strong_team.players[:4])
        equal_team2 = Team(4, "Equal 2", self.strong_team.players[:4])
        prob_equal = self.sim.calculate_match_probability(equal_team1, equal_team2)
        assert 0.45 < prob_equal < 0.55
        
    def test_upset_generation(self):
        """Test upset generation."""
        from synthetic_data.tournament_generator import Match
        
        # Run many simulations to test upset rate
        upset_count = 0
        n_sims = 100
        
        for i in range(n_sims):
            match = Match(
                match_id=i,
                team_a=self.strong_team,
                team_b=self.weak_team,
                round_number=1,
                stage="Test"
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
            from synthetic_data.tournament_generator import Match
            
            match = Match(
                match_id=i,
                team_a=self.strong_team,
                team_b=self.weak_team,
                round_number=1,
                stage="Test"
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
            n_rounds=3
        )
        
        data = self.serializer.serialize_tournament(
            tournament,
            simulate_matches=True,
            match_simulator=self.match_sim
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
            team_size=4
        )
        
        data = self.serializer.serialize_tournament(
            tournament,
            simulate_matches=True,
            match_simulator=self.match_sim
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
            players=players,
            format=TournamentFormat.ROUND_ROBIN,
            team_size=4
        )
        
        data_list = [self.serializer.serialize_tournament(
            tournament,
            simulate_matches=True,
            match_simulator=self.match_sim
        )]
        
        # Should parse without errors
        tables = parse_tournaments_data(data_list)
        
        assert tables["matches"] is not None
        assert tables["teams"] is not None
        assert tables["players"] is not None
        assert len(tables["matches"]) > 0
        assert len(tables["teams"]) == 8
        assert len(tables["players"]) == 32


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
            players=players,
            format=TournamentFormat.SWISS,
            team_size=4
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
                            "opponent2": {"id": 999}  # Invalid team reference
                        }
                    ]
                },
                "ctx": {
                    "id": 1,
                    "teams": [{"id": 1, "members": []}]
                }
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
            team_size=4
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
            team_size=4
        )
        
        self.validator.validate_tournament(tournament)
        report = self.validator.get_validation_report()
        
        assert "WARNING" in report or "All validations passed" in report


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from generation to ranking."""
        # Generate tournament series
        player_gen = PlayerGenerator(seed=42)
        tournament_gen = TournamentGenerator(seed=42)
        match_sim = MatchSimulator(seed=42, alpha=1.0)
        serializer = DataSerializer()
        
        # Create player pool
        players = player_gen.create_player_pool_with_categories(
            n_elite=20,
            n_competitive=40,
            n_casual=60
        )
        
        all_players = []
        for category_players in players.values():
            all_players.extend(category_players)
            
        # Generate multiple tournaments
        tournaments = []
        for i in range(3):
            # Sample players
            import random
            rng = random.Random(42 + i)
            participants = rng.sample(all_players, k=32)
            
            tournament = tournament_gen.generate_tournament(
                players=participants,
                format=TournamentFormat.SWISS,
                team_size=4,
                name=f"Test Tournament {i+1}"
            )
            
            data = serializer.serialize_tournament(
                tournament,
                simulate_matches=True,
                match_simulator=match_sim
            )
            
            tournaments.append(data)
            
        # Parse tournaments
        tables = parse_tournaments_data(tournaments)
        
        # Run ranking engine
        from rankings import RatingEngine
        
        engine = RatingEngine()
        rankings = engine.rank_players(tables["matches"], tables["players"])
        
        # Verify we got rankings
        assert len(rankings) > 0
        assert "rating" in rankings.columns
        assert "user_id" in rankings.columns
        
        # Top players should generally be from elite category
        top_10 = rankings.head(10)
        top_player_ids = set(top_10["user_id"].to_list())
        elite_ids = {p.user_id for p in players["elite"]}
        
        # At least some elite players should be in top 10
        assert len(top_player_ids & elite_ids) > 0
        
    def test_cross_validation_workflow(self):
        """Test synthetic data with cross-validation."""
        from rankings.evaluation import create_time_based_folds, cross_validate_ratings
        
        # Generate time-series tournaments
        player_gen = PlayerGenerator(seed=99)
        tournament_gen = TournamentGenerator(seed=99) 
        serializer = DataSerializer()
        
        players = player_gen.generate_players(64)
        
        tournaments = []
        base_date = datetime(2024, 1, 1)
        
        for i in range(5):
            tournament = tournament_gen.generate_tournament(
                players=players,
                format=TournamentFormat.SWISS,
                team_size=4,
                start_date=base_date.replace(day=1 + i * 7)
            )
            
            data = serializer.serialize_tournament(tournament)
            tournaments.append(data)
            
        # Parse and create folds
        tables = parse_tournaments_data(tournaments)
        folds = create_time_based_folds(
            tables["matches"],
            tables["players"],
            n_folds=3
        )
        
        # Run cross-validation
        from rankings import RatingEngine
        
        results = cross_validate_ratings(
            folds=folds,
            rating_engine=RatingEngine()
        )
        
        # Should have results for each fold
        assert len(results) == 3
        assert all("train_loss" in r for r in results)
        assert all("val_loss" in r for r in results)