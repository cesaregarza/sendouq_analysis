"""Tests for player generation functionality."""

import numpy as np
import pytest

from synthetic_data import PlayerGenerator, SyntheticPlayer


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
            skill_params={"mean": 0.0, "std": 1.0},
        )
        skills = [p.true_skill for p in normal_players]
        assert -3 < np.mean(skills) < 3
        assert 0.5 < np.std(skills) < 1.5

        # Uniform distribution
        uniform_players = gen.generate_players(
            n_players=100,
            skill_distribution="uniform",
            skill_params={"low": -2.0, "high": 2.0},
        )
        skills = [p.true_skill for p in uniform_players]
        assert min(skills) >= -2.0
        assert max(skills) <= 2.0

        # Bimodal distribution
        bimodal_players = gen.generate_players(
            n_players=100, skill_distribution="bimodal"
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
            n_elite=10, n_competitive=20, n_casual=30
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
            user_id=1, username="test", true_skill=1.0, skill_variance=0.2
        )

        rng = np.random.default_rng(42)
        performances = [player.get_performance(rng) for _ in range(100)]

        assert 0.5 < np.mean(performances) < 1.5
        assert 0.1 < np.std(performances) < 0.3
