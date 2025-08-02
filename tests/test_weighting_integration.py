"""Integration tests for weighting schemes with cross-validation."""

import numpy as np
import polars as pl
import pytest

from rankings import RatingEngine
from rankings.evaluation.cross_validation import evaluate_on_split
from rankings.evaluation.loss import (
    compute_tournament_loss,
    compute_weighted_log_loss,
)


class TestWeightingIntegration:
    """Integration tests for weighting schemes in real scenarios."""

    @pytest.fixture
    def sample_matches(self):
        """Create sample match data with varying confidence levels."""
        np.random.seed(42)
        n_tournaments = 20
        n_matches_per_tournament = 50

        matches = []
        for t_id in range(1, n_tournaments + 1):
            for m_id in range(n_matches_per_tournament):
                # Create matches with varying skill gaps
                if m_id < 10:
                    # Close matches (small skill gap)
                    winner_id = np.random.randint(1, 50)
                    loser_id = np.random.randint(51, 100)
                elif m_id < 30:
                    # Medium gap
                    winner_id = np.random.randint(1, 30)
                    loser_id = np.random.randint(71, 100)
                else:
                    # Large gap
                    winner_id = np.random.randint(1, 20)
                    loser_id = np.random.randint(81, 100)

                matches.append(
                    {
                        "tournament_id": t_id,
                        "match_id": f"t{t_id}_m{m_id}",
                        "winner_team_id": winner_id,
                        "loser_team_id": loser_id,
                        "is_bye": False,
                        "last_game_finished_at": 1600000000
                        + t_id * 86400
                        + m_id * 3600,  # Fake timestamps
                    }
                )

        return pl.DataFrame(matches)

    @pytest.fixture
    def sample_players(self):
        """Create sample player data."""
        players = []
        # Create players for teams 1-100
        for team_id in range(1, 101):
            for player_num in range(4):  # 4 players per team
                players.append(
                    {
                        "tournament_id": 1,  # Simplified - all from tournament 1
                        "team_id": team_id,
                        "user_id": team_id * 100 + player_num,
                    }
                )
        return pl.DataFrame(players)

    def test_weighting_affects_ratings(self, sample_matches, sample_players):
        """Test that different weightings produce different rating distributions."""
        # Train rating engines with different beta values
        results = {}

        for beta in [0.0, 1.0]:
            engine = RatingEngine(
                decay_half_life_days=30.0,
                damping_factor=0.85,
                beta=beta,
                influence_agg_method="mean",  # Use mean for simplicity
            )

            rankings = engine.rank_players(
                matches=sample_matches,
                players=sample_players,
            )

            # Store rating statistics
            ratings = rankings["player_rank"].to_numpy()
            results[beta] = {
                "mean": np.mean(ratings),
                "std": np.std(ratings),
                "max": np.max(ratings),
                "min": np.min(ratings),
            }

        # Beta should affect rating distribution
        # With beta=1.0, ratings should be more spread out
        assert results[1.0]["std"] != results[0.0]["std"]

    def test_weighting_scheme_loss_ordering(self, sample_matches):
        """Test that different weighting schemes produce expected loss ordering."""
        # Create a simple rating map where lower IDs are stronger
        rating_map = {}
        for team_id in range(1, 101):
            # Teams 1-20 are strong, 81-100 are weak
            if team_id <= 20:
                rating_map[team_id] = 0.8 + (20 - team_id) * 0.01
            elif team_id >= 81:
                rating_map[team_id] = 0.2 - (team_id - 80) * 0.01
            else:
                rating_map[team_id] = 0.5

        # Test different schemes
        losses = {}
        for scheme in ["none", "entropy", "entropy_sqrt", "entropy_squared"]:
            loss, metrics = compute_tournament_loss(
                sample_matches.head(100),  # Use subset for speed
                rating_map,
                alpha=1.0,
                scheme=scheme,
            )
            losses[scheme] = loss

        # With this setup, entropy_squared should have lowest loss
        # because it focuses on the confident predictions
        assert losses["entropy_squared"] < losses["entropy"]
        assert losses["entropy"] < losses["none"]

    def test_beta_improvement_varies_by_scheme(
        self, sample_matches, sample_players
    ):
        """Test that beta improvement differs across weighting schemes."""
        # We'll use a smaller dataset for speed
        small_matches = sample_matches.head(200)

        improvements = {}

        for scheme in ["entropy", "entropy_sqrt", "entropy_squared"]:
            losses = {}

            for beta in [0.0, 1.0]:
                # Train engine
                engine = RatingEngine(
                    decay_half_life_days=30.0,
                    damping_factor=0.85,
                    beta=beta,
                )

                # Get rankings
                rankings = engine.rank_players(
                    matches=sample_matches,
                    players=sample_players,
                )

                # Create rating map
                rating_map = dict(zip(rankings["id"], rankings["player_rank"]))

                # Compute loss
                loss, _ = compute_tournament_loss(
                    small_matches,
                    rating_map,
                    alpha=1.0,
                    scheme=scheme,
                    winner_id_col="winner_team_id",
                    loser_id_col="loser_team_id",
                )
                losses[beta] = loss

            # Calculate improvement
            if losses[0.0] > 0:
                improvements[scheme] = (
                    (losses[0.0] - losses[1.0]) / losses[0.0] * 100
                )
            else:
                improvements[scheme] = 0.0

        # Different schemes should show different improvements
        # Can't guarantee exact ordering due to randomness, but they should differ
        assert len(set(improvements.values())) > 1

    def test_extreme_probabilities_handling(self):
        """Test that weighting schemes handle extreme probabilities correctly."""
        # Test with probabilities very close to 0 and 1
        extreme_probs = np.array([0.001, 0.01, 0.99, 0.999])

        # None of these should cause errors or return inf/nan
        for scheme in ["entropy", "entropy_sqrt", "entropy_squared"]:
            loss = compute_weighted_log_loss(extreme_probs, scheme=scheme)
            assert np.isfinite(loss)
            assert loss > 0

    def test_empty_matches_handling(self):
        """Test that weighting schemes handle empty data gracefully."""
        empty_matches = pl.DataFrame(
            {
                "winner_team_id": [],
                "loser_team_id": [],
            }
        )

        rating_map = {}

        for scheme in ["none", "entropy", "entropy_sqrt", "entropy_squared"]:
            loss, metrics = compute_tournament_loss(
                empty_matches,
                rating_map,
                scheme=scheme,
            )
            assert loss == np.inf
            assert metrics["n_matches"] == 0


if __name__ == "__main__":
    # Run a quick test
    test = TestWeightingIntegration()

    # Create fixtures
    matches = test.sample_matches()
    players = test.sample_players()

    # Run tests
    test.test_weighting_affects_ratings(matches, players)
    test.test_extreme_probabilities_handling()
    print("Manual integration tests passed!")
