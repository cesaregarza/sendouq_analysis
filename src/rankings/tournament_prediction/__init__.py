"""Tournament prediction and seeding system based on player ratings.

This module implements the complete tournament prediction pipeline:
1. Team strength calculation from player ratings
2. Win probability estimation with calibrated logistic regression
3. Tournament seeding based on team ratings
4. Monte Carlo simulation for tournament outcome predictions
5. Backtesting framework for validation
"""

from rankings.tournament_prediction.backtester import TournamentBacktester
from rankings.tournament_prediction.match_predictor import MatchPredictor
from rankings.tournament_prediction.monte_carlo import MonteCarloSimulator
from rankings.tournament_prediction.seeder import TournamentSeeder
from rankings.tournament_prediction.team_strength import (
    TeamRatingConfig,
    TeamStrengthCalculator,
)

__all__ = [
    "TeamRatingConfig",
    "TeamStrengthCalculator",
    "MatchPredictor",
    "TournamentSeeder",
    "MonteCarloSimulator",
    "TournamentBacktester",
]
