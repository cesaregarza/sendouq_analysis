"""Synthetic tournament data generation for ranking system evaluation.

This package provides tools for generating synthetic tournament data with known
ground truth for evaluating ranking algorithms.

Submodules:
- core: Core data models and generation (players, tournaments, matches)
- circuits: Tournament circuits and scheduling
- evaluation: Evaluation tools for ranking algorithms
- io: Data serialization and validation
- utils: Utility functions
"""

# Re-export main classes from submodules for backward compatibility
from synthetic_data.circuits import (
    TournamentCircuit,
    TournamentConfig,
    TournamentScheduleGenerator,
    TournamentType,
)
from synthetic_data.core import (
    Match,
    MatchSimulator,
    PlayerGenerator,
    SyntheticPlayer,
    Team,
    Tournament,
    TournamentFormat,
    TournamentGenerator,
)
from synthetic_data.evaluation import PageRankEvaluator
from synthetic_data.io import DataSerializer, DataValidator

__all__ = [
    # Core - Player generation
    "PlayerGenerator",
    "SyntheticPlayer",
    # Core - Tournament generation
    "TournamentGenerator",
    "TournamentFormat",
    "Tournament",
    "Team",
    "Match",
    # Core - Match simulation
    "MatchSimulator",
    # Circuits - Tournament circuits and scheduling
    "TournamentCircuit",
    "TournamentConfig",
    "TournamentType",
    "TournamentScheduleGenerator",
    # IO - Data handling
    "DataSerializer",
    "DataValidator",
    # Evaluation
    "PageRankEvaluator",
]
