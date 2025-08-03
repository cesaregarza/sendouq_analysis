"""
Tournament circuit generation module for synthetic tournament data.

This module generates a series of tournaments (a "circuit") with various
entry criteria and formats, using a fixed population of players.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from synthetic_data.match_simulator import MatchSimulator
from synthetic_data.player_generator import PlayerGenerator, SyntheticPlayer
from synthetic_data.tournament_generator import (
    Tournament,
    TournamentFormat,
    TournamentGenerator,
)


class TournamentType(Enum):
    """Types of tournaments based on entry criteria."""

    OPEN = "open"  # No restrictions, random selection
    SKILL_CAPPED = "skill_capped"  # Upper skill limit
    INVITATIONAL = "invitational"  # Lower skill limit (top players only)
    MIXED = "mixed"  # Combination of criteria


@dataclass
class TournamentConfig:
    """Configuration for a single tournament in the circuit."""

    name: str
    tournament_type: TournamentType
    format: TournamentFormat
    team_size: int = 4
    n_teams: int = 16
    skill_cap: Optional[float] = None  # Max skill for skill-capped
    skill_floor: Optional[float] = None  # Min skill for invitational
    selection_bias: float = 0.0  # Bias towards higher skilled players (0-1)
    start_offset_days: float = 0.0  # Days after circuit start

    # Format-specific parameters
    swiss_rounds: Optional[int] = None
    double_round_robin: bool = False
    seeded_bracket: bool = True


@dataclass
class CircuitResults:
    """Results from a tournament circuit simulation."""

    tournaments: List[Tournament]
    player_participation: Dict[int, List[int]]  # player_id -> [tournament_ids]
    player_wins: Dict[int, int]  # player_id -> total wins
    player_matches: Dict[int, int]  # player_id -> total matches


class TournamentCircuit:
    """Generates and manages a circuit of tournaments."""

    def __init__(
        self,
        seed: Optional[int] = None,
        player_pool_size: int = 200,
        skill_distribution: str = "normal",
        skill_params: Optional[Dict] = None,
    ):
        """
        Initialize the tournament circuit.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility
        player_pool_size : int
            Total number of players in the circuit
        skill_distribution : str
            Distribution type for player skills
        skill_params : dict, optional
            Parameters for skill distribution
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Initialize generators
        self.player_gen = PlayerGenerator(seed=seed)
        self.tournament_gen = TournamentGenerator(seed=seed)
        self.match_sim = MatchSimulator(seed=seed)

        # Generate player pool
        self.player_pool = self._generate_player_pool(
            player_pool_size, skill_distribution, skill_params
        )

        # Sort players by skill for easy selection
        self.players_by_skill = sorted(
            self.player_pool, key=lambda p: p.true_skill, reverse=True
        )

        # Track circuit statistics
        self.circuit_results = CircuitResults(
            tournaments=[],
            player_participation={p.user_id: [] for p in self.player_pool},
            player_wins={p.user_id: 0 for p in self.player_pool},
            player_matches={p.user_id: 0 for p in self.player_pool},
        )

    def _generate_player_pool(
        self,
        size: int,
        distribution: str,
        params: Optional[Dict],
    ) -> List[SyntheticPlayer]:
        """Generate the fixed player pool for the circuit."""
        if distribution == "realistic":
            # Create a realistic distribution with elite, competitive, and casual
            pool_dict = self.player_gen.create_player_pool_with_categories(
                n_elite=int(size * 0.1),  # 10% elite
                n_competitive=int(size * 0.35),  # 35% competitive
                n_casual=int(size * 0.55),  # 55% casual
            )
            return (
                pool_dict["elite"]
                + pool_dict["competitive"]
                + pool_dict["casual"]
            )
        else:
            return self.player_gen.generate_players(size, distribution, params)

    def generate_circuit(
        self,
        tournament_configs: List[TournamentConfig],
        start_date: Optional[datetime] = None,
    ) -> CircuitResults:
        """
        Generate a complete tournament circuit.

        Parameters
        ----------
        tournament_configs : List[TournamentConfig]
            Configurations for each tournament
        start_date : datetime, optional
            Start date for the circuit

        Returns
        -------
        CircuitResults
            Results from the circuit simulation
        """
        if start_date is None:
            start_date = datetime.now()

        # Sort configs by start time
        sorted_configs = sorted(
            tournament_configs, key=lambda c: c.start_offset_days
        )

        for config in sorted_configs:
            # Calculate tournament start date
            tourney_start = start_date + timedelta(
                days=config.start_offset_days
            )

            # Select participants based on tournament type
            participants = self._select_participants(config)

            # Generate tournament
            tournament = self._generate_tournament(
                participants, config, tourney_start
            )

            # Simulate all matches
            self._simulate_tournament_matches(tournament)

            # Update circuit statistics
            self._update_circuit_stats(tournament)

            self.circuit_results.tournaments.append(tournament)

        return self.circuit_results

    def _select_participants(
        self, config: TournamentConfig
    ) -> List[SyntheticPlayer]:
        """Select players for a tournament based on its type."""
        eligible_players = self.player_pool.copy()

        # Apply skill restrictions
        if config.tournament_type == TournamentType.SKILL_CAPPED:
            if config.skill_cap is not None:
                eligible_players = [
                    p
                    for p in eligible_players
                    if p.true_skill <= config.skill_cap
                ]

        elif config.tournament_type == TournamentType.INVITATIONAL:
            if config.skill_floor is not None:
                eligible_players = [
                    p
                    for p in eligible_players
                    if p.true_skill >= config.skill_floor
                ]

        elif config.tournament_type == TournamentType.MIXED:
            # Apply both caps if specified
            if config.skill_cap is not None:
                eligible_players = [
                    p
                    for p in eligible_players
                    if p.true_skill <= config.skill_cap
                ]
            if config.skill_floor is not None:
                eligible_players = [
                    p
                    for p in eligible_players
                    if p.true_skill >= config.skill_floor
                ]

        # Calculate number of players needed
        n_players_needed = config.n_teams * config.team_size

        if len(eligible_players) < n_players_needed:
            # Not enough eligible players, take all
            selected = eligible_players
        else:
            # Select players with optional bias
            if config.selection_bias > 0:
                # Bias selection towards higher skilled players
                selected = self._biased_selection(
                    eligible_players, n_players_needed, config.selection_bias
                )
            else:
                # Random selection
                selected = self.rng.choice(
                    eligible_players, n_players_needed, replace=False
                ).tolist()

        return selected

    def _biased_selection(
        self,
        players: List[SyntheticPlayer],
        n_select: int,
        bias: float,
    ) -> List[SyntheticPlayer]:
        """Select players with bias towards higher skill."""
        # Sort by skill
        sorted_players = sorted(
            players, key=lambda p: p.true_skill, reverse=True
        )

        # Create selection weights
        n_players = len(sorted_players)
        positions = np.arange(n_players)

        # Exponential decay weights based on position
        weights = np.exp(-bias * positions / n_players)
        weights = weights / weights.sum()

        # Select without replacement
        indices = self.rng.choice(n_players, n_select, replace=False, p=weights)

        return [sorted_players[i] for i in indices]

    def _generate_tournament(
        self,
        participants: List[SyntheticPlayer],
        config: TournamentConfig,
        start_date: datetime,
    ) -> Tournament:
        """Generate a single tournament."""
        # Build format-specific kwargs
        kwargs = {}

        if config.format == TournamentFormat.SWISS:
            if config.swiss_rounds is not None:
                kwargs["n_rounds"] = config.swiss_rounds

        elif config.format == TournamentFormat.ROUND_ROBIN:
            kwargs["double"] = config.double_round_robin

        elif config.format in [
            TournamentFormat.SINGLE_ELIMINATION,
            TournamentFormat.DOUBLE_ELIMINATION,
        ]:
            kwargs["seeded"] = config.seeded_bracket

        # Generate tournament
        tournament = self.tournament_gen.generate_tournament(
            participants,
            config.format,
            team_size=config.team_size,
            name=config.name,
            start_date=start_date,
            **kwargs,
        )

        return tournament

    def _simulate_tournament_matches(self, tournament: Tournament):
        """Simulate all matches in a tournament."""
        for stage in tournament.stages:
            for round_num in sorted(stage.rounds.keys()):
                for match in stage.rounds[round_num]:
                    self.match_sim.simulate_match(match)

    def _update_circuit_stats(self, tournament: Tournament):
        """Update circuit statistics after a tournament."""
        # Track player participation
        for team in tournament.all_teams:
            for player in team.players:
                self.circuit_results.player_participation[
                    player.user_id
                ].append(tournament.tournament_id)

        # Track match results
        for stage in tournament.stages:
            for round_matches in stage.rounds.values():
                for match in round_matches:
                    # Update match counts
                    for player in match.team_a.players:
                        self.circuit_results.player_matches[player.user_id] += 1
                    for player in match.team_b.players:
                        self.circuit_results.player_matches[player.user_id] += 1

                    # Update win counts
                    if match.winner:
                        for player in match.winner.players:
                            self.circuit_results.player_wins[
                                player.user_id
                            ] += 1

    def generate_standard_circuit(
        self,
        n_tournaments: int = 20,
        start_date: Optional[datetime] = None,
        tournament_interval_days: float = 7.0,
    ) -> CircuitResults:
        """
        Generate a standard tournament circuit with mixed tournament types.

        Parameters
        ----------
        n_tournaments : int
            Number of tournaments to generate
        start_date : datetime, optional
            Circuit start date
        tournament_interval_days : float
            Days between tournament starts

        Returns
        -------
        CircuitResults
            Circuit simulation results
        """
        configs = []

        for i in range(n_tournaments):
            # Vary tournament types
            if i % 5 == 0:
                # Elite invitational every 5th tournament
                config = TournamentConfig(
                    name=f"Elite_Invitational_{i+1}",
                    tournament_type=TournamentType.INVITATIONAL,
                    format=TournamentFormat.DOUBLE_ELIMINATION,
                    n_teams=8,
                    skill_floor=1.5,  # Top players only
                    seeded_bracket=True,
                    start_offset_days=i * tournament_interval_days,
                )
            elif i % 3 == 0:
                # Skill-capped tournament every 3rd
                config = TournamentConfig(
                    name=f"Amateur_Cup_{i+1}",
                    tournament_type=TournamentType.SKILL_CAPPED,
                    format=TournamentFormat.SWISS,
                    n_teams=32,
                    skill_cap=0.5,  # Lower skilled players
                    swiss_rounds=5,
                    start_offset_days=i * tournament_interval_days,
                )
            else:
                # Open tournaments
                formats = [
                    TournamentFormat.SWISS,
                    TournamentFormat.SINGLE_ELIMINATION,
                    TournamentFormat.ROUND_ROBIN,
                ]
                config = TournamentConfig(
                    name=f"Open_Tournament_{i+1}",
                    tournament_type=TournamentType.OPEN,
                    format=self.rng.choice(formats),
                    n_teams=self.rng.choice([16, 24, 32]),
                    selection_bias=0.2,  # Slight bias towards skilled players
                    start_offset_days=i * tournament_interval_days,
                )

            configs.append(config)

        return self.generate_circuit(configs, start_date)
