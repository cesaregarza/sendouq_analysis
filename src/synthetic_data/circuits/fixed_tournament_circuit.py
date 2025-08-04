"""
Fixed tournament circuit that reuses teams across tournaments.

This fixes the critical issue where every tournament creates new teams,
preventing beta from having any effect.
"""

import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

import numpy as np
import polars as pl

from synthetic_data.circuits.tournament_circuit import (
    CircuitResults,
    TournamentCircuit,
    TournamentConfig,
)
from synthetic_data.configs.realistic_tournament_config import (
    RealisticCircuitConfig,
    create_realistic_skill_distribution,
)
from synthetic_data.core.player_generator import (
    PlayerGenerator,
    SyntheticPlayer,
)
from synthetic_data.core.tournament_generator import (
    Team,
    Tournament,
    TournamentStage,
)


class FixedTournamentCircuit(TournamentCircuit):
    """
    Tournament circuit that properly reuses teams across tournaments.

    Key fixes:
    1. Creates a stable pool of teams at initialization
    2. Teams participate in multiple tournaments
    3. Team composition remains consistent
    4. Creates cross-tournament connections needed for beta
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        config: Optional[RealisticCircuitConfig] = None,
        team_overlap_rate: float = 0.3,  # Fraction of teams that play in multiple tournaments
    ):
        """
        Initialize fixed tournament circuit.

        Parameters
        ----------
        seed : int, optional
            Random seed
        config : RealisticCircuitConfig, optional
            Circuit configuration
        team_overlap_rate : float
            Fraction of teams that should play in multiple tournaments
        """
        if config is None:
            config = RealisticCircuitConfig()

        self.config = config
        self.team_overlap_rate = team_overlap_rate

        # Initialize base circuit
        super().__init__(
            seed=seed,
            player_pool_size=config.player_pool_size,
            skill_distribution=config.skill_distribution,
            skill_params=config.skill_params,
        )

        # Create stable team pool
        self._create_stable_teams()

    def _create_stable_teams(self):
        """Create a stable pool of teams that will be reused across tournaments."""
        # Generate realistic skill distribution for players
        skills = create_realistic_skill_distribution(
            self.config.player_pool_size,
            self.config.skill_distribution,
            self.seed,
        )

        # Create players with these skills
        self.player_pool = []
        for i, skill in enumerate(skills):
            player = self.player_gen.generate_players(
                1,
                skill_distribution="uniform",
                skill_params={"low": skill, "high": skill},
            )[0]
            player.user_id = i + 1
            player.true_skill = skill
            self.player_pool.append(player)

        # Sort by skill for easier team formation
        self.player_pool.sort(key=lambda p: p.true_skill, reverse=True)

        # Form stable teams
        # Group players by skill tier to create balanced teams
        self.stable_teams = []
        self.team_skill_tiers = {}

        team_size = 4
        n_teams = len(self.player_pool) // team_size

        # Create teams with players of similar skill
        skill_tiers = 5  # Number of skill tiers
        players_per_tier = len(self.player_pool) // skill_tiers

        team_id = 1
        for tier in range(skill_tiers):
            tier_start = tier * players_per_tier
            tier_end = min((tier + 1) * players_per_tier, len(self.player_pool))
            tier_players = self.player_pool[tier_start:tier_end]

            # Shuffle within tier to create varied teams
            self.rng.shuffle(tier_players)

            # Form teams from this tier
            for i in range(0, len(tier_players), team_size):
                if i + team_size <= len(tier_players):
                    team_players = tier_players[i : i + team_size]
                    team = Team(
                        team_id=team_id,
                        name=f"Team_{team_id}",
                        players=team_players,
                    )
                    self.stable_teams.append(team)
                    self.team_skill_tiers[team_id] = tier
                    team_id += 1

        print(
            f"Created {len(self.stable_teams)} stable teams across {skill_tiers} skill tiers"
        )

        # Track which tournaments each team has played in
        self.team_tournament_history: Dict[int, Set[int]] = {
            team.team_id: set() for team in self.stable_teams
        }

    def _select_teams_for_tournament(
        self,
        config: TournamentConfig,
        tournament_id: int,
    ) -> List[Team]:
        """
        Select teams for a tournament, ensuring some teams play in multiple tournaments.

        Parameters
        ----------
        config : TournamentConfig
            Tournament configuration
        tournament_id : int
            Tournament ID

        Returns
        -------
        List[Team]
            Selected teams for the tournament
        """
        eligible_teams = self.stable_teams.copy()

        # Apply skill restrictions based on tournament type
        if config.skill_floor is not None or config.skill_cap is not None:
            # Filter by team average skill
            filtered_teams = []
            for team in eligible_teams:
                avg_skill = team.avg_skill

                if (
                    config.skill_floor is not None
                    and avg_skill < config.skill_floor
                ):
                    continue
                if (
                    config.skill_cap is not None
                    and avg_skill > config.skill_cap
                ):
                    continue

                filtered_teams.append(team)

            eligible_teams = filtered_teams

        # Prioritize teams that haven't played recently
        # and ensure some teams play in multiple tournaments
        team_scores = []
        for team in eligible_teams:
            n_tournaments_played = len(
                self.team_tournament_history[team.team_id]
            )

            # Score based on:
            # 1. Number of tournaments played (fewer is better initially)
            # 2. But some teams should play more (overlap)
            # 3. Skill bias from config

            if n_tournaments_played == 0:
                # New teams get high priority
                base_score = 10.0
            elif (
                n_tournaments_played < 3
                and self.rng.random() < self.team_overlap_rate
            ):
                # Some teams should play multiple tournaments
                base_score = 5.0
            else:
                # Reduce priority for teams that have played a lot
                base_score = 1.0 / (n_tournaments_played + 1)

            # Apply skill bias
            if config.selection_bias > 0:
                skill_factor = (1 + config.selection_bias) ** (
                    team.avg_skill / 2.0
                )
                base_score *= skill_factor

            team_scores.append((team, base_score))

        # Sort by score and select top N teams
        team_scores.sort(key=lambda x: x[1], reverse=True)
        n_teams_needed = min(config.n_teams, len(team_scores))

        selected_teams = [team for team, _ in team_scores[:n_teams_needed]]

        # Update tournament history
        for team in selected_teams:
            self.team_tournament_history[team.team_id].add(tournament_id)

        return selected_teams

    def generate_fixed_circuit(
        self,
        n_tournaments: Optional[int] = None,
        start_date: Optional[datetime] = None,
    ) -> CircuitResults:
        """
        Generate a tournament circuit with proper team reuse.

        Parameters
        ----------
        n_tournaments : int, optional
            Number of tournaments
        start_date : datetime, optional
            Start date

        Returns
        -------
        CircuitResults
            Circuit results
        """
        if n_tournaments is None:
            n_tournaments = self.config.n_tournaments

        if start_date is None:
            start_date = self.config.start_date or datetime.now()

        # Generate tournament configurations
        player_skills = np.array([p.true_skill for p in self.player_pool])
        configs = self.config.generate_tournament_configs(player_skills)[
            :n_tournaments
        ]

        # Initialize results
        self.circuit_results = CircuitResults(
            tournaments=[],
            player_participation={},
            player_wins={},
            player_matches={},
        )

        # Generate each tournament
        for i, config in enumerate(configs):
            tourney_start = start_date + timedelta(
                days=config.start_offset_days
            )

            # Select teams for this tournament
            selected_teams = self._select_teams_for_tournament(config, i + 1)

            # Create tournament with selected teams
            tournament = Tournament(
                tournament_id=i + 1,
                name=config.name,
                start_date=tourney_start,
                all_teams=selected_teams,
            )

            # Generate tournament structure
            stage = self._generate_tournament_stage(
                selected_teams,
                config.format,
                config,
            )
            tournament.stages = [stage]

            # Simulate matches
            self._simulate_tournament_matches(tournament)

            # Add to results
            self.circuit_results.tournaments.append(tournament)

        # Report team participation
        print(f"\nTeam participation across {n_tournaments} tournaments:")
        participation_counts = {}
        for team_id, tournaments in self.team_tournament_history.items():
            count = len(tournaments)
            if count > 0:
                if count not in participation_counts:
                    participation_counts[count] = 0
                participation_counts[count] += 1

        for count in sorted(participation_counts.keys()):
            print(
                f"  {participation_counts[count]} teams played in {count} tournament(s)"
            )

        return self.circuit_results

    def _generate_tournament_stage(self, teams, format, config):
        """Generate a tournament stage with the given format."""
        # Use the parent class methods for actual tournament generation
        # This is simplified - in reality you'd implement all formats
        if format.value == "single_elimination":
            return self.tournament_gen._generate_single_elimination_stage(
                teams, seeded=config.seeded_bracket
            )
        elif format.value == "double_elimination":
            return self.tournament_gen._generate_double_elimination_stage(
                teams, seeded=config.seeded_bracket
            )
        elif format.value == "swiss":
            return self.tournament_gen._generate_swiss_stage(
                teams, n_rounds=config.swiss_rounds
            )
        else:
            # Default to single elimination
            return self.tournament_gen._generate_single_elimination_stage(
                teams, seeded=True
            )

    def _simulate_tournament_matches(self, tournament):
        """Simulate all matches in a tournament."""
        for stage in tournament.stages:
            for round_matches in stage.rounds.values():
                for match in round_matches:
                    self.match_sim.simulate_match(match)
