"""
Player-centric tournament circuit generator.

This module generates tournaments where:
1. Players are the persistent entities (not teams)
2. Players form different teams in different tournaments
3. Players participate in multiple tournaments
4. Teams are tournament-specific (like real data)

This matches the real tournament structure where beta actually works.
"""

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from synthetic_data.circuits.tournament_circuit import (
    CircuitResults,
    TournamentType,
)
from synthetic_data.core.match_simulator import MatchSimulator
from synthetic_data.core.player_generator import (
    PlayerGenerator,
    SyntheticPlayer,
)
from synthetic_data.core.tournament_generator import (
    Team,
    Tournament,
    TournamentFormat,
    TournamentGenerator,
)


@dataclass
class PlayerCircuitConfig:
    """Configuration for player-centric tournament circuit."""

    # Player pool
    n_players: int = 2000
    skill_distribution: str = "lognormal"
    skill_params: Optional[dict] = None

    # Tournament configuration
    n_tournaments: int = 100
    team_size: int = 4
    teams_per_tournament_range: Tuple[int, int] = (
        16,
        64,
    )  # Min/max teams per tournament
    tournament_interval_days: float = 3.0

    # Player participation
    player_activity_distribution: str = (
        "power_law"  # How many tournaments each player joins
    )
    min_tournaments_per_player: int = 1
    max_tournaments_per_player: int = 30
    avg_tournaments_per_player: float = 5.0

    # Team formation
    team_formation_strategy: str = (
        "mixed"  # "random", "skill_based", "friend_groups", "mixed"
    )
    friend_group_probability: float = (
        0.3  # Chance players team with friends again
    )
    skill_similarity_weight: float = (
        0.4  # Preference for similar skill teammates
    )

    # Tournament types
    elite_percentile: float = 0.85  # Top 15% for elite tournaments
    amateur_percentile: float = 0.35  # Bottom 35% for amateur tournaments


@dataclass
class PlayerTournamentStats:
    """Track player statistics across tournaments."""

    player_id: int
    tournaments_played: Set[int] = field(default_factory=set)
    teams_formed: Dict[int, int] = field(
        default_factory=dict
    )  # tournament_id -> team_id
    teammates_history: Set[int] = field(
        default_factory=set
    )  # All players teamed with
    wins: int = 0
    matches: int = 0


class PlayerTournamentCircuit:
    """
    Generates tournament circuits with player-centric participation.

    Key features:
    1. Players participate in multiple tournaments
    2. Teams are formed fresh for each tournament
    3. Players may team with different people each time
    4. Realistic participation patterns (some players very active, most play a few)
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        config: Optional[PlayerCircuitConfig] = None,
    ):
        """Initialize player-centric tournament circuit."""
        if config is None:
            config = PlayerCircuitConfig()

        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Initialize generators
        self.player_gen = PlayerGenerator(seed=seed)
        self.tournament_gen = TournamentGenerator(seed=seed)
        self.match_sim = MatchSimulator(seed=seed)

        # Generate player pool
        self._generate_player_pool()

        # Initialize player stats
        self.player_stats = {
            p.user_id: PlayerTournamentStats(player_id=p.user_id)
            for p in self.player_pool
        }

        # Track team ID generation
        self._next_team_id = 1

    def _generate_player_pool(self):
        """Generate pool of players with realistic skill distribution."""
        # Generate players
        self.player_pool = self.player_gen.generate_players(
            n_players=self.config.n_players,
            skill_distribution=self.config.skill_distribution,
            skill_params=self.config.skill_params,
        )

        # Create player lookup
        self.player_lookup = {p.user_id: p for p in self.player_pool}

        # Assign activity levels (how many tournaments each player will join)
        self._assign_player_activity_levels()

        # Create friend groups for team formation
        self._create_friend_groups()

    def _assign_player_activity_levels(self):
        """Assign how many tournaments each player will participate in."""
        n_players = len(self.player_pool)

        if self.config.player_activity_distribution == "power_law":
            # Power law: Most players play few tournaments, few players play many
            # Use Pareto distribution
            shape = 1.5  # Shape parameter (lower = more extreme)
            raw_values = self.rng.pareto(shape, n_players)

            # Scale to desired range
            min_t = self.config.min_tournaments_per_player
            max_t = self.config.max_tournaments_per_player
            avg_t = self.config.avg_tournaments_per_player

            # Normalize and scale
            raw_values = raw_values / raw_values.mean() * avg_t
            raw_values = np.clip(raw_values, min_t, max_t)

        elif self.config.player_activity_distribution == "normal":
            # Normal distribution around average
            mean = self.config.avg_tournaments_per_player
            std = mean / 3
            raw_values = self.rng.normal(mean, std, n_players)
            raw_values = np.clip(
                raw_values,
                self.config.min_tournaments_per_player,
                self.config.max_tournaments_per_player,
            )
        else:  # uniform
            raw_values = self.rng.uniform(
                self.config.min_tournaments_per_player,
                self.config.max_tournaments_per_player,
                n_players,
            )

        # Assign to players
        for player, n_tournaments in zip(self.player_pool, raw_values):
            player.target_tournaments = int(n_tournaments)

    def _create_friend_groups(self):
        """Create friend groups - players who prefer to team together."""
        # Group players by skill tier
        skill_tiers = 10
        players_by_tier = {i: [] for i in range(skill_tiers)}

        # Normalize skills to 0-1 range for tier assignment
        all_skills = [p.true_skill for p in self.player_pool]
        min_skill = min(all_skills)
        max_skill = max(all_skills)
        skill_range = max_skill - min_skill if max_skill > min_skill else 1.0

        for player in self.player_pool:
            # Normalize skill to 0-1
            normalized_skill = (player.true_skill - min_skill) / skill_range
            tier = min(int(normalized_skill * skill_tiers), skill_tiers - 1)
            tier = max(0, tier)  # Ensure non-negative
            players_by_tier[tier].append(player)

        # Create friend groups within tiers
        for player in self.player_pool:
            # Each player has 3-10 "friends" they prefer to team with
            n_friends = self.rng.integers(3, 11)

            # Get player's tier (normalize again)
            normalized_skill = (player.true_skill - min_skill) / skill_range
            tier = min(int(normalized_skill * skill_tiers), skill_tiers - 1)
            tier = max(0, tier)

            # Select friends from same or adjacent tiers
            potential_friends = []
            for t in range(max(0, tier - 1), min(skill_tiers, tier + 2)):
                potential_friends.extend(players_by_tier[t])

            # Remove self
            potential_friends = [
                p for p in potential_friends if p.user_id != player.user_id
            ]

            # Select random friends
            if potential_friends:
                n_friends = min(n_friends, len(potential_friends))
                friends = self.rng.choice(
                    potential_friends, n_friends, replace=False
                )
                player.friend_group = [f.user_id for f in friends]
            else:
                player.friend_group = []

    def generate_circuit(
        self,
        n_tournaments: Optional[int] = None,
        start_date: Optional[datetime] = None,
    ) -> CircuitResults:
        """
        Generate a tournament circuit with player-centric participation.

        Parameters
        ----------
        n_tournaments : int, optional
            Number of tournaments to generate
        start_date : datetime, optional
            Start date for the circuit

        Returns
        -------
        CircuitResults
            Generated circuit results
        """
        if n_tournaments is None:
            n_tournaments = self.config.n_tournaments

        if start_date is None:
            start_date = datetime.now()

        tournaments = []

        for t_idx in range(n_tournaments):
            # Determine tournament date
            tournament_date = start_date + timedelta(
                days=t_idx * self.config.tournament_interval_days
            )

            # Generate tournament
            tournament = self._generate_tournament(
                tournament_id=t_idx + 1, tournament_date=tournament_date
            )

            tournaments.append(tournament)

        # Calculate circuit statistics
        player_participation = {}
        player_wins = {}
        player_matches = {}

        for player_id, stats in self.player_stats.items():
            player_participation[player_id] = list(stats.tournaments_played)
            player_wins[player_id] = stats.wins
            player_matches[player_id] = stats.matches

        # Report participation statistics
        self._report_participation_stats()

        return CircuitResults(
            tournaments=tournaments,
            player_participation=player_participation,
            player_wins=player_wins,
            player_matches=player_matches,
        )

    def _generate_tournament(
        self, tournament_id: int, tournament_date: datetime
    ) -> Tournament:
        """Generate a single tournament with player-based team formation."""

        # Determine tournament type and size
        tournament_type = self._determine_tournament_type(tournament_id)
        n_teams = self.rng.integers(
            self.config.teams_per_tournament_range[0],
            self.config.teams_per_tournament_range[1] + 1,
        )

        # Select participating players
        participating_players = self._select_tournament_players(
            tournament_id, tournament_type, n_teams * self.config.team_size
        )

        # Form teams from participating players
        teams = self._form_teams(participating_players, tournament_id)

        # Create tournament
        tournament = Tournament(
            tournament_id=tournament_id,
            name=f"Tournament_{tournament_id}",
            start_date=tournament_date,
            all_teams=teams,
        )

        # Generate tournament structure (Swiss, Single Elim, etc.)
        format = self._select_tournament_format(tournament_type, len(teams))

        if format == TournamentFormat.SWISS:
            n_rounds = min(7, int(np.log2(len(teams))) + 2)
            stage = self.tournament_gen._generate_swiss_stage(
                teams, n_rounds=n_rounds
            )
        elif format == TournamentFormat.SINGLE_ELIMINATION:
            stage = self.tournament_gen._generate_single_elimination_stage(
                teams, seeded=True
            )
        elif format == TournamentFormat.DOUBLE_ELIMINATION:
            stage = self.tournament_gen._generate_double_elimination_stage(
                teams, seeded=True
            )
        else:
            # Default to round robin for small tournaments
            stage = self.tournament_gen._generate_round_robin_stage(
                teams, double=False
            )

        tournament.stages = [stage]

        # Simulate matches
        for round_matches in stage.rounds.values():
            for match in round_matches:
                self.match_sim.simulate_match(match)
                self._update_player_stats(match)

        return tournament

    def _determine_tournament_type(self, tournament_id: int) -> TournamentType:
        """Determine tournament type based on ID."""
        if tournament_id % 7 == 0:
            return TournamentType.INVITATIONAL  # Elite
        elif tournament_id % 5 == 0:
            return TournamentType.SKILL_CAPPED  # Amateur
        elif tournament_id % 3 == 0:
            return TournamentType.MIXED
        else:
            return TournamentType.OPEN

    def _select_tournament_players(
        self,
        tournament_id: int,
        tournament_type: TournamentType,
        n_players_needed: int,
    ) -> List[SyntheticPlayer]:
        """Select players for a tournament based on type and activity."""

        # Filter by tournament type
        if tournament_type == TournamentType.INVITATIONAL:
            # Elite tournaments - high skill players
            skill_threshold = np.percentile(
                [p.true_skill for p in self.player_pool],
                self.config.elite_percentile * 100,
            )
            eligible = [
                p for p in self.player_pool if p.true_skill >= skill_threshold
            ]
        elif tournament_type == TournamentType.SKILL_CAPPED:
            # Amateur tournaments - lower skill players
            skill_threshold = np.percentile(
                [p.true_skill for p in self.player_pool],
                self.config.amateur_percentile * 100,
            )
            eligible = [
                p for p in self.player_pool if p.true_skill <= skill_threshold
            ]
        else:
            # Open or mixed - all players eligible
            eligible = self.player_pool.copy()

        # Filter by player activity (have they played too many tournaments?)
        available = []
        for player in eligible:
            stats = self.player_stats[player.user_id]
            if len(stats.tournaments_played) < player.target_tournaments:
                # Player still wants to play more tournaments
                # Add some randomness so not everyone plays their first N tournaments
                if (
                    self.rng.random() < 0.7
                ):  # 70% chance to join if under target
                    available.append(player)
            elif self.rng.random() < 0.1:  # 10% chance to play extra
                available.append(player)

        # Select players
        if len(available) >= n_players_needed:
            selected = self.rng.choice(
                available, n_players_needed, replace=False
            )
        else:
            # Not enough available, pad with any eligible
            selected = available
            remaining_needed = n_players_needed - len(selected)
            remaining_eligible = [p for p in eligible if p not in selected]
            if remaining_eligible:
                extra = self.rng.choice(
                    remaining_eligible,
                    min(remaining_needed, len(remaining_eligible)),
                    replace=False,
                )
                selected = list(selected) + list(extra)

        # Update stats
        for player in selected:
            self.player_stats[player.user_id].tournaments_played.add(
                tournament_id
            )

        return list(selected)

    def _form_teams(
        self, players: List[SyntheticPlayer], tournament_id: int
    ) -> List[Team]:
        """Form teams from participating players."""
        teams = []
        remaining_players = players.copy()
        self.rng.shuffle(remaining_players)

        while len(remaining_players) >= self.config.team_size:
            # Form one team
            if self.config.team_formation_strategy == "random":
                # Random team formation
                team_players = remaining_players[: self.config.team_size]
                remaining_players = remaining_players[self.config.team_size :]

            elif self.config.team_formation_strategy == "skill_based":
                # Group by similar skill
                team_players = self._form_skill_based_team(remaining_players)

            elif self.config.team_formation_strategy == "friend_groups":
                # Prefer teaming with friends
                team_players = self._form_friend_based_team(remaining_players)

            else:  # mixed
                # Mix of strategies
                strategy = self.rng.choice(
                    ["random", "skill_based", "friend_groups"]
                )
                if strategy == "random":
                    team_players = remaining_players[: self.config.team_size]
                    remaining_players = remaining_players[
                        self.config.team_size :
                    ]
                elif strategy == "skill_based":
                    team_players = self._form_skill_based_team(
                        remaining_players
                    )
                else:
                    team_players = self._form_friend_based_team(
                        remaining_players
                    )

            # Create team with unique ID for this tournament
            team = Team(
                team_id=self._next_team_id,
                name=f"Team_{self._next_team_id}",
                players=team_players,
            )
            teams.append(team)
            self._next_team_id += 1

            # Update player stats
            for player in team_players:
                stats = self.player_stats[player.user_id]
                stats.teams_formed[tournament_id] = team.team_id
                for teammate in team_players:
                    if teammate.user_id != player.user_id:
                        stats.teammates_history.add(teammate.user_id)

            # Remove from remaining
            for player in team_players:
                if player in remaining_players:
                    remaining_players.remove(player)

        return teams

    def _form_skill_based_team(
        self, available: List[SyntheticPlayer]
    ) -> List[SyntheticPlayer]:
        """Form a team based on similar skill levels."""
        if len(available) < self.config.team_size:
            return available

        # Sort by skill
        sorted_players = sorted(available, key=lambda p: p.true_skill)

        # Take consecutive players (similar skill)
        start_idx = self.rng.integers(
            0, max(1, len(sorted_players) - self.config.team_size + 1)
        )
        return sorted_players[start_idx : start_idx + self.config.team_size]

    def _form_friend_based_team(
        self, available: List[SyntheticPlayer]
    ) -> List[SyntheticPlayer]:
        """Form a team based on friend groups."""
        if len(available) < self.config.team_size:
            return available

        # Pick a random starting player
        team = [self.rng.choice(available)]
        available_ids = {p.user_id for p in available}

        # Try to add friends
        while len(team) < self.config.team_size:
            # Get friends of current team members
            potential_friends = set()
            for player in team:
                if hasattr(player, "friend_group"):
                    potential_friends.update(player.friend_group)

            # Filter to available players
            potential_friends &= available_ids
            potential_friends -= {p.user_id for p in team}

            if potential_friends:
                # Add a friend
                friend_id = self.rng.choice(list(potential_friends))
                friend = next(p for p in available if p.user_id == friend_id)
                team.append(friend)
            else:
                # No friends available, add random
                remaining = [p for p in available if p not in team]
                if remaining:
                    team.append(self.rng.choice(remaining))
                else:
                    break

        return team

    def _select_tournament_format(
        self, tournament_type: TournamentType, n_teams: int
    ) -> TournamentFormat:
        """Select appropriate tournament format."""
        if tournament_type == TournamentType.INVITATIONAL:
            # Elite tournaments use double elimination
            return TournamentFormat.DOUBLE_ELIMINATION
        elif n_teams <= 8:
            # Small tournaments use round robin
            return TournamentFormat.ROUND_ROBIN
        elif n_teams <= 32:
            # Medium tournaments use single elimination
            return TournamentFormat.SINGLE_ELIMINATION
        else:
            # Large tournaments use Swiss
            return TournamentFormat.SWISS

    def _update_player_stats(self, match):
        """Update player statistics after a match."""
        if match.winner:
            # Update match counts
            for team in [match.team_a, match.team_b]:
                for player in team.players:
                    self.player_stats[player.user_id].matches += 1

            # Update wins
            for player in match.winner.players:
                self.player_stats[player.user_id].wins += 1

    def _report_participation_stats(self):
        """Report statistics about player participation."""
        participation_counts = {}
        for stats in self.player_stats.values():
            n_tournaments = len(stats.tournaments_played)
            if n_tournaments > 0:
                if n_tournaments not in participation_counts:
                    participation_counts[n_tournaments] = 0
                participation_counts[n_tournaments] += 1

        print(
            f"\nPlayer participation across {self.config.n_tournaments} tournaments:"
        )
        print("Tournaments | Players")
        print("-" * 20)
        for count in sorted(participation_counts.keys())[:20]:
            print(f"    {count:3}     | {participation_counts[count]:5}")

        # Calculate overlap statistics
        total_players = len(
            [
                s
                for s in self.player_stats.values()
                if len(s.tournaments_played) > 0
            ]
        )
        multi_tournament = len(
            [
                s
                for s in self.player_stats.values()
                if len(s.tournaments_played) > 1
            ]
        )

        print(
            f"\nPlayers in multiple tournaments: {multi_tournament}/{total_players} "
            f"({100*multi_tournament/total_players:.1f}%)"
        )

        avg_tournaments = np.mean(
            [len(s.tournaments_played) for s in self.player_stats.values()]
        )
        print(f"Average tournaments per player: {avg_tournaments:.1f}")
