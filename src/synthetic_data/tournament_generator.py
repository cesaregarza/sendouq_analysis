"""
Tournament structure generation module for synthetic tournament data.

This module provides functionality to generate various tournament formats
including Swiss, Round Robin, Single Elimination, and Double Elimination.
"""

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from synthetic_data.player_generator import SyntheticPlayer


class TournamentFormat(Enum):
    """Supported tournament formats."""

    SWISS = "swiss"
    ROUND_ROBIN = "round_robin"
    SINGLE_ELIMINATION = "single_elimination"
    DOUBLE_ELIMINATION = "double_elimination"
    GROUP_STAGE = "group_stage"


@dataclass
class Team:
    """Represents a team in a tournament."""

    team_id: int
    name: str
    players: List[SyntheticPlayer]
    seed: Optional[int] = None

    @property
    def avg_skill(self) -> float:
        """Calculate average team skill."""
        if not self.players:
            return 0.0
        return sum(p.true_skill for p in self.players) / len(self.players)


@dataclass
class Match:
    """Represents a match between two teams."""

    match_id: int
    team_a: Team
    team_b: Team
    round_number: int
    stage: str
    group: Optional[str] = None
    winner: Optional[Team] = None
    score_a: Optional[int] = None
    score_b: Optional[int] = None
    timestamp: Optional[datetime] = None


@dataclass
class TournamentStage:
    """Represents a stage within a tournament."""

    stage_id: int
    name: str
    format: TournamentFormat
    rounds: Dict[int, List[Match]] = field(default_factory=dict)
    teams: List[Team] = field(default_factory=list)


@dataclass
class Tournament:
    """Represents a complete tournament."""

    tournament_id: int
    name: str
    start_date: datetime
    stages: List[TournamentStage] = field(default_factory=list)
    all_teams: List[Team] = field(default_factory=list)


class TournamentGenerator:
    """Generates synthetic tournament structures."""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the tournament generator.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._next_tournament_id = 1
        self._next_team_id = 1
        self._next_match_id = 1
        self._next_stage_id = 1

    def generate_tournament(
        self,
        players: List[SyntheticPlayer],
        format: TournamentFormat,
        team_size: int = 4,
        name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        **kwargs,
    ) -> Tournament:
        """
        Generate a complete tournament.

        Parameters
        ----------
        players : List[SyntheticPlayer]
            Pool of players to form teams
        format : TournamentFormat
            Tournament format to use
        team_size : int
            Number of players per team
        name : str, optional
            Tournament name
        start_date : datetime, optional
            Tournament start date
        **kwargs
            Additional format-specific parameters

        Returns
        -------
        Tournament
            Generated tournament structure
        """
        if name is None:
            name = f"Tournament_{self._next_tournament_id}"
        if start_date is None:
            start_date = datetime.now()

        # Form teams from player pool
        teams = self._form_teams(players, team_size)

        # Create tournament
        tournament = Tournament(
            tournament_id=self._next_tournament_id,
            name=name,
            start_date=start_date,
            all_teams=teams,
        )
        self._next_tournament_id += 1

        # Generate stages based on format
        if format == TournamentFormat.SWISS:
            stage = self._generate_swiss_stage(teams, **kwargs)
        elif format == TournamentFormat.ROUND_ROBIN:
            stage = self._generate_round_robin_stage(teams, **kwargs)
        elif format == TournamentFormat.SINGLE_ELIMINATION:
            stage = self._generate_single_elimination_stage(teams, **kwargs)
        elif format == TournamentFormat.DOUBLE_ELIMINATION:
            stage = self._generate_double_elimination_stage(teams, **kwargs)
        elif format == TournamentFormat.GROUP_STAGE:
            stage = self._generate_group_stage(teams, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

        tournament.stages.append(stage)

        # Assign timestamps to matches
        self._assign_match_timestamps(tournament, start_date)

        return tournament

    def generate_multi_stage_tournament(
        self,
        players: List[SyntheticPlayer],
        stages_config: List[Dict],
        team_size: int = 4,
        name: Optional[str] = None,
        start_date: Optional[datetime] = None,
    ) -> Tournament:
        """
        Generate a tournament with multiple stages.

        Parameters
        ----------
        players : List[SyntheticPlayer]
            Pool of players to form teams
        stages_config : List[Dict]
            Configuration for each stage
        team_size : int
            Number of players per team
        name : str, optional
            Tournament name
        start_date : datetime, optional
            Tournament start date

        Returns
        -------
        Tournament
            Generated multi-stage tournament
        """
        if name is None:
            name = f"MultiStage_Tournament_{self._next_tournament_id}"
        if start_date is None:
            start_date = datetime.now()

        # Form initial teams
        teams = self._form_teams(players, team_size)

        tournament = Tournament(
            tournament_id=self._next_tournament_id,
            name=name,
            start_date=start_date,
            all_teams=teams,
        )
        self._next_tournament_id += 1

        # Generate each stage
        current_teams = teams
        for config in stages_config:
            format = TournamentFormat(config["format"])

            if format == TournamentFormat.SWISS:
                stage = self._generate_swiss_stage(current_teams, **config)
            elif format == TournamentFormat.ROUND_ROBIN:
                stage = self._generate_round_robin_stage(
                    current_teams, **config
                )
            elif format == TournamentFormat.SINGLE_ELIMINATION:
                stage = self._generate_single_elimination_stage(
                    current_teams, **config
                )
            elif format == TournamentFormat.DOUBLE_ELIMINATION:
                stage = self._generate_double_elimination_stage(
                    current_teams, **config
                )
            elif format == TournamentFormat.GROUP_STAGE:
                stage = self._generate_group_stage(current_teams, **config)

            tournament.stages.append(stage)

            # Advance teams to next stage if specified
            if "advance_count" in config:
                current_teams = self._get_advancing_teams(
                    stage, config["advance_count"]
                )

        # Assign timestamps
        self._assign_match_timestamps(tournament, start_date)

        return tournament

    def _form_teams(
        self, players: List[SyntheticPlayer], team_size: int
    ) -> List[Team]:
        """Form teams from player pool."""
        # Shuffle players
        shuffled_players = players.copy()
        self.rng.shuffle(shuffled_players)

        teams = []
        for i in range(0, len(shuffled_players), team_size):
            if i + team_size <= len(shuffled_players):
                team_players = shuffled_players[i : i + team_size]
                team = Team(
                    team_id=self._next_team_id,
                    name=f"Team_{self._next_team_id}",
                    players=team_players,
                )
                teams.append(team)
                self._next_team_id += 1

        return teams

    def _generate_swiss_stage(
        self, teams: List[Team], n_rounds: Optional[int] = None, **kwargs
    ) -> TournamentStage:
        """Generate Swiss system stage."""
        if n_rounds is None:
            # Standard Swiss rounds based on team count
            n_rounds = int(np.ceil(np.log2(len(teams))))

        stage = TournamentStage(
            stage_id=self._next_stage_id,
            name="Swiss",
            format=TournamentFormat.SWISS,
            teams=teams,
        )
        self._next_stage_id += 1

        # Track team scores for pairing
        scores = {team.team_id: 0 for team in teams}
        played_pairs: Set[Tuple[int, int]] = set()

        for round_num in range(1, n_rounds + 1):
            # Sort teams by score (and randomize within same score)
            sorted_teams = sorted(
                teams,
                key=lambda t: (scores[t.team_id], self.rng.random()),
                reverse=True,
            )

            # Pair teams with similar scores
            round_matches = []
            paired_teams = set()

            for i, team_a in enumerate(sorted_teams):
                if team_a.team_id in paired_teams:
                    continue

                # Find best opponent
                for j in range(i + 1, len(sorted_teams)):
                    team_b = sorted_teams[j]
                    if team_b.team_id in paired_teams:
                        continue

                    pair = tuple(sorted([team_a.team_id, team_b.team_id]))
                    if pair not in played_pairs:
                        # Create match
                        match = Match(
                            match_id=self._next_match_id,
                            team_a=team_a,
                            team_b=team_b,
                            round_number=round_num,
                            stage="Swiss",
                        )
                        self._next_match_id += 1

                        round_matches.append(match)
                        paired_teams.add(team_a.team_id)
                        paired_teams.add(team_b.team_id)
                        played_pairs.add(pair)

                        # Update scores (will be replaced by match simulator)
                        if self.rng.random() > 0.5:
                            scores[team_a.team_id] += 1
                        else:
                            scores[team_b.team_id] += 1
                        break

            stage.rounds[round_num] = round_matches

        return stage

    def _generate_round_robin_stage(
        self, teams: List[Team], double: bool = False, **kwargs
    ) -> TournamentStage:
        """Generate Round Robin stage."""
        stage = TournamentStage(
            stage_id=self._next_stage_id,
            name="Round Robin",
            format=TournamentFormat.ROUND_ROBIN,
            teams=teams,
        )
        self._next_stage_id += 1

        n_teams = len(teams)
        n_rounds = n_teams - 1 if n_teams % 2 == 0 else n_teams

        # Generate round robin schedule
        for round_num in range(1, n_rounds + 1):
            round_matches = []

            for i in range(n_teams // 2):
                if i == 0 and n_teams % 2 == 1:
                    continue  # Bye for odd number of teams

                team_a_idx = i
                team_b_idx = n_teams - 1 - i

                # Rotate teams for round robin
                if round_num > 1:
                    team_a_idx = (team_a_idx + round_num - 1) % n_teams
                    team_b_idx = (team_b_idx + round_num - 1) % n_teams

                match = Match(
                    match_id=self._next_match_id,
                    team_a=teams[team_a_idx],
                    team_b=teams[team_b_idx],
                    round_number=round_num,
                    stage="Round Robin",
                )
                self._next_match_id += 1
                round_matches.append(match)

            stage.rounds[round_num] = round_matches

        # Double round robin if requested
        if double:
            for round_num in range(n_rounds + 1, 2 * n_rounds + 1):
                original_round = round_num - n_rounds
                round_matches = []

                for match in stage.rounds[original_round]:
                    # Swap teams for return leg
                    return_match = Match(
                        match_id=self._next_match_id,
                        team_a=match.team_b,
                        team_b=match.team_a,
                        round_number=round_num,
                        stage="Round Robin",
                    )
                    self._next_match_id += 1
                    round_matches.append(return_match)

                stage.rounds[round_num] = round_matches

        return stage

    def _generate_single_elimination_stage(
        self, teams: List[Team], seeded: bool = True, **kwargs
    ) -> TournamentStage:
        """Generate Single Elimination bracket."""
        stage = TournamentStage(
            stage_id=self._next_stage_id,
            name="Single Elimination",
            format=TournamentFormat.SINGLE_ELIMINATION,
            teams=teams,
        )
        self._next_stage_id += 1

        # Seed teams if requested
        if seeded:
            sorted_teams = sorted(
                teams, key=lambda t: t.avg_skill, reverse=True
            )
            for i, team in enumerate(sorted_teams):
                team.seed = i + 1
        else:
            shuffled_teams = teams.copy()
            self.rng.shuffle(shuffled_teams)
            sorted_teams = shuffled_teams

        # Generate bracket
        current_teams = sorted_teams
        round_num = 1

        while len(current_teams) > 1:
            round_matches = []
            next_round_teams = []

            # Pair teams (seeded pairing if applicable)
            if seeded and round_num == 1:
                # First round: 1 vs last, 2 vs second-last, etc.
                n = len(current_teams)
                for i in range(n // 2):
                    match = Match(
                        match_id=self._next_match_id,
                        team_a=current_teams[i],
                        team_b=current_teams[n - 1 - i],
                        round_number=round_num,
                        stage="Single Elimination",
                    )
                    self._next_match_id += 1
                    round_matches.append(match)

                    # Simulate winner (higher seed more likely)
                    if self.rng.random() < 0.7:  # 70% chance for better team
                        winner = (
                            current_teams[i]
                            if current_teams[i].avg_skill
                            > current_teams[n - 1 - i].avg_skill
                            else current_teams[n - 1 - i]
                        )
                    else:
                        winner = (
                            current_teams[n - 1 - i]
                            if current_teams[i].avg_skill
                            > current_teams[n - 1 - i].avg_skill
                            else current_teams[i]
                        )
                    next_round_teams.append(winner)
            else:
                # Subsequent rounds: adjacent pairing
                for i in range(0, len(current_teams), 2):
                    if i + 1 < len(current_teams):
                        match = Match(
                            match_id=self._next_match_id,
                            team_a=current_teams[i],
                            team_b=current_teams[i + 1],
                            round_number=round_num,
                            stage="Single Elimination",
                        )
                        self._next_match_id += 1
                        round_matches.append(match)

                        # Simulate winner
                        if self.rng.random() < 0.6:
                            winner = (
                                current_teams[i]
                                if current_teams[i].avg_skill
                                > current_teams[i + 1].avg_skill
                                else current_teams[i + 1]
                            )
                        else:
                            winner = (
                                current_teams[i + 1]
                                if current_teams[i].avg_skill
                                > current_teams[i + 1].avg_skill
                                else current_teams[i]
                            )
                        next_round_teams.append(winner)
                    else:
                        # Bye
                        next_round_teams.append(current_teams[i])

            stage.rounds[round_num] = round_matches
            current_teams = next_round_teams
            round_num += 1

        return stage

    def _generate_double_elimination_stage(
        self, teams: List[Team], seeded: bool = True, **kwargs
    ) -> TournamentStage:
        """Generate Double Elimination bracket."""
        # Simplified version - would need more complex logic for full implementation
        stage = TournamentStage(
            stage_id=self._next_stage_id,
            name="Double Elimination",
            format=TournamentFormat.DOUBLE_ELIMINATION,
            teams=teams,
        )
        self._next_stage_id += 1

        # For now, generate upper bracket similar to single elimination
        # Full implementation would include lower bracket and grand finals
        return self._generate_single_elimination_stage(teams, seeded=seeded)

    def _generate_group_stage(
        self,
        teams: List[Team],
        n_groups: int = 4,
        advance_per_group: int = 2,
        **kwargs,
    ) -> TournamentStage:
        """Generate Group Stage with round robin within groups."""
        stage = TournamentStage(
            stage_id=self._next_stage_id,
            name="Group Stage",
            format=TournamentFormat.GROUP_STAGE,
            teams=teams,
        )
        self._next_stage_id += 1

        # Divide teams into groups
        shuffled_teams = teams.copy()
        self.rng.shuffle(shuffled_teams)

        groups = [[] for _ in range(n_groups)]
        for i, team in enumerate(shuffled_teams):
            groups[i % n_groups].append(team)

        # Generate round robin within each group
        max_rounds = max(len(group) - 1 for group in groups)

        for round_num in range(1, max_rounds + 1):
            round_matches = []

            for group_idx, group_teams in enumerate(groups):
                group_name = f"Group_{chr(65 + group_idx)}"  # A, B, C, D...

                # Round robin matches for this group
                n_teams = len(group_teams)
                if n_teams < 2:
                    continue

                for i in range(n_teams // 2):
                    team_a_idx = (i + round_num - 1) % n_teams
                    team_b_idx = (n_teams - 1 - i + round_num - 1) % n_teams

                    if team_a_idx != team_b_idx:
                        match = Match(
                            match_id=self._next_match_id,
                            team_a=group_teams[team_a_idx],
                            team_b=group_teams[team_b_idx],
                            round_number=round_num,
                            stage="Group Stage",
                            group=group_name,
                        )
                        self._next_match_id += 1
                        round_matches.append(match)

            stage.rounds[round_num] = round_matches

        return stage

    def _get_advancing_teams(
        self, stage: TournamentStage, advance_count: int
    ) -> List[Team]:
        """Get teams advancing from a stage."""
        # Simplified - would calculate based on match results
        teams = stage.teams.copy()
        self.rng.shuffle(teams)
        return teams[:advance_count]

    def _assign_match_timestamps(
        self, tournament: Tournament, start_date: datetime
    ):
        """Assign timestamps to all matches in tournament."""
        current_time = start_date
        match_duration = timedelta(minutes=30)  # Assume 30 min per match
        round_break = timedelta(hours=1)  # 1 hour between rounds

        for stage in tournament.stages:
            for round_num in sorted(stage.rounds.keys()):
                for match in stage.rounds[round_num]:
                    match.timestamp = current_time
                    current_time += match_duration
                current_time += round_break
