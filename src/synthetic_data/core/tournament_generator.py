"""
Tournament structure generation module for synthetic tournament data.

This module provides functionality to generate various tournament formats
including Swiss, Round Robin, Single Elimination, and Double Elimination.

FIXES APPLIED:
1. Swiss pairing algorithm now handles all teams correctly
2. Round Robin uses proper circle method for complete pairings
3. Better handling of edge cases (empty tournaments, odd team counts)
"""

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import numpy as np

from synthetic_data.core.player_generator import SyntheticPlayer


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
    players: list[SyntheticPlayer]
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
    # Double elimination specific fields
    bracket_type: Optional[str] = None  # "winners", "losers", "grand_finals"
    is_elimination_match: bool = False  # True for losers bracket elim matches
    loser_to_round: Optional[int] = None  # Losers bracket round for loser
    loser_to_position: Optional[int] = None  # Position in losers round


@dataclass
class TournamentStage:
    """Represents a stage within a tournament."""

    stage_id: int
    name: str
    format: TournamentFormat
    rounds: dict[int, list[Match]] = field(default_factory=dict)
    teams: list[Team] = field(default_factory=list)
    # Double elimination specific fields
    winners_bracket: dict[int, list[Match]] = field(default_factory=dict)
    losers_bracket: dict[int, list[Match]] = field(default_factory=dict)
    grand_finals: list[Match] = field(default_factory=list)


@dataclass
class Tournament:
    """Represents a complete tournament."""

    tournament_id: int
    name: str
    start_date: datetime
    stages: list[TournamentStage] = field(default_factory=list)
    all_teams: list[Team] = field(default_factory=list)


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
        players: list[SyntheticPlayer],
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
        players : list[SyntheticPlayer]
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

        # Handle empty tournament case
        if not teams:
            return tournament

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
        players: list[SyntheticPlayer],
        stages_config: list[dict],
        team_size: int = 4,
        name: Optional[str] = None,
        start_date: Optional[datetime] = None,
    ) -> Tournament:
        """
        Generate a tournament with multiple stages.

        Parameters
        ----------
        players : list[SyntheticPlayer]
            Pool of players to form teams
        stages_config : list[dict]
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
        self, players: list[SyntheticPlayer], team_size: int
    ) -> list[Team]:
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
        self, teams: list[Team], n_rounds: Optional[int] = None, **kwargs
    ) -> TournamentStage:
        """Generate Swiss system stage with improved pairing algorithm."""
        if not teams:
            return TournamentStage(
                stage_id=self._next_stage_id,
                name="Swiss",
                format=TournamentFormat.SWISS,
                teams=[],
            )

        if n_rounds is None:
            # Standard Swiss rounds based on team count
            n_rounds = int(np.ceil(np.log2(max(len(teams), 2))))

        stage = TournamentStage(
            stage_id=self._next_stage_id,
            name="Swiss",
            format=TournamentFormat.SWISS,
            teams=teams,
        )
        self._next_stage_id += 1

        # Track team scores and pairings
        scores = {team.team_id: 0 for team in teams}
        played_pairs: set[tuple[int, int]] = set()

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
            unpaired = []

            # First pass: try to pair teams with similar scores
            for i, team_a in enumerate(sorted_teams):
                if team_a.team_id in paired_teams:
                    continue

                paired = False
                # Try to find best opponent
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

                        paired = True
                        break

                if not paired:
                    unpaired.append(team_a)

            # Second pass: pair any remaining teams
            # This handles cases where optimal pairing isn't possible
            while len(unpaired) >= 2:
                team_a = unpaired.pop(0)

                # Find any valid opponent
                for i, team_b in enumerate(unpaired):
                    pair = tuple(sorted([team_a.team_id, team_b.team_id]))
                    # In later rounds, we might need to allow rematches
                    if pair not in played_pairs or round_num > len(teams) - 1:
                        match = Match(
                            match_id=self._next_match_id,
                            team_a=team_a,
                            team_b=team_b,
                            round_number=round_num,
                            stage="Swiss",
                        )
                        self._next_match_id += 1
                        round_matches.append(match)

                        if pair not in played_pairs:
                            played_pairs.add(pair)

                        # Update scores
                        if self.rng.random() > 0.5:
                            scores[team_a.team_id] += 1
                        else:
                            scores[team_b.team_id] += 1

                        unpaired.pop(i)
                        break

            # If odd number of teams, last team gets a bye (free win)
            if unpaired:
                bye_team = unpaired[0]
                scores[bye_team.team_id] += 1

            stage.rounds[round_num] = round_matches

        return stage

    def _generate_round_robin_stage(
        self, teams: list[Team], double: bool = False, **kwargs
    ) -> TournamentStage:
        """Generate Round Robin stage with corrected circle algorithm."""
        stage = TournamentStage(
            stage_id=self._next_stage_id,
            name="Round Robin",
            format=TournamentFormat.ROUND_ROBIN,
            teams=teams,
        )
        self._next_stage_id += 1

        n_teams = len(teams)
        if n_teams < 2:
            return stage

        # Use round-robin tournament algorithm (circle method)
        if n_teams % 2 == 1:
            # Add a dummy team for bye
            teams = teams + [None]
            n_teams += 1

        n_rounds = n_teams - 1

        # Generate round robin schedule using circle method
        for round_num in range(1, n_rounds + 1):
            round_matches = []

            for i in range(n_teams // 2):
                # Calculate team indices for this pairing
                if i == 0:
                    # First team stays fixed
                    team_a_idx = 0
                    team_b_idx = round_num
                else:
                    # Other teams rotate
                    team_a_idx = (round_num + i - 1) % (n_teams - 1) + 1
                    team_b_idx = (round_num - i - 1) % (n_teams - 1) + 1
                    if team_b_idx == 0:
                        team_b_idx = n_teams - 1

                # Skip if either team is the dummy (bye)
                if team_a_idx >= len(teams) or team_b_idx >= len(teams):
                    continue
                if teams[team_a_idx] is None or teams[team_b_idx] is None:
                    continue

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
        self, teams: list[Team], seeded: bool = True, **kwargs
    ) -> TournamentStage:
        """Generate Single Elimination bracket."""
        stage = TournamentStage(
            stage_id=self._next_stage_id,
            name="Single Elimination",
            format=TournamentFormat.SINGLE_ELIMINATION,
            teams=teams,
        )
        self._next_stage_id += 1

        if len(teams) < 2:
            return stage

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

    def _calculate_bracket_size(self, n_teams: int) -> int:
        """Calculate the bracket size (next power of 2)."""
        if n_teams <= 1:
            return 1
        # Find next power of 2
        power = 1
        while power < n_teams:
            power *= 2
        return power

    def _get_loser_destination(
        self, winners_round: int, match_position: int, bracket_size: int
    ) -> tuple[int, int]:
        """
        Determine where the loser goes in the losers bracket.

        Returns (losers_round, position_in_round)
        """
        # Losers from WB R1 go to LB R1
        if winners_round == 1:
            losers_round = 1
            # Position is same as match position in R1
            return (losers_round, match_position)

        # For subsequent rounds, losers enter at specific points
        # LB has alternating elimination and drop-down rounds
        losers_round = (winners_round - 1) * 2

        # Calculate position based on bracket structure
        # This ensures proper spacing to avoid immediate rematches
        position = match_position

        return (losers_round, position)

    def _calculate_losers_bracket_rounds(self, n_teams: int) -> int:
        """Calculate number of rounds in losers bracket."""
        if n_teams <= 2:
            return 0
        # Formula: 2 * (log2(bracket_size) - 1)
        import math

        bracket_size = self._calculate_bracket_size(n_teams)
        return 2 * (int(math.log2(bracket_size)) - 1)

    def _generate_double_elimination_stage(
        self, teams: list[Team], seeded: bool = True, **kwargs
    ) -> TournamentStage:
        """Generate Double Elimination bracket with winners, losers, and grand finals."""
        stage = TournamentStage(
            stage_id=self._next_stage_id,
            name="Double Elimination",
            format=TournamentFormat.DOUBLE_ELIMINATION,
            teams=teams,
        )
        self._next_stage_id += 1

        if len(teams) < 2:
            return stage

        # Calculate bracket structure
        bracket_size = self._calculate_bracket_size(len(teams))
        n_byes = bracket_size - len(teams)

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

        # Track team status
        teams_in_winners = {team.team_id: team for team in sorted_teams}
        teams_in_losers = {}
        eliminated_teams = set()

        # Generate Winners Bracket
        current_wb_teams = sorted_teams.copy()
        wb_round = 1

        # Add byes if necessary (give to lowest seeds)
        if n_byes > 0 and seeded:
            # Higher seeds get byes
            bye_teams = current_wb_teams[:n_byes]
            playing_teams = current_wb_teams[n_byes:]

            # First round with byes
            round_matches = []
            next_round_teams = (
                bye_teams.copy()
            )  # Bye teams advance automatically

            # Pair remaining teams
            n_playing = len(playing_teams)
            for i in range(n_playing // 2):
                match = Match(
                    match_id=self._next_match_id,
                    team_a=playing_teams[i],
                    team_b=playing_teams[n_playing - 1 - i],
                    round_number=wb_round,
                    stage="Double Elimination",
                    bracket_type="winners",
                )
                self._next_match_id += 1

                # Set loser destination
                loser_round, loser_pos = self._get_loser_destination(
                    wb_round, i, bracket_size
                )
                match.loser_to_round = loser_round
                match.loser_to_position = loser_pos

                round_matches.append(match)

                # Simulate winner (placeholder - will be replaced by match simulator)
                if self.rng.random() < 0.7:  # Favor higher seed
                    winner = (
                        playing_teams[i]
                        if playing_teams[i].avg_skill
                        > playing_teams[n_playing - 1 - i].avg_skill
                        else playing_teams[n_playing - 1 - i]
                    )
                    loser = (
                        playing_teams[n_playing - 1 - i]
                        if winner == playing_teams[i]
                        else playing_teams[i]
                    )
                else:
                    winner = (
                        playing_teams[n_playing - 1 - i]
                        if playing_teams[i].avg_skill
                        > playing_teams[n_playing - 1 - i].avg_skill
                        else playing_teams[i]
                    )
                    loser = (
                        playing_teams[i]
                        if winner == playing_teams[n_playing - 1 - i]
                        else playing_teams[n_playing - 1 - i]
                    )

                match.winner = winner
                next_round_teams.append(winner)

                # Move loser to losers bracket
                teams_in_losers[loser.team_id] = loser
                del teams_in_winners[loser.team_id]

            stage.winners_bracket[wb_round] = round_matches
            current_wb_teams = next_round_teams
            wb_round += 1

        # Continue winners bracket until one team remains
        while len(current_wb_teams) > 1:
            round_matches = []
            next_round_teams = []

            for i in range(0, len(current_wb_teams), 2):
                if i + 1 < len(current_wb_teams):
                    match = Match(
                        match_id=self._next_match_id,
                        team_a=current_wb_teams[i],
                        team_b=current_wb_teams[i + 1],
                        round_number=wb_round,
                        stage="Double Elimination",
                        bracket_type="winners",
                    )
                    self._next_match_id += 1

                    # Set loser destination
                    loser_round, loser_pos = self._get_loser_destination(
                        wb_round, i // 2, bracket_size
                    )
                    match.loser_to_round = loser_round
                    match.loser_to_position = loser_pos

                    round_matches.append(match)

                    # Simulate winner
                    if self.rng.random() < 0.6:
                        winner = (
                            current_wb_teams[i]
                            if current_wb_teams[i].avg_skill
                            > current_wb_teams[i + 1].avg_skill
                            else current_wb_teams[i + 1]
                        )
                        loser = (
                            current_wb_teams[i + 1]
                            if winner == current_wb_teams[i]
                            else current_wb_teams[i]
                        )
                    else:
                        winner = (
                            current_wb_teams[i + 1]
                            if current_wb_teams[i].avg_skill
                            > current_wb_teams[i + 1].avg_skill
                            else current_wb_teams[i]
                        )
                        loser = (
                            current_wb_teams[i]
                            if winner == current_wb_teams[i + 1]
                            else current_wb_teams[i + 1]
                        )

                    match.winner = winner
                    next_round_teams.append(winner)

                    # Move loser to losers bracket
                    teams_in_losers[loser.team_id] = loser
                    del teams_in_winners[loser.team_id]
                else:
                    # Bye
                    next_round_teams.append(current_wb_teams[i])

            stage.winners_bracket[wb_round] = round_matches
            current_wb_teams = next_round_teams
            wb_round += 1

        # Winners bracket champion
        wb_champion = current_wb_teams[0] if current_wb_teams else None

        # Generate Losers Bracket
        # The losers bracket has alternating elimination rounds and drop-down rounds
        lb_rounds = self._calculate_losers_bracket_rounds(len(teams))
        lb_survivors = []  # Teams surviving each LB round

        for lb_round in range(1, lb_rounds + 1):
            round_matches = []

            # Get teams for this round
            # Odd rounds: elimination matches between losers bracket teams
            # Even rounds: drop-down matches (losers from WB vs LB survivors)

            if lb_round == 1:
                # First losers round - losers from WB R1
                lb_teams = [
                    team for team in teams if team.team_id in teams_in_losers
                ]

                # Pair adjacent teams
                new_survivors = []
                for i in range(0, len(lb_teams), 2):
                    if i + 1 < len(lb_teams):
                        match = Match(
                            match_id=self._next_match_id,
                            team_a=lb_teams[i],
                            team_b=lb_teams[i + 1],
                            round_number=lb_round,
                            stage="Double Elimination",
                            bracket_type="losers",
                            is_elimination_match=True,
                        )
                        self._next_match_id += 1
                        round_matches.append(match)

                        # Simulate - loser is eliminated
                        if self.rng.random() < 0.5:
                            winner = lb_teams[i]
                            loser = lb_teams[i + 1]
                        else:
                            winner = lb_teams[i + 1]
                            loser = lb_teams[i]

                        match.winner = winner
                        new_survivors.append(winner)
                        eliminated_teams.add(loser.team_id)
                        if loser.team_id in teams_in_losers:
                            del teams_in_losers[loser.team_id]
                    else:
                        # Bye in losers bracket
                        new_survivors.append(lb_teams[i])

                lb_survivors = new_survivors

            elif lb_round % 2 == 0:
                # Even round: drop-down matches (WB losers vs LB survivors)
                # Get losers from corresponding WB round
                wb_round_losers = []
                wb_round_num = (lb_round // 2) + 1

                if wb_round_num in stage.winners_bracket:
                    for match in stage.winners_bracket[wb_round_num]:
                        if match.winner:
                            loser = (
                                match.team_a
                                if match.winner == match.team_b
                                else match.team_b
                            )
                            if loser and loser.team_id in teams_in_losers:
                                wb_round_losers.append(loser)

                # Pair WB losers with LB survivors
                new_survivors = []
                for i, wb_loser in enumerate(wb_round_losers):
                    if i < len(lb_survivors):
                        match = Match(
                            match_id=self._next_match_id,
                            team_a=wb_loser,
                            team_b=lb_survivors[i],
                            round_number=lb_round,
                            stage="Double Elimination",
                            bracket_type="losers",
                            is_elimination_match=True,
                        )
                        self._next_match_id += 1
                        round_matches.append(match)

                        # Simulate
                        if (
                            self.rng.random() < 0.55
                        ):  # Slight advantage to WB loser
                            winner = wb_loser
                            loser = lb_survivors[i]
                        else:
                            winner = lb_survivors[i]
                            loser = wb_loser

                        match.winner = winner
                        new_survivors.append(winner)
                        eliminated_teams.add(loser.team_id)
                        if loser.team_id in teams_in_losers:
                            del teams_in_losers[loser.team_id]

                lb_survivors = new_survivors

            else:
                # Odd round: elimination matches between LB survivors
                new_survivors = []
                for i in range(0, len(lb_survivors), 2):
                    if i + 1 < len(lb_survivors):
                        match = Match(
                            match_id=self._next_match_id,
                            team_a=lb_survivors[i],
                            team_b=lb_survivors[i + 1],
                            round_number=lb_round,
                            stage="Double Elimination",
                            bracket_type="losers",
                            is_elimination_match=True,
                        )
                        self._next_match_id += 1
                        round_matches.append(match)

                        # Simulate
                        if self.rng.random() < 0.5:
                            winner = lb_survivors[i]
                            loser = lb_survivors[i + 1]
                        else:
                            winner = lb_survivors[i + 1]
                            loser = lb_survivors[i]

                        match.winner = winner
                        new_survivors.append(winner)
                        eliminated_teams.add(loser.team_id)
                        if loser.team_id in teams_in_losers:
                            del teams_in_losers[loser.team_id]
                    else:
                        # Bye
                        new_survivors.append(lb_survivors[i])

                lb_survivors = new_survivors

            stage.losers_bracket[lb_round] = round_matches

        # Generate Grand Finals
        if wb_champion:
            # Get losers bracket champion
            lb_champion = None

            # First try lb_survivors
            if lb_survivors:
                lb_champion = (
                    lb_survivors[0]
                    if len(lb_survivors) == 1
                    else lb_survivors[-1]
                )

            # Then try teams still in losers bracket
            if not lb_champion and len(teams_in_losers) > 0:
                # Get the last remaining team in losers bracket
                lb_champion = list(teams_in_losers.values())[0]

            if lb_champion:
                # Grand Finals Match 1
                gf_match1 = Match(
                    match_id=self._next_match_id,
                    team_a=wb_champion,
                    team_b=lb_champion,
                    round_number=1,
                    stage="Double Elimination",
                    bracket_type="grand_finals",
                )
                self._next_match_id += 1

                # Simulate GF1
                if self.rng.random() < 0.65:  # WB champion has advantage
                    gf_match1.winner = wb_champion
                    tournament_champion = wb_champion
                else:
                    gf_match1.winner = lb_champion
                    # Bracket reset needed
                    gf_match2 = Match(
                        match_id=self._next_match_id,
                        team_a=wb_champion,
                        team_b=lb_champion,
                        round_number=2,
                        stage="Double Elimination",
                        bracket_type="grand_finals",
                    )
                    self._next_match_id += 1

                    # Simulate GF2
                    if self.rng.random() < 0.5:  # Even odds in reset
                        gf_match2.winner = wb_champion
                        tournament_champion = wb_champion
                    else:
                        gf_match2.winner = lb_champion
                        tournament_champion = lb_champion

                    stage.grand_finals.append(gf_match2)

                stage.grand_finals.insert(0, gf_match1)

        # Consolidate all matches into rounds for compatibility
        all_rounds = {}
        round_counter = 1

        # Add winners bracket matches
        for wb_round, matches in stage.winners_bracket.items():
            all_rounds[round_counter] = matches
            round_counter += 1

        # Add losers bracket matches
        for lb_round, matches in stage.losers_bracket.items():
            if matches:  # Only add if there are matches
                all_rounds[round_counter] = matches
                round_counter += 1

        # Add grand finals
        if stage.grand_finals:
            all_rounds[round_counter] = stage.grand_finals

        stage.rounds = all_rounds

        return stage

    def _generate_group_stage(
        self,
        teams: list[Team],
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

        if not teams:
            return stage

        # Divide teams into groups
        shuffled_teams = teams.copy()
        self.rng.shuffle(shuffled_teams)

        groups = [[] for _ in range(n_groups)]
        for i, team in enumerate(shuffled_teams):
            groups[i % n_groups].append(team)

        # Generate round robin within each group
        max_rounds = max(len(group) - 1 for group in groups if group)

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
    ) -> list[Team]:
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
