"""
Data serialization module for synthetic tournament data.

This module converts generated tournament structures into Sendou.ink JSON format
compatible with the existing parser and analysis tools.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from synthetic_data.match_simulator import MatchSimulator
from synthetic_data.tournament_generator import (
    Match,
    Team,
    Tournament,
    TournamentFormat,
    TournamentStage,
)


class DataSerializer:
    """Serializes synthetic tournament data to Sendou.ink JSON format."""

    def __init__(self):
        """Initialize the data serializer."""
        self._stage_id_counter = 1
        self._group_id_counter = 1
        self._round_id_counter = 1

    def serialize_tournament(
        self,
        tournament: Tournament,
        simulate_matches: bool = True,
        match_simulator: Optional[MatchSimulator] = None,
    ) -> Dict[str, Any]:
        """
        Serialize a tournament to Sendou.ink JSON format.

        Parameters
        ----------
        tournament : Tournament
            Tournament to serialize
        simulate_matches : bool
            Whether to simulate match outcomes
        match_simulator : MatchSimulator, optional
            Match simulator to use (creates default if not provided)

        Returns
        -------
        Dict[str, Any]
            Tournament data in Sendou.ink JSON format
        """
        if simulate_matches and match_simulator is None:
            match_simulator = MatchSimulator()

        # Reset ID counters for this tournament
        self._stage_id_counter = 1
        self._group_id_counter = 1
        self._round_id_counter = 1

        # Build tournament structure
        tournament_data = {
            "tournament": {
                "data": {"stage": [], "group": [], "round": [], "match": []},
                "ctx": {
                    "id": tournament.tournament_id,
                    "name": tournament.name,
                    "startTime": tournament.start_date.isoformat()
                    if tournament.start_date
                    else None,
                    "teams": [],
                },
            }
        }

        # Serialize teams
        for team in tournament.all_teams:
            tournament_data["tournament"]["ctx"]["teams"].append(
                self._serialize_team(team, tournament.tournament_id)
            )

        # Serialize stages
        for stage in tournament.stages:
            self._serialize_stage(
                stage,
                tournament.tournament_id,
                tournament_data["tournament"]["data"],
                simulate_matches,
                match_simulator,
            )

        return tournament_data

    def serialize_tournaments(
        self,
        tournaments: List[Tournament],
        simulate_matches: bool = True,
        match_simulator: Optional[MatchSimulator] = None,
    ) -> List[Dict[str, Any]]:
        """
        Serialize multiple tournaments.

        Parameters
        ----------
        tournaments : List[Tournament]
            Tournaments to serialize
        simulate_matches : bool
            Whether to simulate match outcomes
        match_simulator : MatchSimulator, optional
            Match simulator to use

        Returns
        -------
        List[Dict[str, Any]]
            List of tournament data in Sendou.ink JSON format
        """
        return [
            self.serialize_tournament(t, simulate_matches, match_simulator)
            for t in tournaments
        ]

    def _serialize_team(self, team: Team, tournament_id: int) -> Dict[str, Any]:
        """Serialize a team and its roster."""
        team_data = {
            "id": team.team_id,
            "name": team.name,
            "seed": team.seed,
            "prefersNotToHost": False,
            "noScreen": False,
            "droppedOut": False,
            "inviteCode": f"SYNTH_{team.team_id}",
            "createdAt": datetime.now().isoformat(),
            "members": [],
        }

        # Add team members
        for i, player in enumerate(team.players):
            team_data["members"].append(
                {
                    "userId": player.user_id,
                    "username": player.username,
                    "discordId": f"discord_{player.user_id}",
                    "inGameName": player.username,
                    "country": "XX",  # Synthetic country code
                    "twitch": None,
                    "isOwner": i == 0,  # First player is owner
                    "createdAt": datetime.now().isoformat(),
                }
            )

        return team_data

    def _serialize_stage(
        self,
        stage: TournamentStage,
        tournament_id: int,
        data_section: Dict[str, List],
        simulate_matches: bool,
        match_simulator: Optional[MatchSimulator],
    ):
        """Serialize a tournament stage."""
        # Map format to stage type string
        stage_type_map = {
            TournamentFormat.SWISS: "swiss",
            TournamentFormat.ROUND_ROBIN: "round_robin",
            TournamentFormat.SINGLE_ELIMINATION: "single_elim",
            TournamentFormat.DOUBLE_ELIMINATION: "double_elim",
            TournamentFormat.GROUP_STAGE: "groups",
        }

        stage_data = {
            "id": self._stage_id_counter,
            "name": stage.name,
            "number": self._stage_id_counter,
            "type": stage_type_map.get(stage.format, "round_robin"),
            "settings": {
                "size": len(stage.teams),
                "grandFinalType": "simple",
                "matchesChildCount": 0,
                "groupCount": 4
                if stage.format == TournamentFormat.GROUP_STAGE
                else 0,
                "roundRobinMode": "simple"
                if stage.format == TournamentFormat.ROUND_ROBIN
                else None,
                "seedOrdering": ["natural"],
                "balanceByes": True,
            },
        }

        data_section["stage"].append(stage_data)
        stage_id = self._stage_id_counter
        self._stage_id_counter += 1

        # Handle group stages
        group_id_map = {}
        if stage.format == TournamentFormat.GROUP_STAGE:
            # Create groups based on matches
            groups_seen = set()
            for round_matches in stage.rounds.values():
                for match in round_matches:
                    if match.group and match.group not in groups_seen:
                        groups_seen.add(match.group)
                        group_data = {
                            "id": self._group_id_counter,
                            "stage_id": stage_id,
                            "number": len(groups_seen),
                        }
                        data_section["group"].append(group_data)
                        group_id_map[match.group] = self._group_id_counter
                        self._group_id_counter += 1

        # Serialize rounds and matches
        for round_num, round_matches in sorted(stage.rounds.items()):
            # Create round
            round_data = {
                "id": self._round_id_counter,
                "stage_id": stage_id,
                "group_id": None,  # Will be set for group stages
                "number": round_num,
                "maps": {
                    "count": 1,  # Simplified - always best of 1 map
                    "type": "best_of",
                },
            }
            data_section["round"].append(round_data)
            round_id = self._round_id_counter
            self._round_id_counter += 1

            # Serialize matches
            for match_idx, match in enumerate(round_matches):
                # Simulate match if requested
                if simulate_matches and match_simulator and not match.winner:
                    match = match_simulator.simulate_match(match)

                match_data = self._serialize_match(
                    match,
                    tournament_id,
                    stage_id,
                    round_id,
                    match_idx + 1,
                    group_id_map.get(match.group) if match.group else None,
                )
                data_section["match"].append(match_data)

    def _serialize_match(
        self,
        match: Match,
        tournament_id: int,
        stage_id: int,
        round_id: int,
        match_number: int,
        group_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Serialize a single match."""
        # Determine match status
        if match.winner:
            status = "completed"
        elif match.team_b is None:
            status = "bye"
        else:
            status = "waiting"

        match_data = {
            "id": match.match_id,
            "stage_id": stage_id,
            "group_id": group_id,
            "round_id": round_id,
            "number": match_number,
            "status": status,
            "lastGameFinishedAt": match.timestamp.isoformat()
            if match.timestamp and match.winner
            else None,
            "createdAt": match.timestamp.isoformat()
            if match.timestamp
            else datetime.now().isoformat(),
            "opponent1": None,
            "opponent2": None,
        }

        # Add team 1 data
        if match.team_a:
            match_data["opponent1"] = {
                "id": match.team_a.team_id,
                "position": match.team_a.seed
                if match.team_a.seed
                else match_number * 2 - 1,
                "score": match.score_a,
                "result": "win"
                if match.winner == match.team_a
                else ("loss" if match.winner else None),
            }

        # Add team 2 data (might be None for byes)
        if match.team_b:
            match_data["opponent2"] = {
                "id": match.team_b.team_id,
                "position": match.team_b.seed
                if match.team_b.seed
                else match_number * 2,
                "score": match.score_b,
                "result": "win"
                if match.winner == match.team_b
                else ("loss" if match.winner else None),
            }

        return match_data

    def to_json_file(
        self,
        tournaments: List[Tournament],
        filename: str,
        simulate_matches: bool = True,
        match_simulator: Optional[MatchSimulator] = None,
    ):
        """
        Write tournaments to JSON file.

        Parameters
        ----------
        tournaments : List[Tournament]
            Tournaments to write
        filename : str
            Output filename
        simulate_matches : bool
            Whether to simulate match outcomes
        match_simulator : MatchSimulator, optional
            Match simulator to use
        """
        import json

        data = self.serialize_tournaments(
            tournaments, simulate_matches, match_simulator
        )

        with open(filename, "w") as f:
            json.dump(data, f, indent=2, default=str)
