"""
Data validation module for synthetic tournament data.

This module validates generated tournament data to ensure consistency,
completeness, and compatibility with the ranking system.
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import polars as pl

from rankings.core import parse_tournaments_data
from synthetic_data.tournament_generator import Tournament


class DataValidator:
    """Validates synthetic tournament data for consistency and completeness."""
    
    def __init__(self):
        """Initialize the data validator."""
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def validate_tournament(self, tournament: Tournament) -> bool:
        """
        Validate a tournament structure.
        
        Parameters
        ----------
        tournament : Tournament
            Tournament to validate
            
        Returns
        -------
        bool
            True if valid, False otherwise
        """
        self.errors = []
        self.warnings = []
        
        # Basic structure validation
        self._validate_tournament_structure(tournament)
        
        # Team validation
        self._validate_teams(tournament)
        
        # Stage validation
        for stage in tournament.stages:
            self._validate_stage(stage, tournament)
            
        return len(self.errors) == 0
    
    def validate_serialized_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate serialized tournament data.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Serialized tournament data
            
        Returns
        -------
        bool
            True if valid, False otherwise
        """
        self.errors = []
        self.warnings = []
        
        # Check required structure
        if "tournament" not in data:
            self.errors.append("Missing 'tournament' key in data")
            return False
            
        tournament = data["tournament"]
        
        if "data" not in tournament:
            self.errors.append("Missing 'data' section in tournament")
            return False
            
        if "ctx" not in tournament:
            self.errors.append("Missing 'ctx' section in tournament")
            return False
            
        # Validate data sections
        self._validate_data_section(tournament["data"])
        
        # Validate context
        self._validate_ctx_section(tournament["ctx"])
        
        # Cross-reference validation
        self._validate_cross_references(tournament)
        
        return len(self.errors) == 0
    
    def validate_with_parser(self, data: List[Dict[str, Any]]) -> bool:
        """
        Validate data by attempting to parse it.
        
        Parameters
        ----------
        data : List[Dict[str, Any]]
            List of serialized tournament data
            
        Returns
        -------
        bool
            True if parseable, False otherwise
        """
        try:
            tables = parse_tournaments_data(data)
            
            # Check that we got expected tables
            expected_tables = ["stages", "groups", "rounds", "teams", "players", "matches"]
            for table_name in expected_tables:
                if table_name in tables and tables[table_name] is not None:
                    df = tables[table_name]
                    if len(df) == 0:
                        self.warnings.append(f"Table '{table_name}' is empty")
                else:
                    self.warnings.append(f"Table '{table_name}' is None")
                    
            # Validate parsed data integrity
            if tables.get("matches") is not None:
                self._validate_match_integrity(tables["matches"])
                
            if tables.get("players") is not None and tables.get("teams") is not None:
                self._validate_roster_integrity(tables["players"], tables["teams"])
                
            return True
            
        except Exception as e:
            self.errors.append(f"Parser failed: {str(e)}")
            return False
    
    def _validate_tournament_structure(self, tournament: Tournament):
        """Validate basic tournament structure."""
        if not tournament.tournament_id:
            self.errors.append("Tournament missing ID")
            
        if not tournament.name:
            self.errors.append("Tournament missing name")
            
        if not tournament.start_date:
            self.warnings.append("Tournament missing start date")
            
        if not tournament.stages:
            self.errors.append("Tournament has no stages")
            
        if not tournament.all_teams:
            self.errors.append("Tournament has no teams")
    
    def _validate_teams(self, tournament: Tournament):
        """Validate team composition."""
        team_ids = set()
        player_ids = set()
        
        for team in tournament.all_teams:
            # Check for duplicate team IDs
            if team.team_id in team_ids:
                self.errors.append(f"Duplicate team ID: {team.team_id}")
            team_ids.add(team.team_id)
            
            # Validate team has players
            if not team.players:
                self.errors.append(f"Team {team.team_id} has no players")
            
            # Check team size consistency
            if len(team.players) != 4:
                self.warnings.append(
                    f"Team {team.team_id} has {len(team.players)} players (expected 4)"
                )
                
            # Check for duplicate players
            for player in team.players:
                if player.user_id in player_ids:
                    self.errors.append(
                        f"Player {player.user_id} appears on multiple teams"
                    )
                player_ids.add(player.user_id)
    
    def _validate_stage(self, stage, tournament):
        """Validate a tournament stage."""
        if not stage.stage_id:
            self.errors.append("Stage missing ID")
            
        if not stage.name:
            self.errors.append("Stage missing name")
            
        if not stage.rounds:
            self.errors.append(f"Stage {stage.stage_id} has no rounds")
            
        # Validate each round
        for round_num, matches in stage.rounds.items():
            if not matches:
                self.warnings.append(
                    f"Round {round_num} in stage {stage.stage_id} has no matches"
                )
                
            # Check match participation
            teams_in_round = set()
            for match in matches:
                if match.team_a:
                    teams_in_round.add(match.team_a.team_id)
                if match.team_b:
                    teams_in_round.add(match.team_b.team_id)
                    
            # Validate match structure
            for match in matches:
                self._validate_match(match)
    
    def _validate_match(self, match):
        """Validate a single match."""
        if not match.match_id:
            self.errors.append("Match missing ID")
            
        if not match.team_a:
            self.errors.append(f"Match {match.match_id} missing team A")
            
        # Team B can be None for byes
        if match.team_b is None and match.winner:
            if match.winner != match.team_a:
                self.errors.append(
                    f"Match {match.match_id}: bye match but winner is not team A"
                )
                
        # Validate scores if match is completed
        if match.winner:
            if match.score_a is None or match.score_b is None:
                self.warnings.append(
                    f"Match {match.match_id} has winner but missing scores"
                )
            elif match.score_a == match.score_b:
                self.errors.append(
                    f"Match {match.match_id} has tied scores but a winner"
                )
    
    def _validate_data_section(self, data: Dict[str, List]):
        """Validate the data section of serialized tournament."""
        required_keys = ["stage", "group", "round", "match"]
        
        for key in required_keys:
            if key not in data:
                self.errors.append(f"Missing '{key}' in data section")
                
        # Validate stages
        if "stage" in data:
            stage_ids = set()
            for stage in data["stage"]:
                if "id" not in stage:
                    self.errors.append("Stage missing ID")
                elif stage["id"] in stage_ids:
                    self.errors.append(f"Duplicate stage ID: {stage['id']}")
                else:
                    stage_ids.add(stage["id"])
    
    def _validate_ctx_section(self, ctx: Dict[str, Any]):
        """Validate the context section of serialized tournament."""
        if "id" not in ctx:
            self.errors.append("Tournament context missing ID")
            
        if "teams" not in ctx:
            self.errors.append("Tournament context missing teams")
        else:
            # Validate teams
            team_ids = set()
            for team in ctx["teams"]:
                if "id" not in team:
                    self.errors.append("Team missing ID")
                elif team["id"] in team_ids:
                    self.errors.append(f"Duplicate team ID: {team['id']}")
                else:
                    team_ids.add(team["id"])
                    
                # Validate roster
                if "members" not in team:
                    self.errors.append(f"Team {team.get('id', '?')} missing members")
    
    def _validate_cross_references(self, tournament: Dict[str, Any]):
        """Validate cross-references between data sections."""
        data = tournament["data"]
        ctx = tournament["ctx"]
        
        # Collect all IDs
        stage_ids = {s["id"] for s in data.get("stage", []) if "id" in s}
        team_ids = {t["id"] for t in ctx.get("teams", []) if "id" in t}
        
        # Validate matches reference valid teams and stages
        for match in data.get("match", []):
            if "stage_id" in match and match["stage_id"] not in stage_ids:
                self.errors.append(
                    f"Match {match.get('id', '?')} references invalid stage {match['stage_id']}"
                )
                
            # Check team references
            if match.get("opponent1") and "id" in match["opponent1"]:
                if match["opponent1"]["id"] not in team_ids:
                    self.errors.append(
                        f"Match {match.get('id', '?')} references invalid team {match['opponent1']['id']}"
                    )
                    
            if match.get("opponent2") and "id" in match["opponent2"]:
                if match["opponent2"]["id"] not in team_ids:
                    self.errors.append(
                        f"Match {match.get('id', '?')} references invalid team {match['opponent2']['id']}"
                    )
    
    def _validate_match_integrity(self, matches_df: pl.DataFrame):
        """Validate match data integrity in parsed dataframe."""
        # Check for required columns
        required_cols = [
            "match_id", "tournament_id", "team1_id", "team2_id",
            "winner_team_id", "loser_team_id"
        ]
        
        for col in required_cols:
            if col not in matches_df.columns:
                self.errors.append(f"Matches dataframe missing column: {col}")
                return
                
        # Validate winner/loser consistency
        completed_matches = matches_df.filter(
            pl.col("winner_team_id").is_not_null()
        )
        
        for row in completed_matches.iter_rows(named=True):
            winner = row["winner_team_id"]
            loser = row["loser_team_id"]
            team1 = row["team1_id"]
            team2 = row["team2_id"]
            
            # Winner should be one of the teams
            if winner not in [team1, team2]:
                self.errors.append(
                    f"Match {row['match_id']}: winner {winner} not in teams {team1}, {team2}"
                )
                
            # Loser should be the other team
            if loser is not None:
                if winner == team1 and loser != team2:
                    self.errors.append(
                        f"Match {row['match_id']}: inconsistent winner/loser"
                    )
                elif winner == team2 and loser != team1:
                    self.errors.append(
                        f"Match {row['match_id']}: inconsistent winner/loser"
                    )
    
    def _validate_roster_integrity(self, players_df: pl.DataFrame, teams_df: pl.DataFrame):
        """Validate roster integrity across players and teams."""
        # Check all players belong to valid teams
        team_ids = set(teams_df["team_id"].unique())
        player_team_ids = set(players_df["team_id"].unique())
        
        invalid_teams = player_team_ids - team_ids
        if invalid_teams:
            self.errors.append(
                f"Players assigned to non-existent teams: {invalid_teams}"
            )
            
        # Check for reasonable team sizes
        team_sizes = (
            players_df.group_by(["tournament_id", "team_id"])
            .count()
            .rename({"count": "team_size"})
        )
        
        # Flag unusual team sizes
        unusual_sizes = team_sizes.filter(
            (pl.col("team_size") < 3) | (pl.col("team_size") > 5)
        )
        
        if len(unusual_sizes) > 0:
            for row in unusual_sizes.iter_rows(named=True):
                self.warnings.append(
                    f"Team {row['team_id']} in tournament {row['tournament_id']} "
                    f"has unusual size: {row['team_size']}"
                )
    
    def get_validation_report(self) -> str:
        """
        Get a formatted validation report.
        
        Returns
        -------
        str
            Formatted report of errors and warnings
        """
        report = []
        
        if self.errors:
            report.append("ERRORS:")
            for error in self.errors:
                report.append(f"  - {error}")
                
        if self.warnings:
            if report:
                report.append("")
            report.append("WARNINGS:")
            for warning in self.warnings:
                report.append(f"  - {warning}")
                
        if not self.errors and not self.warnings:
            report.append("All validations passed successfully!")
            
        return "\n".join(report)