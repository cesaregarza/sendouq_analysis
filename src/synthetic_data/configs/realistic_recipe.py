"""
Realistic tournament recipe configuration based on analysis of real Sendou.ink data.

This recipe mirrors the characteristics found in ~1800 real tournaments:
- Tournament sizes: 4-32 teams (most common: 6-12)
- Team sizes: 4-5 players (occasionally 1-6)
- Tournament formats: Single/Double elimination dominant, some Round Robin and Swiss
- Score patterns: Best-of-3 and Best-of-5 matches
- Player participation: Mix of casual and frequent players
- Temporal patterns: Multiple tournaments per day
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np


class RealisticTournamentRecipe:
    """Recipe that mirrors real Sendou.ink tournament data characteristics."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

        # Based on real data analysis
        self.team_size_weights = {
            4: 0.40,  # Most common
            5: 0.35,  # Second most common
            6: 0.10,
            2: 0.08,
            1: 0.05,
            3: 0.02,
        }

        self.tournament_size_weights = {
            6: 0.10,
            8: 0.12,
            7: 0.09,
            4: 0.08,
            5: 0.08,
            9: 0.08,
            11: 0.08,
            13: 0.07,
            12: 0.06,
            16: 0.06,
            20: 0.04,
            24: 0.03,
            32: 0.03,
            10: 0.04,
            14: 0.02,
            15: 0.01,
            18: 0.01,
        }

        self.format_weights = {
            "single_elimination": 0.51,
            "double_elimination": 0.27,
            "round_robin": 0.15,
            "swiss": 0.07,
        }

        self.match_format_weights = {
            "bo3": 0.70,  # Best of 3
            "bo5": 0.25,  # Best of 5
            "bo7": 0.05,  # Best of 7
        }

        # Game modes (SZ=Splat Zones, TC=Tower Control, RM=Rainmaker, CB=Clam Blitz)
        self.game_mode_weights = {
            "SZ": 0.53,
            "TC": 0.16,
            "RM": 0.16,
            "CB": 0.15,
        }

        # Player skill distribution parameters (normalized ordinal values)
        self.skill_params = {"mean": 20.0, "std": 8.0, "min": 5.0, "max": 35.0}

        # Player participation patterns
        self.player_frequency_weights = {
            "casual": 0.25,  # 1 tournament
            "occasional": 0.27,  # 2-5 tournaments
            "regular": 0.48,  # 6+ tournaments
        }

    def generate_tournament_schedule(
        self,
        start_date: datetime,
        end_date: datetime,
        avg_tournaments_per_day: float = 3.0,
    ) -> List[datetime]:
        """Generate realistic tournament schedule."""
        tournaments = []
        current = start_date

        while current < end_date:
            # Number of tournaments this day (Poisson distribution)
            n_tournaments = max(1, self.rng.poisson(avg_tournaments_per_day))

            # Distribute tournaments throughout the day
            # Peak hours: 18:00-22:00 local time
            for _ in range(n_tournaments):
                hour = self.rng.choice(
                    [14, 15, 16, 17, 18, 19, 20, 21, 22],
                    p=[0.05, 0.05, 0.08, 0.10, 0.15, 0.17, 0.17, 0.13, 0.10],
                )
                minute = self.rng.choice([0, 30])
                tournament_time = current.replace(
                    hour=hour, minute=minute, second=0, microsecond=0
                )
                tournaments.append(tournament_time)

            current += timedelta(days=1)

        return sorted(tournaments)

    def generate_player_pool(
        self, n_players: int = 1000
    ) -> List[Dict[str, Any]]:
        """Generate a pool of players with realistic characteristics."""
        players = []

        for i in range(n_players):
            # Skill follows truncated normal distribution
            skill = np.clip(
                self.rng.normal(
                    self.skill_params["mean"], self.skill_params["std"]
                ),
                self.skill_params["min"],
                self.skill_params["max"],
            )

            # Participation frequency
            freq_type = self.rng.choice(
                list(self.player_frequency_weights.keys()),
                p=list(self.player_frequency_weights.values()),
            )

            if freq_type == "casual":
                participation_rate = self.rng.uniform(0.01, 0.05)
            elif freq_type == "occasional":
                participation_rate = self.rng.uniform(0.05, 0.20)
            else:  # regular
                participation_rate = self.rng.uniform(0.20, 0.80)

            players.append(
                {
                    "id": i + 1,
                    "username": f"Player_{i+1}",
                    "skill_ordinal": skill,
                    "participation_rate": participation_rate,
                    "frequency_type": freq_type,
                    "country": self.rng.choice(
                        [
                            "US",
                            "JP",
                            "FR",
                            "DE",
                            "UK",
                            "CA",
                            "AU",
                            "NL",
                            "BE",
                            "IT",
                        ]
                    ),
                    "created_at": datetime(2023, 1, 1)
                    + timedelta(days=self.rng.randint(0, 730)),
                }
            )

        return players

    def select_tournament_players(
        self, player_pool: List[Dict], n_teams: int, tournament_date: datetime
    ) -> List[List[Dict]]:
        """Select players for a tournament based on participation rates."""
        teams = []
        available_players = []

        # Filter players who would participate
        for player in player_pool:
            # Check if player existed at tournament date
            if player["created_at"] <= tournament_date:
                # Participation probability
                if self.rng.random() < player["participation_rate"]:
                    available_players.append(player)

        # Need enough players for teams
        team_size = self.rng.choice(
            list(self.team_size_weights.keys()),
            p=list(self.team_size_weights.values()),
        )

        required_players = n_teams * team_size

        if len(available_players) < required_players:
            # Generate additional temporary players if needed
            for i in range(required_players - len(available_players)):
                skill = np.clip(
                    self.rng.normal(
                        self.skill_params["mean"], self.skill_params["std"]
                    ),
                    self.skill_params["min"],
                    self.skill_params["max"],
                )
                available_players.append(
                    {
                        "id": 10000 + i,
                        "username": f"Guest_{i+1}",
                        "skill_ordinal": skill,
                        "participation_rate": 0.01,
                        "frequency_type": "casual",
                        "country": "US",
                        "created_at": tournament_date,
                    }
                )

        # Shuffle and form teams
        self.rng.shuffle(available_players)

        for i in range(n_teams):
            team = available_players[i * team_size : (i + 1) * team_size]
            teams.append(team)

        return teams

    def generate_tournament_config(self) -> Dict[str, Any]:
        """Generate a tournament configuration."""
        n_teams = self.rng.choice(
            list(self.tournament_size_weights.keys()),
            p=list(self.tournament_size_weights.values()),
        )

        format_type = self.rng.choice(
            list(self.format_weights.keys()),
            p=list(self.format_weights.values()),
        )

        match_format = self.rng.choice(
            list(self.match_format_weights.keys()),
            p=list(self.match_format_weights.values()),
        )

        return {
            "n_teams": n_teams,
            "format": format_type,
            "match_format": match_format,
            "game_modes": self._generate_map_pool(),
        }

    def _generate_map_pool(self, n_maps: int = 7) -> List[str]:
        """Generate a map pool for a tournament."""
        modes = []
        for _ in range(n_maps):
            mode = self.rng.choice(
                list(self.game_mode_weights.keys()),
                p=list(self.game_mode_weights.values()),
            )
            modes.append(mode)
        return modes

    def calculate_match_outcome(
        self, team1: List[Dict], team2: List[Dict], match_format: str
    ) -> tuple:
        """Calculate match outcome based on team skills."""
        # Average team skills
        skill1 = np.mean([p["skill_ordinal"] for p in team1])
        skill2 = np.mean([p["skill_ordinal"] for p in team2])

        # Win probability using logistic function
        skill_diff = skill1 - skill2
        win_prob = 1 / (1 + np.exp(-skill_diff / 5))

        # Determine match scores
        if match_format == "bo3":
            max_score = 2
        elif match_format == "bo5":
            max_score = 3
        else:  # bo7
            max_score = 4

        # Simulate individual games
        team1_score = 0
        team2_score = 0

        while team1_score < max_score and team2_score < max_score:
            if self.rng.random() < win_prob:
                team1_score += 1
            else:
                team2_score += 1

        return team1_score, team2_score

    def generate_tournament_data(
        self,
        tournament_date: datetime,
        player_pool: List[Dict],
        config: Dict = None,
    ) -> Dict[str, Any]:
        """Generate complete tournament data."""
        if config is None:
            config = self.generate_tournament_config()

        # Select teams
        teams = self.select_tournament_players(
            player_pool, config["n_teams"], tournament_date
        )

        # Generate tournament structure based on format
        tournament_data = {
            "date": tournament_date.isoformat(),
            "format": config["format"],
            "match_format": config["match_format"],
            "teams": [],
            "matches": [],
        }

        # Add team data
        for i, team in enumerate(teams):
            tournament_data["teams"].append(
                {
                    "id": i + 1,
                    "name": f"Team_{i+1}",
                    "seed": i + 1,
                    "players": [
                        {"id": p["id"], "username": p["username"]} for p in team
                    ],
                    "avg_skill": np.mean([p["skill_ordinal"] for p in team]),
                }
            )

        # Generate matches based on format
        if config["format"] == "single_elimination":
            matches = self._generate_single_elimination_matches(
                teams, config["match_format"]
            )
        elif config["format"] == "double_elimination":
            matches = self._generate_double_elimination_matches(
                teams, config["match_format"]
            )
        elif config["format"] == "round_robin":
            matches = self._generate_round_robin_matches(
                teams, config["match_format"]
            )
        else:  # swiss
            matches = self._generate_swiss_matches(
                teams, config["match_format"]
            )

        tournament_data["matches"] = matches
        return tournament_data

    def _generate_single_elimination_matches(
        self, teams: List, match_format: str
    ) -> List[Dict]:
        """Generate single elimination bracket matches."""
        matches = []
        current_round = list(range(len(teams)))
        round_num = 1

        while len(current_round) > 1:
            next_round = []
            for i in range(0, len(current_round), 2):
                if i + 1 < len(current_round):
                    team1_idx = current_round[i]
                    team2_idx = current_round[i + 1]
                    score1, score2 = self.calculate_match_outcome(
                        teams[team1_idx], teams[team2_idx], match_format
                    )

                    winner_idx = team1_idx if score1 > score2 else team2_idx
                    loser_idx = team2_idx if score1 > score2 else team1_idx
                    next_round.append(winner_idx)

                    matches.append(
                        {
                            "round": round_num,
                            "team1_id": team1_idx + 1,
                            "team2_id": team2_idx + 1,
                            "score1": score1,
                            "score2": score2,
                            "winner_id": winner_idx + 1,
                            "loser_id": loser_idx + 1,
                        }
                    )
                else:
                    # Bye
                    next_round.append(current_round[i])

            current_round = next_round
            round_num += 1

        return matches

    def _generate_double_elimination_matches(
        self, teams: List, match_format: str
    ) -> List[Dict]:
        """Generate double elimination bracket matches."""
        # Simplified version - just add more matches
        matches = self._generate_single_elimination_matches(teams, match_format)

        # Add losers bracket matches (simplified)
        n_additional = len(matches) // 2
        for i in range(n_additional):
            team1_idx = self.rng.randint(0, len(teams))
            team2_idx = self.rng.randint(0, len(teams))
            if team1_idx != team2_idx:
                score1, score2 = self.calculate_match_outcome(
                    teams[team1_idx], teams[team2_idx], match_format
                )
                matches.append(
                    {
                        "round": f"L{i+1}",
                        "team1_id": team1_idx + 1,
                        "team2_id": team2_idx + 1,
                        "score1": score1,
                        "score2": score2,
                        "winner_id": (team1_idx + 1)
                        if score1 > score2
                        else (team2_idx + 1),
                        "loser_id": (team2_idx + 1)
                        if score1 > score2
                        else (team1_idx + 1),
                    }
                )

        return matches

    def _generate_round_robin_matches(
        self, teams: List, match_format: str
    ) -> List[Dict]:
        """Generate round robin matches."""
        matches = []

        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                score1, score2 = self.calculate_match_outcome(
                    teams[i], teams[j], match_format
                )
                matches.append(
                    {
                        "round": "RR",
                        "team1_id": i + 1,
                        "team2_id": j + 1,
                        "score1": score1,
                        "score2": score2,
                        "winner_id": (i + 1) if score1 > score2 else (j + 1),
                        "loser_id": (j + 1) if score1 > score2 else (i + 1),
                    }
                )

        return matches

    def _generate_swiss_matches(
        self, teams: List, match_format: str, rounds: int = 5
    ) -> List[Dict]:
        """Generate Swiss system matches."""
        matches = []
        team_scores = {i: 0 for i in range(len(teams))}

        for round_num in range(1, min(rounds + 1, len(teams))):
            # Pair teams by current scores
            sorted_teams = sorted(
                team_scores.keys(), key=lambda x: team_scores[x], reverse=True
            )

            paired = set()
            for i in range(0, len(sorted_teams), 2):
                if i + 1 < len(sorted_teams):
                    team1_idx = sorted_teams[i]
                    team2_idx = sorted_teams[i + 1]

                    if team1_idx not in paired and team2_idx not in paired:
                        score1, score2 = self.calculate_match_outcome(
                            teams[team1_idx], teams[team2_idx], match_format
                        )

                        if score1 > score2:
                            team_scores[team1_idx] += 1
                        else:
                            team_scores[team2_idx] += 1

                        matches.append(
                            {
                                "round": round_num,
                                "team1_id": team1_idx + 1,
                                "team2_id": team2_idx + 1,
                                "score1": score1,
                                "score2": score2,
                                "winner_id": (team1_idx + 1)
                                if score1 > score2
                                else (team2_idx + 1),
                                "loser_id": (team2_idx + 1)
                                if score1 > score2
                                else (team1_idx + 1),
                            }
                        )

                        paired.add(team1_idx)
                        paired.add(team2_idx)

        return matches


# Example usage function
def create_realistic_tournament_dataset(
    n_tournaments: int = 100,
    start_date: datetime = datetime(2024, 1, 1),
    end_date: datetime = datetime(2024, 12, 31),
    n_players: int = 2000,
    seed: int = 42,
) -> List[Dict]:
    """Create a dataset of realistic tournaments."""
    recipe = RealisticTournamentRecipe(seed=seed)

    # Generate player pool
    player_pool = recipe.generate_player_pool(n_players)

    # Generate tournament schedule
    schedule = recipe.generate_tournament_schedule(
        start_date, end_date, avg_tournaments_per_day=n_tournaments / 365
    )

    # Generate tournaments
    tournaments = []
    for i, tournament_date in enumerate(schedule[:n_tournaments]):
        tournament_data = recipe.generate_tournament_data(
            tournament_date, player_pool
        )
        tournament_data["id"] = i + 1
        tournament_data["name"] = f"Tournament_{i+1}"
        tournaments.append(tournament_data)

    return tournaments


if __name__ == "__main__":
    # Test the recipe
    tournaments = create_realistic_tournament_dataset(
        n_tournaments=10, n_players=500
    )

    print(f"Generated {len(tournaments)} tournaments")
    print(f"First tournament: {tournaments[0]['date']}")
    print(f"  Teams: {len(tournaments[0]['teams'])}")
    print(f"  Format: {tournaments[0]['format']}")
    print(f"  Matches: {len(tournaments[0]['matches'])}")
