"""
Realistic PageRank evaluator based on actual data patterns:
- Lognormal skill distribution
- Power law participation (higher skill = more tournaments)
- Mostly pickup teams (85% new, 15% recurring)
- Minimal skill growth (1-2 players with capped +1.5 growth)
- Bradley-Terry win probability
- Geometric mean for team aggregation
"""

import os
import sys

sys.path.append("src")

import json
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import lognorm, spearmanr

from synthetic_data.configs.realistic_recipe import RealisticTournamentRecipe


class RealisticPageRankEvaluator:
    """PageRank evaluator matching real tournament patterns."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.recipe = RealisticTournamentRecipe(seed)

    def generate_players_realistic(self, n_players: int = 8000) -> List[Dict]:
        """Generate players matching real data patterns.

        Real data shows:
        - 8401 unique players
        - 25% play only 1 tournament
        - Top 50 average 186 tournaments
        - Power law distribution
        - Higher skill correlates with more participation
        """
        players = []

        # Lognormal skill distribution (matches real data)
        log_mean = 2.5
        log_std = 0.5
        skills = lognorm.rvs(
            s=log_std,
            scale=np.exp(log_mean),
            size=n_players,
            random_state=self.rng,
        )
        skills = np.clip(skills, 5, 40)

        # Select 1-2 players for explosive growth
        explosive_growth_players = self.rng.choice(
            n_players, size=self.rng.randint(1, 3), replace=False
        )

        for i in range(n_players):
            initial_skill = skills[i]

            # Growth rates (minimal for most)
            if i in explosive_growth_players:
                growth_rate = (
                    self.rng.uniform(1.0, 1.5) / 180
                )  # Max +1.5 over period
                is_explosive = True
            else:
                growth_rate = self.rng.uniform(-0.1, 0.2) / 180  # Tiny changes
                is_explosive = False

            # CRITICAL: Participation correlates with skill!
            # Calculate skill percentile
            skill_percentile = np.sum(skills <= initial_skill) / len(skills)

            # Power law participation based on skill
            # Use sigmoid-like function for smooth transition
            # Higher skill = much higher participation
            skill_normalized = (initial_skill - 5) / 35  # Normalize to 0-1

            # Base participation rate increases with skill
            base_rate = 1 / (1 + np.exp(-(skill_normalized - 0.3) * 8))

            # Add noise but maintain correlation
            noise = self.rng.normal(0, 0.05)
            participation_rate = np.clip(base_rate + noise, 0.005, 0.95)

            # Ensure very low skill players rarely play
            if initial_skill < 8:
                participation_rate = min(participation_rate, 0.02)
            # Ensure high skill players play a lot
            elif initial_skill > 25:
                participation_rate = max(participation_rate, 0.60)
            # Top tier players play almost every tournament
            if initial_skill > 35:
                participation_rate = max(participation_rate, 0.80)

            # Explosive growth players become more active over time
            if is_explosive:
                base_participation = max(0.20, participation_rate)
                participation_growth = 0.30
            else:
                base_participation = participation_rate
                participation_growth = 0

            players.append(
                {
                    "id": i + 1,
                    "username": f"Player_{i+1}",
                    "initial_skill": initial_skill,
                    "current_skill": initial_skill,
                    "skill_percentile": skill_percentile,
                    "growth_rate": growth_rate,
                    "base_participation": base_participation,
                    "participation_growth": participation_growth,
                    "current_participation": base_participation,
                    "is_explosive_growth": is_explosive,
                    "country": self.rng.choice(["US", "JP", "FR", "DE", "UK"]),
                    "created_at": datetime(2024, 1, 1)
                    + timedelta(days=self.rng.randint(0, 30)),
                    "tournaments_played": 0,
                    "matches_won": 0,
                    "matches_lost": 0,
                    "unique_teammates": set(),
                    "unique_teams": 0,
                }
            )

        # Sort by skill for easier analysis
        players.sort(key=lambda p: p["initial_skill"], reverse=True)

        # Report skill-participation correlation
        skills_list = [p["initial_skill"] for p in players]
        participation_list = [p["base_participation"] for p in players]
        corr, _ = spearmanr(skills_list, participation_list)
        print(f"  Skill-participation correlation: {corr:.3f}")

        return players

    def update_player_state(
        self,
        player: Dict,
        current_date: datetime,
        base_date: datetime = datetime(2024, 1, 1),
    ):
        """Update player skill and participation."""
        days_elapsed = (current_date - base_date).days

        # Update skill
        skill_change = player["growth_rate"] * days_elapsed
        if player["is_explosive_growth"]:
            skill_change = min(skill_change, 1.5)  # Cap at +1.5
            # Increase participation as they improve
            progress = min(skill_change / 1.5, 1.0)
            player["current_participation"] = (
                player["base_participation"]
                + progress * player["participation_growth"]
            )

        player["current_skill"] = np.clip(
            player["initial_skill"] + skill_change, 3.0, 40.0
        )

    def form_pickup_teams(
        self, available_players: List[Dict], n_teams: int, team_size: int
    ) -> List[List[Dict]]:
        """Form teams realistically - mostly pickup teams with some recurring.

        Real data shows:
        - 86% of teams are one-off
        - 14% play together 2+ times
        - Very few stable teams
        """
        if len(available_players) < n_teams * team_size:
            return None

        teams = []
        used_players = set()

        # Sort by skill for seeding
        available_players.sort(key=lambda p: p["current_skill"], reverse=True)

        # 85% pickup teams (skill-based but random teammates)
        # 15% recurring teams (players who played together before)

        for team_idx in range(n_teams):
            if (
                len(
                    [
                        p
                        for p in available_players
                        if p["id"] not in used_players
                    ]
                )
                < team_size
            ):
                break

            team = []

            # Decide team formation strategy
            use_recurring = self.rng.random() < 0.15 and team_idx > 0

            if use_recurring and teams:
                # Try to recreate a previous team (with some changes)
                prev_team_idx = self.rng.randint(0, len(teams))
                prev_team = teams[prev_team_idx]
                available_prev = [
                    p
                    for p in prev_team
                    if p["id"] not in used_players and p in available_players
                ]

                if (
                    len(available_prev) >= team_size // 2
                ):  # At least half the team
                    team.extend(available_prev[: team_size // 2])
                    for p in team:
                        used_players.add(p["id"])

            # Fill remaining spots with skill-based selection
            remaining_needed = team_size - len(team)
            if remaining_needed > 0:
                # Pick anchor player based on skill tier
                available_unused = [
                    p for p in available_players if p["id"] not in used_players
                ]

                if not available_unused:
                    break

                # Create skill-based team
                # Teams tend to have similar skill players
                if not team:
                    # New team - pick anchor
                    tier_size = len(available_unused) // n_teams
                    tier_start = team_idx * tier_size
                    tier_end = min(
                        tier_start + tier_size * 2, len(available_unused)
                    )

                    tier_players = available_unused[tier_start:tier_end]
                    if tier_players:
                        anchor = self.rng.choice(tier_players)
                        team.append(anchor)
                        used_players.add(anchor["id"])
                        remaining_needed -= 1

                if remaining_needed > 0 and team:
                    # Find players near anchor skill
                    anchor_skill = team[0]["current_skill"]
                    available_unused = [
                        p
                        for p in available_players
                        if p["id"] not in used_players
                    ]

                    # Sort by distance from anchor skill
                    available_unused.sort(
                        key=lambda p: abs(p["current_skill"] - anchor_skill)
                    )

                    # Take closest players with some randomness
                    pool_size = min(len(available_unused), remaining_needed * 3)
                    pool = available_unused[:pool_size]

                    if pool:
                        self.rng.shuffle(pool)
                        for p in pool[:remaining_needed]:
                            team.append(p)
                            used_players.add(p["id"])

            if len(team) == team_size:
                # Track unique teammates for each player
                for p1 in team:
                    for p2 in team:
                        if p1["id"] != p2["id"]:
                            p1["unique_teammates"].add(p2["id"])
                    p1["unique_teams"] += 1

                teams.append(team)

        if len(teams) < n_teams:
            return None

        return teams[:n_teams]

    def calculate_team_skill_geometric(self, team: List[Dict]) -> float:
        """Geometric mean - weakest player has bigger impact."""
        skills = [p["current_skill"] for p in team]
        return np.exp(np.mean(np.log(skills)))

    def bradley_terry_win_probability(
        self, team1_skill: float, team2_skill: float
    ) -> float:
        """Bradley-Terry model for win probability."""
        exp_skill1 = np.exp(team1_skill / 10)
        exp_skill2 = np.exp(team2_skill / 10)

        win_prob = exp_skill1 / (exp_skill1 + exp_skill2)

        # Small upset factor
        upset = self.rng.normal(0, 0.01)
        return np.clip(win_prob + upset, 0.01, 0.99)

    def generate_tournament_realistic(
        self, tournament_date: datetime, players: List[Dict], tournament_id: int
    ) -> Dict:
        """Generate tournament with realistic patterns."""

        # Update player states
        for player in players:
            self.update_player_state(player, tournament_date)

        # Get configuration
        config = self.recipe.generate_tournament_config()

        # Select participating players based on participation rates
        available_players = []
        for player in players:
            if player["created_at"] <= tournament_date:
                if self.rng.random() < player["current_participation"]:
                    available_players.append(player)

        # Form mostly pickup teams
        team_size = self.rng.choice(
            [4, 5], p=[0.6, 0.4]
        )  # Match real distribution
        team_list = self.form_pickup_teams(
            available_players, config["n_teams"], team_size
        )

        if team_list is None:
            return None

        # Create team objects
        teams = []
        for i, team_players in enumerate(team_list):
            team_skill = self.calculate_team_skill_geometric(team_players)
            teams.append(
                {
                    "id": i + 1,
                    "name": f"Team_{i+1}",
                    "players": team_players,
                    "skill": team_skill,
                    "player_ids": [p["id"] for p in team_players],
                }
            )

        # Seed by skill
        teams = sorted(teams, key=lambda t: t["skill"], reverse=True)
        for i, team in enumerate(teams):
            team["seed"] = i + 1

        # Generate matches
        tournament_data = {
            "id": tournament_id,
            "date": tournament_date.isoformat(),
            "format": config["format"],
            "match_format": config["match_format"],
            "teams": teams,
            "matches": [],
        }

        # Generate matches based on format
        if config["format"] == "single_elimination":
            matches = self._generate_single_elim_matches(
                teams, config["match_format"]
            )
        elif config["format"] == "double_elimination":
            matches = self._generate_double_elim_matches(
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

        # Update player statistics
        for match in matches:
            winner_team = next(
                t for t in teams if t["id"] == match["winner_id"]
            )
            loser_team = next(
                (t for t in teams if t["id"] == match.get("loser_id")), None
            )

            for player in winner_team["players"]:
                player["matches_won"] += 1
                player["tournaments_played"] += 1

            if loser_team:
                for player in loser_team["players"]:
                    player["matches_lost"] += 1
                    player["tournaments_played"] += 1

        return tournament_data

    def simulate_match_set(
        self, team1: Dict, team2: Dict, match_format: str
    ) -> Tuple[int, int, int, int]:
        """Simulate a match set using Bradley-Terry."""
        win_prob = self.bradley_terry_win_probability(
            team1["skill"], team2["skill"]
        )

        if match_format == "bo3":
            max_wins = 2
        elif match_format == "bo5":
            max_wins = 3
        else:  # bo7
            max_wins = 4

        team1_wins = 0
        team2_wins = 0

        while team1_wins < max_wins and team2_wins < max_wins:
            if self.rng.random() < win_prob:
                team1_wins += 1
            else:
                team2_wins += 1

        winner_id = team1["id"] if team1_wins > team2_wins else team2["id"]
        loser_id = team2["id"] if team1_wins > team2_wins else team1["id"]

        return team1_wins, team2_wins, winner_id, loser_id

    def _generate_single_elim_matches(
        self, teams: List[Dict], match_format: str
    ) -> List[Dict]:
        """Single elimination with proper seeding."""
        matches = []
        n_teams = len(teams)

        # Create bracket pairings
        round_matches = []
        for i in range(n_teams // 2):
            high_seed_idx = i
            low_seed_idx = n_teams - 1 - i
            round_matches.append((high_seed_idx, low_seed_idx))

        if n_teams % 2 == 1:
            round_matches.append((n_teams - 1, None))

        match_id = 1
        round_num = 1

        while len(round_matches) > 0:
            next_round = []

            for team1_idx, team2_idx in round_matches:
                if team2_idx is None:
                    next_round.append(team1_idx)
                else:
                    (
                        score1,
                        score2,
                        winner_id,
                        loser_id,
                    ) = self.simulate_match_set(
                        teams[team1_idx], teams[team2_idx], match_format
                    )

                    winner_idx = (
                        team1_idx
                        if winner_id == teams[team1_idx]["id"]
                        else team2_idx
                    )
                    next_round.append(winner_idx)

                    matches.append(
                        {
                            "id": match_id,
                            "round": round_num,
                            "team1_id": teams[team1_idx]["id"],
                            "team2_id": teams[team2_idx]["id"],
                            "score1": score1,
                            "score2": score2,
                            "winner_id": winner_id,
                            "loser_id": loser_id,
                        }
                    )
                    match_id += 1

            if len(next_round) > 1:
                round_matches = []
                for i in range(0, len(next_round), 2):
                    if i + 1 < len(next_round):
                        round_matches.append((next_round[i], next_round[i + 1]))
                    else:
                        round_matches.append((next_round[i], None))
                round_num += 1
            else:
                break

        return matches

    def _generate_double_elim_matches(
        self, teams: List[Dict], match_format: str
    ) -> List[Dict]:
        """Double elimination bracket."""
        matches = self._generate_single_elim_matches(teams, match_format)

        # Add losers bracket
        match_id = len(matches) + 1
        n_loser_matches = min(len(teams) // 2, 5)

        for _ in range(n_loser_matches):
            # Pick mid-tier teams for losers bracket
            mid_start = len(teams) // 4
            mid_end = 3 * len(teams) // 4

            if mid_end > mid_start + 1:
                idx1 = self.rng.randint(mid_start, mid_end)
                idx2 = self.rng.randint(mid_start, mid_end)

                if idx1 != idx2:
                    (
                        score1,
                        score2,
                        winner_id,
                        loser_id,
                    ) = self.simulate_match_set(
                        teams[idx1], teams[idx2], match_format
                    )

                    matches.append(
                        {
                            "id": match_id,
                            "round": f"L{match_id - len(matches) + n_loser_matches}",
                            "team1_id": teams[idx1]["id"],
                            "team2_id": teams[idx2]["id"],
                            "score1": score1,
                            "score2": score2,
                            "winner_id": winner_id,
                            "loser_id": loser_id,
                        }
                    )
                    match_id += 1

        return matches

    def _generate_round_robin_matches(
        self, teams: List[Dict], match_format: str
    ) -> List[Dict]:
        """Round robin - everyone plays everyone."""
        matches = []
        match_id = 1

        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                score1, score2, winner_id, loser_id = self.simulate_match_set(
                    teams[i], teams[j], match_format
                )

                matches.append(
                    {
                        "id": match_id,
                        "round": "RR",
                        "team1_id": teams[i]["id"],
                        "team2_id": teams[j]["id"],
                        "score1": score1,
                        "score2": score2,
                        "winner_id": winner_id,
                        "loser_id": loser_id,
                    }
                )
                match_id += 1

        return matches

    def _generate_swiss_matches(
        self, teams: List[Dict], match_format: str, rounds: int = 5
    ) -> List[Dict]:
        """Swiss system."""
        matches = []
        match_id = 1
        team_scores = {team["id"]: 0 for team in teams}

        for round_num in range(1, min(rounds + 1, len(teams))):
            sorted_teams = sorted(
                teams,
                key=lambda t: (team_scores[t["id"]], t["skill"]),
                reverse=True,
            )

            paired = set()
            for i in range(0, len(sorted_teams), 2):
                if i + 1 < len(sorted_teams):
                    team1 = sorted_teams[i]
                    team2 = sorted_teams[i + 1]

                    if team1["id"] not in paired and team2["id"] not in paired:
                        (
                            score1,
                            score2,
                            winner_id,
                            loser_id,
                        ) = self.simulate_match_set(team1, team2, match_format)

                        team_scores[winner_id] += 1

                        matches.append(
                            {
                                "id": match_id,
                                "round": round_num,
                                "team1_id": team1["id"],
                                "team2_id": team2["id"],
                                "score1": score1,
                                "score2": score2,
                                "winner_id": winner_id,
                                "loser_id": loser_id,
                            }
                        )
                        match_id += 1

                        paired.add(team1["id"])
                        paired.add(team2["id"])

        return matches

    def build_pagerank_graph(self, tournaments: List[Dict]) -> nx.DiGraph:
        """Build directed graph from tournament results."""
        G = nx.DiGraph()

        player_wins = defaultdict(lambda: defaultdict(int))

        for tournament in tournaments:
            if tournament is None:
                continue

            for match in tournament["matches"]:
                winner_id = match["winner_id"]
                loser_id = match.get("loser_id")

                if not loser_id:
                    continue

                winner_team = next(
                    (t for t in tournament["teams"] if t["id"] == winner_id),
                    None,
                )
                loser_team = next(
                    (t for t in tournament["teams"] if t["id"] == loser_id),
                    None,
                )

                if winner_team and loser_team:
                    # Edges from losers to winners (match outcome only, not games)
                    for loser_player in loser_team["players"]:
                        for winner_player in winner_team["players"]:
                            player_wins[winner_player["id"]][
                                loser_player["id"]
                            ] += 1

        # Build graph
        for winner_id, losers in player_wins.items():
            for loser_id, count in losers.items():
                G.add_edge(loser_id, winner_id, weight=count)

        return G

    def calculate_pagerank(
        self, G: nx.DiGraph, alpha: float = 0.85
    ) -> Dict[int, float]:
        """Calculate PageRank scores."""
        if len(G) == 0:
            return {}
        return nx.pagerank(G, alpha=alpha, weight="weight")

    def evaluate_correlation(
        self,
        n_tournaments: int = 600,  # ~3.4 per day for 180 days
        n_players: int = 8000,  # Match real data
        evaluation_period_days: int = 180,
    ) -> Dict:
        """Evaluate correlation with realistic parameters."""

        print(f"Generating {n_players} players with realistic patterns...")
        players = self.generate_players_realistic(n_players)

        # Report explosive growth players
        explosive_players = [p for p in players if p["is_explosive_growth"]]
        print(f"  Explosive growth players: {len(explosive_players)}")

        # Generate tournaments (3.4 per day like real data)
        start_date = datetime(2024, 1, 1)
        end_date = start_date + timedelta(days=evaluation_period_days)

        tournaments_per_day = 3.4
        schedule = self.recipe.generate_tournament_schedule(
            start_date, end_date, avg_tournaments_per_day=tournaments_per_day
        )

        print(f"Generating {min(n_tournaments, len(schedule))} tournaments...")
        tournaments = []
        for i, tournament_date in enumerate(schedule[:n_tournaments]):
            if i % 50 == 0:
                print(f"  Generated {i}/{n_tournaments} tournaments...")

            tournament = self.generate_tournament_realistic(
                tournament_date, players, i + 1
            )
            if tournament:
                tournaments.append(tournament)

        print(f"Successfully generated {len(tournaments)} tournaments")

        # Build PageRank graph
        print("Building PageRank graph...")
        G = self.build_pagerank_graph(tournaments)
        print(f"Graph has {len(G.nodes)} nodes and {len(G.edges)} edges")

        # Calculate PageRank
        print("Calculating PageRank...")
        pagerank_scores = self.calculate_pagerank(G)

        # Analyze results
        participating_players = {}
        for player in players:
            if player["id"] in pagerank_scores:
                participating_players[player["id"]] = player

        if len(participating_players) > 1:
            player_ids = list(participating_players.keys())
            true_skills = [
                participating_players[pid]["current_skill"]
                for pid in player_ids
            ]
            pr_scores = [pagerank_scores[pid] for pid in player_ids]
            tournaments_played = [
                participating_players[pid]["tournaments_played"]
                for pid in player_ids
            ]

            # Calculate correlations
            skill_pr_corr, skill_pr_p = spearmanr(true_skills, pr_scores)
            skill_tourn_corr, _ = spearmanr(true_skills, tournaments_played)
            tourn_pr_corr, _ = spearmanr(tournaments_played, pr_scores)

            # Create DataFrame
            df = pd.DataFrame(
                {
                    "player_id": player_ids,
                    "true_skill": true_skills,
                    "pagerank": pr_scores,
                    "tournaments": tournaments_played,
                    "is_explosive": [
                        participating_players[pid]["is_explosive_growth"]
                        for pid in player_ids
                    ],
                }
            )

            df["true_rank"] = df["true_skill"].rank(
                ascending=False, method="min"
            )
            df["pr_rank"] = df["pagerank"].rank(ascending=False, method="min")

            # Participation distribution check
            participation_dist = {
                "1_tournament": len(df[df["tournaments"] == 1]),
                "2-5_tournaments": len(
                    df[(df["tournaments"] >= 2) & (df["tournaments"] <= 5)]
                ),
                "6-20_tournaments": len(
                    df[(df["tournaments"] >= 6) & (df["tournaments"] <= 20)]
                ),
                "21-50_tournaments": len(
                    df[(df["tournaments"] >= 21) & (df["tournaments"] <= 50)]
                ),
                "50+_tournaments": len(df[df["tournaments"] > 50]),
            }

            results = {
                "n_tournaments": len(tournaments),
                "n_players": n_players,
                "n_participating_players": len(participating_players),
                "correlations": {
                    "skill_vs_pagerank": float(skill_pr_corr),
                    "skill_vs_tournaments": float(skill_tourn_corr),
                    "tournaments_vs_pagerank": float(tourn_pr_corr),
                },
                "p_value": float(skill_pr_p),
                "graph_nodes": len(G.nodes),
                "graph_edges": len(G.edges),
                "participation_distribution": participation_dist,
                "top_100_comparison": [],
            }

            # Top 100 analysis (like real data)
            df_top100_skill = df.nlargest(100, "true_skill")
            df_top100_pr = df.nlargest(100, "pagerank")

            results["top_100_stats"] = {
                "overlap": len(
                    set(df_top100_skill["player_id"])
                    & set(df_top100_pr["player_id"])
                ),
                "avg_tournaments_by_true_rank": {
                    "top_10": float(
                        df[df["true_rank"] <= 10]["tournaments"].mean()
                    ),
                    "rank_11-25": float(
                        df[(df["true_rank"] > 10) & (df["true_rank"] <= 25)][
                            "tournaments"
                        ].mean()
                    ),
                    "rank_26-50": float(
                        df[(df["true_rank"] > 25) & (df["true_rank"] <= 50)][
                            "tournaments"
                        ].mean()
                    ),
                    "rank_51-100": float(
                        df[(df["true_rank"] > 50) & (df["true_rank"] <= 100)][
                            "tournaments"
                        ].mean()
                    ),
                },
            }

            # Top 10 comparison
            df_sorted = df.sort_values("true_skill", ascending=False)
            for _, row in df_sorted.head(10).iterrows():
                player = participating_players[row["player_id"]]
                results["top_100_comparison"].append(
                    {
                        "player_id": int(row["player_id"]),
                        "username": player["username"],
                        "true_skill": float(row["true_skill"]),
                        "true_rank": int(row["true_rank"]),
                        "pagerank_rank": int(row["pr_rank"]),
                        "rank_difference": int(
                            row["pr_rank"] - row["true_rank"]
                        ),
                        "tournaments": int(row["tournaments"]),
                        "unique_teammates": len(player["unique_teammates"]),
                        "is_explosive": bool(row["is_explosive"]),
                    }
                )

            return results
        else:
            return {"error": "Not enough participating players"}


def main():
    """Run realistic evaluation."""
    evaluator = RealisticPageRankEvaluator(seed=42)

    print("=" * 60)
    print("REALISTIC PAGERANK EVALUATION")
    print("Matching real data patterns:")
    print("- Power law participation")
    print("- Skill correlates with participation")
    print("- Mostly pickup teams (86% new)")
    print("- 8000 players, 600 tournaments")
    print("=" * 60)

    results = evaluator.evaluate_correlation(
        n_tournaments=600, n_players=8000, evaluation_period_days=180
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if "correlations" in results:
        print(f"\nCorrelations:")
        print(
            f"  Skill vs PageRank: {results['correlations']['skill_vs_pagerank']:.3f}"
        )
        print(
            f"  Skill vs Tournaments: {results['correlations']['skill_vs_tournaments']:.3f}"
        )
        print(
            f"  Tournaments vs PageRank: {results['correlations']['tournaments_vs_pagerank']:.3f}"
        )

        print(f"\nParticipation distribution:")
        for key, count in results["participation_distribution"].items():
            print(f"  {key}: {count} players")

        print(
            f"\nTop 100 overlap: {results['top_100_stats']['overlap']} players"
        )

        print(f"\nAvg tournaments by rank:")
        for rank_group, avg in results["top_100_stats"][
            "avg_tournaments_by_true_rank"
        ].items():
            print(f"  {rank_group}: {avg:.1f} tournaments")

        print(f"\nTop 10 players:")
        print(
            f"{'Username':<15} {'Skill':>8} {'True':>6} {'PR':>6} {'Diff':>6} {'Tourn':>6} {'Team':>6}"
        )
        print("-" * 65)
        for player in results["top_100_comparison"]:
            print(
                f"{player['username']:<15} {player['true_skill']:>8.1f} "
                f"{player['true_rank']:>6} {player['pagerank_rank']:>6} "
                f"{player['rank_difference']:>+6} {player['tournaments']:>6} "
                f"{player['unique_teammates']:>6}"
            )

    # Save results
    with open(
        "src/synthetic_data/evaluation/realistic_final_results.json", "w"
    ) as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("Results saved to realistic_final_results.json")

    # Compare to real data target
    print("\n" + "=" * 60)
    print("COMPARISON TO REAL DATA")
    print("=" * 60)
    print("Real data correlation (top 100): -0.27")
    print("Real data avg tournaments:")
    print("  Top 10: 181")
    print("  Rank 51-100: 106")

    return results


if __name__ == "__main__":
    main()
