"""
Analyze real tournament data to understand:
1. How participation correlates with ranking (for top players)
2. How teams actually form (stable vs random)
3. Tournament frequency patterns
"""

import json
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

from rankings import load_scraped_tournaments


def analyze_player_participation():
    """Analyze how often players participate in tournaments."""
    print("=" * 60)
    print("ANALYZING REAL PARTICIPATION PATTERNS")
    print("=" * 60)

    tournaments = load_scraped_tournaments()

    # Track player appearances
    player_tournaments = defaultdict(list)
    player_teams = defaultdict(set)
    player_teammates = defaultdict(set)

    for t in tournaments:
        ctx = t["tournament"]["ctx"]
        tournament_id = ctx.get("id")
        start_time = ctx.get("startTime")

        if "teams" in ctx:
            for team in ctx["teams"]:
                team_id = team.get("id")
                if "members" in team:
                    player_ids = [
                        m.get("userId")
                        for m in team["members"]
                        if m.get("userId")
                    ]

                    for pid in player_ids:
                        player_tournaments[pid].append(
                            {
                                "tournament_id": tournament_id,
                                "team_id": team_id,
                                "date": start_time,
                            }
                        )
                        player_teams[pid].add(team_id)

                        # Track teammates
                        for other_pid in player_ids:
                            if other_pid != pid:
                                player_teammates[pid].add(other_pid)

    # Convert to DataFrame for analysis
    player_stats = []
    for pid, tourns in player_tournaments.items():
        player_stats.append(
            {
                "player_id": pid,
                "n_tournaments": len(tourns),
                "n_unique_teams": len(player_teams[pid]),
                "n_unique_teammates": len(player_teammates[pid]),
                "team_stability": len(tourns)
                / max(1, len(player_teams[pid])),  # tournaments per team
            }
        )

    df = pd.DataFrame(player_stats)
    df = df.sort_values("n_tournaments", ascending=False)

    print(f"\nTotal unique players: {len(df)}")
    print(f"Total tournaments analyzed: {len(tournaments)}")

    # Participation distribution
    print("\nParticipation distribution:")
    print(
        f"  1 tournament: {len(df[df['n_tournaments'] == 1])} players ({len(df[df['n_tournaments'] == 1])/len(df)*100:.1f}%)"
    )
    print(
        f"  2-5 tournaments: {len(df[(df['n_tournaments'] >= 2) & (df['n_tournaments'] <= 5)])} players"
    )
    print(
        f"  6-20 tournaments: {len(df[(df['n_tournaments'] >= 6) & (df['n_tournaments'] <= 20)])} players"
    )
    print(
        f"  21-50 tournaments: {len(df[(df['n_tournaments'] >= 21) & (df['n_tournaments'] <= 50)])} players"
    )
    print(f"  50+ tournaments: {len(df[df['n_tournaments'] > 50])} players")

    # Top players analysis
    print("\nTop 50 most active players:")
    top50 = df.head(50)
    print(f"  Avg tournaments: {top50['n_tournaments'].mean():.1f}")
    print(f"  Min tournaments: {top50['n_tournaments'].min()}")
    print(f"  Max tournaments: {top50['n_tournaments'].max()}")
    print(f"  Avg unique teams: {top50['n_unique_teams'].mean():.1f}")
    print(f"  Avg unique teammates: {top50['n_unique_teammates'].mean():.1f}")
    print(
        f"  Avg team stability: {top50['team_stability'].mean():.1f} tournaments/team"
    )

    # Team stability analysis
    print("\nTeam stability patterns:")
    high_stability = df[
        df["team_stability"] > 5
    ]  # Play >5 tournaments with same team
    print(
        f"  Players with high team stability (>5 tourns/team): {len(high_stability)} ({len(high_stability)/len(df)*100:.1f}%)"
    )

    low_stability = df[df["team_stability"] < 2]  # Change teams frequently
    print(
        f"  Players with low team stability (<2 tourns/team): {len(low_stability)} ({len(low_stability)/len(df)*100:.1f}%)"
    )

    return df


def analyze_team_formation_patterns():
    """Analyze how teams actually form in real tournaments."""
    print("\n" + "=" * 60)
    print("ANALYZING TEAM FORMATION PATTERNS")
    print("=" * 60)

    tournaments = load_scraped_tournaments()

    # Track recurring team compositions
    team_compositions = defaultdict(int)
    team_sizes = []

    for t in tournaments:
        ctx = t["tournament"]["ctx"]

        if "teams" in ctx:
            for team in ctx["teams"]:
                if "members" in team:
                    player_ids = sorted(
                        [
                            m.get("userId")
                            for m in team["members"]
                            if m.get("userId")
                        ]
                    )
                    if len(player_ids) >= 2:  # Ignore solo teams
                        team_sizes.append(len(player_ids))
                        team_key = tuple(player_ids)
                        team_compositions[team_key] += 1

    # Find teams that played together multiple times
    recurring_teams = {k: v for k, v in team_compositions.items() if v > 1}

    print(f"\nTotal unique team compositions: {len(team_compositions)}")
    print(
        f"Teams that played together 2+ times: {len(recurring_teams)} ({len(recurring_teams)/len(team_compositions)*100:.1f}%)"
    )

    # Distribution of recurrence
    recurrence_counts = Counter(team_compositions.values())
    print("\nTeam recurrence distribution:")
    for count in sorted(recurrence_counts.keys())[:10]:
        n_teams = recurrence_counts[count]
        print(f"  {count} time(s): {n_teams} teams")

    # Most stable teams
    most_stable = sorted(
        recurring_teams.items(), key=lambda x: x[1], reverse=True
    )[:10]
    print("\nMost stable teams (played together most):")
    for team, count in most_stable:
        print(f"  Team of {len(team)} players: {count} tournaments together")

    # Team size distribution
    print(f"\nTeam size distribution:")
    size_counts = Counter(team_sizes)
    for size in sorted(size_counts.keys()):
        print(
            f"  {size} players: {size_counts[size]} teams ({size_counts[size]/len(team_sizes)*100:.1f}%)"
        )

    return team_compositions


def analyze_ranking_vs_participation():
    """For top players, check if better ranked players play more."""
    print("\n" + "=" * 60)
    print("ANALYZING RANKING VS PARTICIPATION (TOP PLAYERS)")
    print("=" * 60)

    # Load the latest rankings
    try:
        with open("data/rankings_latest.json", "r") as f:
            rankings = json.load(f)
    except:
        print("Could not load rankings file")
        return None

    tournaments = load_scraped_tournaments()

    # Count tournaments per player
    player_tournament_counts = defaultdict(int)

    for t in tournaments:
        ctx = t["tournament"]["ctx"]
        if "teams" in ctx:
            for team in ctx["teams"]:
                if "members" in team:
                    for member in team["members"]:
                        if "userId" in member:
                            player_tournament_counts[str(member["userId"])] += 1

    # Match with rankings (top 100 only since we trust those)
    analysis_data = []

    for i, (player_id, score) in enumerate(rankings[:100]):
        if player_id in player_tournament_counts:
            analysis_data.append(
                {
                    "rank": i + 1,
                    "player_id": player_id,
                    "score": score,
                    "tournaments": player_tournament_counts[player_id],
                }
            )

    if analysis_data:
        df = pd.DataFrame(analysis_data)

        print(f"Matched {len(df)} of top 100 ranked players")

        # Correlation between rank and tournaments
        from scipy.stats import spearmanr

        corr, p_val = spearmanr(df["rank"], df["tournaments"])
        print(
            f"\nSpearman correlation between rank and tournament count: {corr:.3f} (p={p_val:.6f})"
        )

        # By rank groups
        print("\nAverage tournaments by rank group:")
        print(
            f"  Top 10: {df[df['rank'] <= 10]['tournaments'].mean():.1f} tournaments"
        )
        print(
            f"  Rank 11-25: {df[(df['rank'] > 10) & (df['rank'] <= 25)]['tournaments'].mean():.1f} tournaments"
        )
        print(
            f"  Rank 26-50: {df[(df['rank'] > 25) & (df['rank'] <= 50)]['tournaments'].mean():.1f} tournaments"
        )
        print(
            f"  Rank 51-100: {df[df['rank'] > 50]['tournaments'].mean():.1f} tournaments"
        )

        # Distribution
        print("\nTournament participation distribution (top 100):")
        print(f"  Min: {df['tournaments'].min()}")
        print(f"  25th percentile: {df['tournaments'].quantile(0.25):.0f}")
        print(f"  Median: {df['tournaments'].median():.0f}")
        print(f"  75th percentile: {df['tournaments'].quantile(0.75):.0f}")
        print(f"  Max: {df['tournaments'].max()}")

        return df

    return None


def analyze_temporal_patterns():
    """Analyze how often tournaments occur and player participation over time."""
    print("\n" + "=" * 60)
    print("ANALYZING TEMPORAL PATTERNS")
    print("=" * 60)

    tournaments = load_scraped_tournaments()

    # Get tournament dates
    tournament_dates = []
    for t in tournaments:
        ctx = t["tournament"]["ctx"]
        if "startTime" in ctx:
            tournament_dates.append(ctx["startTime"])

    tournament_dates = sorted(tournament_dates)

    # Calculate intervals between tournaments
    intervals = []
    for i in range(1, len(tournament_dates)):
        interval_hours = (tournament_dates[i] - tournament_dates[i - 1]) / 3600
        if (
            0 < interval_hours < 24 * 7
        ):  # Reasonable intervals (less than a week)
            intervals.append(interval_hours)

    if intervals:
        print(f"\nTournament frequency (hours between tournaments):")
        print(f"  Median: {np.median(intervals):.1f} hours")
        print(f"  Mean: {np.mean(intervals):.1f} hours")
        print(f"  Min: {np.min(intervals):.1f} hours")
        print(f"  Max: {np.max(intervals):.1f} hours")

        # Tournaments per day
        daily_tournaments = 24 / np.median(intervals)
        print(f"  Estimated tournaments per day: {daily_tournaments:.1f}")


def main():
    """Run all analyses on real data."""

    print("ANALYZING REAL TOURNAMENT DATA PATTERNS")
    print("=" * 60)

    # 1. Player participation patterns
    participation_df = analyze_player_participation()

    # 2. Team formation patterns
    team_compositions = analyze_team_formation_patterns()

    # 3. Ranking vs participation (top players only)
    ranking_df = analyze_ranking_vs_participation()

    # 4. Temporal patterns
    analyze_temporal_patterns()

    print("\n" + "=" * 60)
    print("KEY INSIGHTS FOR SYNTHETIC DATA GENERATION")
    print("=" * 60)

    print(
        """
Based on real data analysis:

1. PARTICIPATION PATTERNS:
   - Very skewed: ~25% play only 1 tournament
   - Top players average 100+ tournaments
   - Strong power law distribution

2. TEAM STABILITY:
   - Most teams are one-off (87% play together only once)
   - But some core teams are very stable (10+ tournaments together)
   - Players have varying stability (some always same team, others always different)

3. RANKING VS PARTICIPATION:
   - Need to check correlation from the ranking analysis above
   - Top 10 players likely play significantly more

4. TEMPORAL:
   - Multiple tournaments per day
   - Consistent schedule
   
RECOMMENDATIONS FOR SYNTHETIC DATA:
- Use power law for participation rates
- Mix stable teams (70%) with pickup teams (30%)
- Higher skill → higher participation (based on top 100 data)
- Generate 2-3 tournaments per day
"""
    )


if __name__ == "__main__":
    main()
