"""
Validator to compare synthetic tournament data with real data characteristics.
"""

import json
from collections import Counter
from datetime import datetime

import numpy as np

from rankings import load_scraped_tournaments
from synthetic_data.configs.realistic_recipe import (
    RealisticTournamentRecipe,
    create_realistic_tournament_dataset,
)


def analyze_real_data():
    """Analyze real tournament data characteristics."""
    tournaments = load_scraped_tournaments()

    stats = {
        "n_tournaments": len(tournaments),
        "team_counts": [],
        "match_counts": [],
        "formats": [],
        "scores": [],
        "team_sizes": [],
        "unique_players": set(),
    }

    for t in tournaments:
        ctx = t["tournament"]["ctx"]
        data = t["tournament"]["data"]

        # Teams
        if "teams" in ctx:
            stats["team_counts"].append(len(ctx["teams"]))

            for team in ctx["teams"]:
                if "members" in team:
                    stats["team_sizes"].append(len(team["members"]))
                    for member in team["members"]:
                        if "userId" in member:
                            stats["unique_players"].add(member["userId"])

        # Matches and scores
        if "match" in data:
            stats["match_counts"].append(len(data["match"]))

            for match in data["match"]:
                if match.get("opponent1") and match.get("opponent2"):
                    score1 = match["opponent1"].get("score")
                    score2 = match["opponent2"].get("score")
                    if score1 is not None:
                        stats["scores"].append(score1)
                    if score2 is not None:
                        stats["scores"].append(score2)

        # Formats
        if "settings" in ctx and "bracketProgression" in ctx["settings"]:
            for bracket in ctx["settings"]["bracketProgression"]:
                if "type" in bracket:
                    stats["formats"].append(bracket["type"])

    return stats


def analyze_synthetic_data(n_tournaments=100, n_players=1000):
    """Analyze synthetic tournament data characteristics."""
    tournaments = create_realistic_tournament_dataset(
        n_tournaments=n_tournaments,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        n_players=n_players,
        seed=42,
    )

    stats = {
        "n_tournaments": len(tournaments),
        "team_counts": [],
        "match_counts": [],
        "formats": [],
        "scores": [],
        "team_sizes": [],
        "unique_players": set(),
    }

    for t in tournaments:
        stats["team_counts"].append(len(t["teams"]))
        stats["match_counts"].append(len(t["matches"]))
        stats["formats"].append(t["format"])

        for team in t["teams"]:
            stats["team_sizes"].append(len(team["players"]))
            for player in team["players"]:
                stats["unique_players"].add(player["id"])

        for match in t["matches"]:
            stats["scores"].append(match["score1"])
            stats["scores"].append(match["score2"])

    return stats


def compare_statistics(real_stats, synthetic_stats):
    """Compare real and synthetic data statistics."""
    comparison = {}

    # Team counts
    real_teams = np.array(real_stats["team_counts"])
    synth_teams = np.array(synthetic_stats["team_counts"])
    comparison["team_counts"] = {
        "real": {
            "mean": float(real_teams.mean()),
            "median": float(np.median(real_teams)),
            "std": float(real_teams.std()),
            "min": int(real_teams.min()),
            "max": int(real_teams.max()),
        },
        "synthetic": {
            "mean": float(synth_teams.mean()),
            "median": float(np.median(synth_teams)),
            "std": float(synth_teams.std()),
            "min": int(synth_teams.min()),
            "max": int(synth_teams.max()),
        },
    }

    # Match counts
    real_matches = np.array(real_stats["match_counts"])
    synth_matches = np.array(synthetic_stats["match_counts"])
    comparison["match_counts"] = {
        "real": {
            "mean": float(real_matches.mean()),
            "median": float(np.median(real_matches)),
            "std": float(real_matches.std()),
            "min": int(real_matches.min()),
            "max": int(real_matches.max()),
        },
        "synthetic": {
            "mean": float(synth_matches.mean()),
            "median": float(np.median(synth_matches)),
            "std": float(synth_matches.std()),
            "min": int(synth_matches.min()),
            "max": int(synth_matches.max()),
        },
    }

    # Formats distribution
    real_format_counts = Counter(real_stats["formats"])
    synth_format_counts = Counter(synthetic_stats["formats"])

    comparison["formats"] = {
        "real": {
            k: v / len(real_stats["formats"])
            for k, v in real_format_counts.most_common()
        },
        "synthetic": {
            k: v / len(synthetic_stats["formats"])
            for k, v in synth_format_counts.most_common()
        },
    }

    # Score distribution
    real_score_counts = Counter(real_stats["scores"])
    synth_score_counts = Counter(synthetic_stats["scores"])

    comparison["scores"] = {
        "real": {
            k: v / len(real_stats["scores"])
            for k, v in real_score_counts.most_common(10)
        },
        "synthetic": {
            k: v / len(synthetic_stats["scores"])
            for k, v in synth_score_counts.most_common(10)
        },
    }

    # Team sizes
    real_team_sizes = Counter(real_stats["team_sizes"])
    synth_team_sizes = Counter(synthetic_stats["team_sizes"])

    comparison["team_sizes"] = {
        "real": {
            k: v / len(real_stats["team_sizes"])
            for k, v in real_team_sizes.most_common()
        },
        "synthetic": {
            k: v / len(synthetic_stats["team_sizes"])
            for k, v in synth_team_sizes.most_common()
        },
    }

    # Player counts
    comparison["unique_players"] = {
        "real": len(real_stats["unique_players"]),
        "synthetic": len(synthetic_stats["unique_players"]),
    }

    return comparison


def validate_recipe(n_synthetic_tournaments=100, n_synthetic_players=1000):
    """Validate the recipe by comparing with real data."""
    print("Loading and analyzing real tournament data...")
    real_stats = analyze_real_data()
    print(f"Analyzed {real_stats['n_tournaments']} real tournaments")

    print(
        f"\nGenerating and analyzing {n_synthetic_tournaments} synthetic tournaments..."
    )
    synthetic_stats = analyze_synthetic_data(
        n_synthetic_tournaments, n_synthetic_players
    )
    print(f"Generated {synthetic_stats['n_tournaments']} synthetic tournaments")

    print("\nComparing statistics...")
    comparison = compare_statistics(real_stats, synthetic_stats)

    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    print("\n1. TEAM COUNTS PER TOURNAMENT:")
    print(
        f"   Real:      mean={comparison['team_counts']['real']['mean']:.1f}, "
        f"median={comparison['team_counts']['real']['median']:.1f}, "
        f"range=[{comparison['team_counts']['real']['min']}, {comparison['team_counts']['real']['max']}]"
    )
    print(
        f"   Synthetic: mean={comparison['team_counts']['synthetic']['mean']:.1f}, "
        f"median={comparison['team_counts']['synthetic']['median']:.1f}, "
        f"range=[{comparison['team_counts']['synthetic']['min']}, {comparison['team_counts']['synthetic']['max']}]"
    )

    print("\n2. MATCHES PER TOURNAMENT:")
    print(
        f"   Real:      mean={comparison['match_counts']['real']['mean']:.1f}, "
        f"median={comparison['match_counts']['real']['median']:.1f}"
    )
    print(
        f"   Synthetic: mean={comparison['match_counts']['synthetic']['mean']:.1f}, "
        f"median={comparison['match_counts']['synthetic']['median']:.1f}"
    )

    print("\n3. TOURNAMENT FORMATS:")
    print("   Real:")
    for fmt, pct in comparison["formats"]["real"].items():
        print(f"      {fmt}: {pct*100:.1f}%")
    print("   Synthetic:")
    for fmt, pct in comparison["formats"]["synthetic"].items():
        print(f"      {fmt}: {pct*100:.1f}%")

    print("\n4. TEAM SIZES:")
    print("   Real:")
    for size, pct in list(comparison["team_sizes"]["real"].items())[:5]:
        print(f"      {size} players: {pct*100:.1f}%")
    print("   Synthetic:")
    for size, pct in list(comparison["team_sizes"]["synthetic"].items())[:5]:
        print(f"      {size} players: {pct*100:.1f}%")

    print("\n5. SCORE DISTRIBUTION:")
    print("   Real:")
    for score, pct in list(comparison["scores"]["real"].items())[:5]:
        print(f"      Score {score}: {pct*100:.1f}%")
    print("   Synthetic:")
    for score, pct in list(comparison["scores"]["synthetic"].items())[:5]:
        print(f"      Score {score}: {pct*100:.1f}%")

    print("\n6. PLAYER POOL:")
    print(
        f"   Real:      {comparison['unique_players']['real']} unique players"
    )
    print(
        f"   Synthetic: {comparison['unique_players']['synthetic']} unique players"
    )

    # Save comparison to file
    with open(
        "src/synthetic_data/configs/recipe_validation_results.json", "w"
    ) as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 60)
    print(
        "Validation complete! Results saved to recipe_validation_results.json"
    )

    return comparison


if __name__ == "__main__":
    validate_recipe(n_synthetic_tournaments=100, n_synthetic_players=1000)
