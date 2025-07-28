#!/usr/bin/env python3
"""
Example usage of the Sendou.ink tournament rankings module with scraping capabilities.

This script demonstrates how to use the reorganized rankings module with its
focused submodules for scraping, analysis, and data management.
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Import the rankings module - now with clean submodule organization
from rankings import (
    RatingEngine,
    discover_tournaments_from_calendar,
    get_tournament_summary,
    load_scraped_tournaments,
    parse_tournaments_data,
    scrape_latest_tournaments,
    scrape_tournament,
    scrape_tournaments_from_calendar,
)
from rankings.analysis.utils import (
    format_top_rankings,
    prepare_player_summary,
    prepare_tournament_summary,
)


def demo_scraping():
    """Demonstrate tournament scraping capabilities."""
    print("=== Tournament Scraping Demo ===\n")

    # Single tournament scraping
    print("1. Scraping a single tournament...")
    try:
        tournament_data = scrape_tournament(1955)  # Example tournament
        print(f"✓ Successfully scraped tournament 1955")
        print(
            f"  Tournament name: {tournament_data.get('tournament', {}).get('ctx', {}).get('name', 'Unknown')}"
        )
    except Exception as e:
        print(f"✗ Failed to scrape tournament 1955: {e}")

    # Calendar-based discovery
    print("\n2. Discovering tournaments from calendar...")
    try:
        tournament_ids = discover_tournaments_from_calendar()
        if tournament_ids:
            print(f"✓ Discovered {len(tournament_ids)} tournaments")
            print(f"  Latest tournament ID: {max(tournament_ids)}")
            print(f"  Oldest tournament ID: {min(tournament_ids)}")
        else:
            print("✗ No tournaments discovered from calendar")
    except Exception as e:
        print(f"✗ Failed to discover tournaments: {e}")

    # Batch scraping (small example)
    print("\n3. Batch scraping example...")
    try:
        results = scrape_latest_tournaments(
            count=5, output_dir="data/example_tournaments"
        )
        print(f"✓ Scraping completed:")
        print(f"  Scraped: {results['scraped']}")
        print(f"  Failed: {results['failed']}")
        if results["failed_ids"]:
            print(f"  Failed IDs: {results['failed_ids']}")
    except Exception as e:
        print(f"✗ Batch scraping failed: {e}")


def load_or_use_sample_data():
    """Load scraped tournament data or use sample data."""
    print("=== Loading Tournament Data ===\n")

    # Try to load scraped data first
    try:
        tournaments = load_scraped_tournaments("data/example_tournaments")
        if tournaments:
            print(f"✓ Loaded {len(tournaments)} tournaments from scraped data")
            return tournaments
    except Exception as e:
        print(f"⚠ Could not load scraped data: {e}")

    # Fallback to existing data directories
    for data_dir in ["data/tournaments", "data"]:
        try:
            tournaments = load_scraped_tournaments(data_dir)
            if tournaments:
                print(
                    f"✓ Loaded {len(tournaments)} tournaments from {data_dir}"
                )
                return tournaments
        except Exception:
            continue

    # If no data found, scrape a small sample
    print("⚠ No existing tournament data found")
    print("Attempting to scrape a small sample...")

    try:
        # Scrape just a few recent tournaments
        results = scrape_latest_tournaments(
            count=3, output_dir="data/sample_tournaments"
        )
        if results["scraped"] > 0:
            tournaments = load_scraped_tournaments("data/sample_tournaments")
            print(f"✓ Scraped and loaded {len(tournaments)} sample tournaments")
            return tournaments
    except Exception as e:
        print(f"✗ Failed to scrape sample data: {e}")

    print("✗ Could not obtain tournament data")
    return []


def demo_analysis(tournaments_data):
    """Demonstrate analysis capabilities on tournament data."""
    if not tournaments_data:
        print("⚠ No tournament data available for analysis")
        return

    print("=== Tournament Analysis Demo ===\n")

    # Parse tournament data
    print("1. Parsing tournament data...")
    tables = parse_tournaments_data(tournaments_data)

    matches_df = tables["matches"]
    teams_df = tables["teams"]
    players_df = tables["players"]

    if matches_df is None or players_df is None:
        print("✗ Insufficient data for analysis")
        return

    print(f"✓ Parsed tournament data:")
    print(f"  Matches: {matches_df.height}")
    print(f"  Teams: {teams_df.height if teams_df else 0}")
    print(f"  Players: {players_df.height}")

    # Advanced rating engine analysis
    print("\n2. Running advanced rating engine...")

    engine = RatingEngine(
        decay_half_life_days=30.0,
        damping_factor=0.85,
        beta=1.0,  # Full tournament strength weighting
        tick_tock_stabilize_tol=1e-6,
        max_tick_tock=15,
        influence_agg_method="top_20_sum",
        now=datetime(2025, 7, 23, tzinfo=ZoneInfo("America/Chicago")),
    )

    try:
        print("  Computing player rankings...")
        player_rankings = engine.rank_players(matches_df, players_df)
        print(f"✓ Ranked {player_rankings.height} players")

        # Display results
        if not player_rankings.is_empty():
            player_summary = prepare_player_summary(
                players_df, player_rankings, score_column="player_rank"
            )

            print(f"\n🏆 Top 10 Players:")
            if not player_summary.is_empty():
                top_10 = player_summary.head(10)
                for i, row in enumerate(top_10.iter_rows(named=True), 1):
                    username = row.get("username", "Unknown")
                    score = row.get("score_normalized", 0)
                    tournaments = row.get("tournament_count", 0)
                    print(
                        f"  {i:2d}. {username:<20} {score:6.1f} ({tournaments} tournaments)"
                    )

            # Tournament strength analysis
            if (
                engine.tournament_influence
                and engine.tournament_strength is not None
            ):
                print(f"\n📊 Tournament Analysis:")
                print(
                    f"  Analyzed {len(engine.tournament_influence)} tournaments"
                )

                # Top tournament strengths
                tournament_summary = prepare_tournament_summary(
                    tournaments_data,
                    engine.tournament_influence,
                    engine.tournament_strength,
                )

                print(f"\n🏟️  Top 5 Strongest Tournaments:")
                if not tournament_summary.is_empty():
                    top_tournaments = tournament_summary.head(5)
                    for i, row in enumerate(
                        top_tournaments.iter_rows(named=True), 1
                    ):
                        name = row.get("name", "Unknown")[:40]
                        influence = row.get("influence", 0)
                        strength = row.get("strength", 0)
                        print(
                            f"  {i}. {name:<40} I:{influence:.2f} S:{strength:.2f}"
                        )

    except Exception as e:
        print(f"✗ Analysis failed: {e}")


def demo_data_management():
    """Demonstrate data management capabilities."""
    print("=== Data Management Demo ===\n")

    # Check available data directories
    data_dirs = [
        "data/tournaments",
        "data/example_tournaments",
        "data/sample_tournaments",
    ]

    for data_dir in data_dirs:
        if Path(data_dir).exists():
            try:
                summary = get_tournament_summary(data_dir)
                if not summary.is_empty():
                    print(f"📁 {data_dir}:")
                    print(f"  Tournaments: {summary.height}")
                    finalized = summary.filter(
                        pl.col("is_finalized") == True
                    ).height
                    print(f"  Finalized: {finalized}")
                    print(
                        f"  With matches: {summary.filter(pl.col('has_matches') == True).height}"
                    )

                    if summary.height > 0:
                        avg_teams = summary["team_count"].mean()
                        print(f"  Avg teams per tournament: {avg_teams:.1f}")
                else:
                    print(f"📁 {data_dir}: No tournament data")
            except Exception as e:
                print(f"📁 {data_dir}: Error reading data - {e}")
        else:
            print(f"📁 {data_dir}: Directory not found")


def main():
    print("=== Sendou.ink Tournament Rankings with Scraping ===\n")

    # Demo scraping capabilities
    demo_scraping()
    print()

    # Load tournament data
    tournaments_data = load_or_use_sample_data()
    print()

    # Demo analysis
    demo_analysis(tournaments_data)
    print()

    # Demo data management
    demo_data_management()

    print("\n=== Demo Complete ===")
    print("\nNext steps:")
    print("- Use scrape_latest_tournaments() for regular data updates")
    print("- Use scrape_tournaments_from_calendar() for comprehensive scraping")
    print("- Configure RatingEngine parameters for your specific needs")
    print("- Explore tournament strength analysis for competitive insights")


if __name__ == "__main__":
    main()
