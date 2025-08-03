"""
Example script demonstrating tournament circuit generation and PageRank evaluation.

This script:
1. Creates a fixed population of 200 players with realistic skill distribution
2. Generates a circuit of tournaments with varying types (open, skill-capped, invitational)
3. Simulates all matches using the Bradley-Terry model
4. Evaluates how well PageRank recovers the true skill rankings
"""

import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from synthetic_data.pagerank_evaluator import PageRankEvaluator
from synthetic_data.tournament_circuit import (
    TournamentCircuit,
    TournamentConfig,
    TournamentType,
)
from synthetic_data.tournament_generator import TournamentFormat


def main():
    """Run tournament circuit simulation and PageRank evaluation."""

    print("=== Tournament Circuit PageRank Evaluation ===\n")

    # Initialize tournament circuit with 200 players
    print("Creating player population...")
    circuit = TournamentCircuit(
        seed=42,
        player_pool_size=200,
        skill_distribution="realistic",  # Elite, competitive, and casual mix
    )

    print(f"Generated {len(circuit.player_pool)} players")
    print(
        f"  - Elite (top 10%): {sum(1 for p in circuit.player_pool if p.true_skill >= 2.0)}"
    )
    print(
        f"  - Competitive: {sum(1 for p in circuit.player_pool if 0.5 <= p.true_skill < 2.0)}"
    )
    print(
        f"  - Casual: {sum(1 for p in circuit.player_pool if p.true_skill < 0.5)}\n"
    )

    # Define tournament configurations
    print("Configuring tournament circuit...")
    configs = [
        # Elite invitational - top players only
        TournamentConfig(
            name="Winter_Elite_Championship",
            tournament_type=TournamentType.INVITATIONAL,
            format=TournamentFormat.DOUBLE_ELIMINATION,
            n_teams=8,
            skill_floor=1.5,
            start_offset_days=0,
        ),
        # Amateur tournament - skill capped
        TournamentConfig(
            name="Beginner_Friendly_Cup_1",
            tournament_type=TournamentType.SKILL_CAPPED,
            format=TournamentFormat.SWISS,
            n_teams=24,
            skill_cap=0.5,
            swiss_rounds=5,
            start_offset_days=7,
        ),
        # Open tournament 1
        TournamentConfig(
            name="Spring_Open_1",
            tournament_type=TournamentType.OPEN,
            format=TournamentFormat.SWISS,
            n_teams=32,
            selection_bias=0.3,  # Slight bias towards skilled players
            swiss_rounds=6,
            start_offset_days=14,
        ),
        # Open tournament 2
        TournamentConfig(
            name="Spring_Open_2",
            tournament_type=TournamentType.OPEN,
            format=TournamentFormat.SINGLE_ELIMINATION,
            n_teams=16,
            seeded_bracket=True,
            start_offset_days=21,
        ),
        # Mid-tier tournament
        TournamentConfig(
            name="Competitive_League_1",
            tournament_type=TournamentType.MIXED,
            format=TournamentFormat.ROUND_ROBIN,
            n_teams=8,
            skill_floor=0.0,
            skill_cap=1.5,
            start_offset_days=28,
        ),
        # Large open Swiss
        TournamentConfig(
            name="Summer_Major",
            tournament_type=TournamentType.OPEN,
            format=TournamentFormat.SWISS,
            n_teams=48,
            swiss_rounds=7,
            start_offset_days=35,
        ),
        # Elite round robin
        TournamentConfig(
            name="Pro_League",
            tournament_type=TournamentType.INVITATIONAL,
            format=TournamentFormat.ROUND_ROBIN,
            n_teams=6,
            skill_floor=1.8,
            double_round_robin=True,
            start_offset_days=42,
        ),
        # Amateur tournament 2
        TournamentConfig(
            name="Beginner_Friendly_Cup_2",
            tournament_type=TournamentType.SKILL_CAPPED,
            format=TournamentFormat.SINGLE_ELIMINATION,
            n_teams=16,
            skill_cap=0.3,
            seeded_bracket=False,  # Random bracket
            start_offset_days=49,
        ),
    ]

    # Generate more tournaments using the standard circuit generator
    print(
        f"Generating {len(configs)} custom tournaments + 12 standard tournaments..."
    )

    # First run custom tournaments
    circuit_results = circuit.generate_circuit(configs)

    # Then add standard tournaments
    additional_results = circuit.generate_standard_circuit(
        n_tournaments=12,
        start_date=datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        ),
        tournament_interval_days=7.0,
    )

    print(f"\nGenerated {len(circuit_results.tournaments)} tournaments total")

    # Print tournament summary
    print("\nTournament Summary:")
    for tourney in circuit_results.tournaments[:10]:  # First 10
        n_matches = sum(
            len(matches)
            for stage in tourney.stages
            for matches in stage.rounds.values()
        )
        print(
            f"  - {tourney.name}: {len(tourney.all_teams)} teams, {n_matches} matches"
        )
    if len(circuit_results.tournaments) > 10:
        print(
            f"  ... and {len(circuit_results.tournaments) - 10} more tournaments"
        )

    # Analyze participation
    avg_participation = sum(
        len(tourneys)
        for tourneys in circuit_results.player_participation.values()
    ) / len(circuit.player_pool)

    print(f"\nAverage tournaments per player: {avg_participation:.1f}")

    # Initialize PageRank evaluator
    print("\n=== Evaluating PageRank Performance ===\n")

    evaluator = PageRankEvaluator(
        damping_factor=0.85,
        decay_rate=0.023,  # ~30 day half-life
        beta=0.0,  # No tournament strength weighting initially
        min_tournaments=3,
    )

    # Evaluate PageRank
    evaluation = evaluator.evaluate_circuit(circuit, circuit_results)

    # Print results
    print("Correlation Metrics:")
    print(f"  - Spearman correlation: {evaluation.spearman_correlation:.3f}")
    print(f"  - Kendall's tau: {evaluation.kendall_tau:.3f}")
    print(f"  - Pearson correlation: {evaluation.pearson_correlation:.3f}")

    print("\nRanking Accuracy:")
    print(f"  - Top 10 accuracy: {evaluation.top_10_accuracy:.1%}")
    print(f"  - Top 20 accuracy: {evaluation.top_20_accuracy:.1%}")
    print(f"  - Top 50 accuracy: {evaluation.top_50_accuracy:.1%}")

    print("\nRank Error Metrics:")
    print(f"  - Mean rank error: {evaluation.mean_rank_error:.1f}")
    print(f"  - Median rank error: {evaluation.median_rank_error:.1f}")
    print(f"  - RMSE rank: {evaluation.rmse_rank:.1f}")

    # Show top 10 comparison
    print("\n=== Top 10 Players Comparison ===")
    print("Rank | PageRank                | True Skill Rank")
    print("-----|------------------------|----------------")

    # Get top 10 by PageRank
    pr_sorted = sorted(
        evaluation.rank_by_player_id.items(), key=lambda x: x[1]
    )[:10]

    for pr_rank, (player_id, _) in enumerate(pr_sorted, 1):
        true_rank = evaluation.true_rank_by_player_id.get(player_id, "N/A")
        player = next(p for p in circuit.player_pool if p.user_id == player_id)
        print(
            f"{pr_rank:4d} | Player_{player_id:3d} (s={player.true_skill:+.2f}) | {true_rank}"
        )

    # Parameter sensitivity analysis
    print("\n=== Parameter Sensitivity Analysis ===\n")

    param_ranges = {
        "damping_factor": [0.75, 0.85, 0.95],
        "beta": [0.0, 0.5, 1.0],
    }

    print("Testing different parameter values...")
    sensitivity_results = evaluator.evaluate_parameter_sensitivity(
        circuit, circuit_results, param_ranges
    )

    print("\nResults by parameter:")
    for (param_name, value), eval_result in sensitivity_results.items():
        print(f"\n{param_name} = {value}:")
        print(f"  - Spearman: {eval_result.spearman_correlation:.3f}")
        print(f"  - Top 20 accuracy: {eval_result.top_20_accuracy:.1%}")
        print(f"  - Mean rank error: {eval_result.mean_rank_error:.1f}")

    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()
