"""
PageRank evaluation with realistic skill compression for better Bradley-Terry probabilities.

Key improvements:
1. Compress skill differences to keep win probabilities in reasonable range (55-85%)
2. Use more realistic skill distribution
3. Add skill-based participation bias (better players play more)
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rankings.analysis.transforms import bt_prob
from synthetic_data.pagerank_evaluator import PageRankEvaluator
from synthetic_data.tournament_circuit import (
    TournamentCircuit,
    TournamentConfig,
    TournamentType,
)
from synthetic_data.tournament_generator import TournamentFormat


def create_realistic_circuit():
    """Create circuit with compressed skill range for realistic win probabilities."""

    print("=== Creating Realistic Tournament Circuit ===\n")

    # First generate raw lognormal distribution
    circuit = TournamentCircuit(
        seed=42,
        player_pool_size=1000,
        skill_distribution="lognormal",
        skill_params={
            "mean": 0.0,
            "sigma": 0.8,
            "scale": 1.0,
            "shift": 0.0,
        },
    )

    # Compress skills to reasonable range
    # We want max skill difference to give ~95% win probability, not 100%
    skills = np.array([p.true_skill for p in circuit.player_pool])

    # Apply logarithmic compression for extreme values
    compressed_skills = np.sign(skills) * np.log1p(np.abs(skills))

    # Scale to desired range (roughly -2 to +3)
    min_skill = compressed_skills.min()
    max_skill = compressed_skills.max()

    # Normalize and rescale
    normalized = (compressed_skills - min_skill) / (max_skill - min_skill)
    final_skills = normalized * 5 - 2  # Range: -2 to +3

    # Update player skills
    for i, player in enumerate(circuit.player_pool):
        player.true_skill = final_skills[i]

    # Show skill distribution
    print(
        f"Generated {len(circuit.player_pool)} players with compressed skills"
    )
    print(f"\nSkill Distribution:")
    print(f"  Min: {final_skills.min():.2f}")
    print(f"  25th %ile: {np.percentile(final_skills, 25):.2f}")
    print(f"  Median: {np.median(final_skills):.2f}")
    print(f"  75th %ile: {np.percentile(final_skills, 75):.2f}")
    print(f"  95th %ile: {np.percentile(final_skills, 95):.2f}")
    print(f"  Max: {final_skills.max():.2f}")

    # Test Bradley-Terry probabilities
    print(f"\nBradley-Terry Win Probabilities:")
    skill_diffs = [
        (0.5, "Small"),
        (1.0, "Medium"),
        (2.0, "Large"),
        (3.0, "Huge"),
        (5.0, "Extreme"),
    ]

    for diff, label in skill_diffs:
        # Calculate probability for skill difference
        p_win = bt_prob(np.exp(diff / 2), np.exp(-diff / 2), alpha=1.0)
        print(f"  {label} difference ({diff:.1f}): {p_win:.1%}")

    return circuit


def add_participation_bias(circuit, configs):
    """
    Modify tournament configs to add realistic participation bias.
    Better players should play more elite tournaments.
    """
    # Sort players by skill
    players_by_skill = sorted(
        circuit.player_pool, key=lambda p: p.true_skill, reverse=True
    )

    # Create participation multipliers based on skill percentile
    for i, player in enumerate(players_by_skill):
        percentile = (i / len(players_by_skill)) * 100

        if percentile <= 5:  # Top 5%
            # Very active, play most tournaments
            player.participation_rate = 0.8 + np.random.uniform(0, 0.2)
        elif percentile <= 20:  # Top 20%
            # Active
            player.participation_rate = 0.6 + np.random.uniform(0, 0.2)
        elif percentile <= 50:  # Top 50%
            # Moderate
            player.participation_rate = 0.4 + np.random.uniform(0, 0.2)
        else:  # Bottom 50%
            # Less active
            player.participation_rate = 0.2 + np.random.uniform(0, 0.2)

    # For elite tournaments, boost elite player participation
    for config in configs:
        if "Elite" in config.name or "Premier" in config.name:
            # Elite players much more likely to participate
            config.selection_bias = 0.8
        elif "Amateur" in config.name or "Beginner" in config.name:
            # Reverse bias for amateur tournaments
            config.selection_bias = -0.3  # Negative bias = prefer lower skill

    return configs


def analyze_win_probabilities(circuit, circuit_results):
    """Analyze actual win probabilities in matches."""

    print("\n=== Analyzing Match Win Probabilities ===")

    # Collect win probabilities from actual matches
    probabilities = []
    outcomes = []  # 1 if favorite won, 0 if underdog won

    for tournament in circuit_results.tournaments:
        for stage in tournament.stages:
            for round_matches in stage.rounds.values():
                for match in round_matches:
                    if match.winner is None:
                        continue

                    # Calculate team average skills
                    team_a_skill = np.mean(
                        [p.true_skill for p in match.team_a.players]
                    )
                    team_b_skill = np.mean(
                        [p.true_skill for p in match.team_b.players]
                    )

                    # Calculate Bradley-Terry probability
                    p_a_wins = bt_prob(
                        np.exp(team_a_skill), np.exp(team_b_skill), alpha=1.0
                    )

                    # Record probability and outcome
                    if p_a_wins >= 0.5:
                        # Team A is favorite
                        probabilities.append(p_a_wins)
                        outcomes.append(
                            1 if match.winner == match.team_a else 0
                        )
                    else:
                        # Team B is favorite
                        probabilities.append(1 - p_a_wins)
                        outcomes.append(
                            1 if match.winner == match.team_b else 0
                        )

    # Bin probabilities and check calibration
    bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]

    print("\nCalibration (Expected vs Actual win rates):")
    print("Probability | Expected | Actual | Count")
    print("------------|----------|--------|-------")

    for min_p, max_p in bins:
        mask = [(min_p <= p < max_p) for p in probabilities]
        bin_probs = [p for p, m in zip(probabilities, mask) if m]
        bin_outcomes = [o for o, m in zip(outcomes, mask) if m]

        if bin_probs:
            expected = np.mean(bin_probs)
            actual = np.mean(bin_outcomes)
            print(
                f"{min_p:.1f}-{max_p:.1f}     | {expected:.1%}   | {actual:.1%} | {len(bin_probs)}"
            )

    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"  Total matches: {len(probabilities)}")
    print(f"  Mean probability: {np.mean(probabilities):.1%}")
    print(f"  Matches with P > 0.9: {sum(1 for p in probabilities if p > 0.9)}")
    print(
        f"  Matches with P > 0.95: {sum(1 for p in probabilities if p > 0.95)}"
    )
    print(
        f"  Matches with P > 0.99: {sum(1 for p in probabilities if p > 0.99)}"
    )


def main():
    """Run realistic PageRank evaluation."""

    # Create circuit with compressed skills
    circuit = create_realistic_circuit()

    # Generate tournament configurations
    print("\n=== Generating Tournament Season ===")

    configs = []
    for i in range(100):  # 100 tournaments
        if i % 10 == 0:
            # Premier tournament
            config = TournamentConfig(
                name=f"Premier_{i//10 + 1}",
                tournament_type=TournamentType.OPEN,
                format=TournamentFormat.DOUBLE_ELIMINATION,
                n_teams=64,
                selection_bias=0.6,
                start_offset_days=i * 3,
            )
        elif i % 10 in [2, 5, 8]:
            # Elite tournaments
            config = TournamentConfig(
                name=f"Elite_{i+1}",
                tournament_type=TournamentType.INVITATIONAL,
                format=TournamentFormat.SWISS,
                n_teams=32,
                skill_floor=1.0,  # Compressed scale
                swiss_rounds=7,
                start_offset_days=i * 3,
            )
        elif i % 10 in [1, 4, 7]:
            # Open tournaments
            config = TournamentConfig(
                name=f"Open_{i+1}",
                tournament_type=TournamentType.OPEN,
                format=TournamentFormat.SWISS,
                n_teams=48,
                selection_bias=0.3,
                swiss_rounds=8,
                start_offset_days=i * 3,
            )
        else:
            # Amateur tournaments
            config = TournamentConfig(
                name=f"Amateur_{i+1}",
                tournament_type=TournamentType.SKILL_CAPPED,
                format=TournamentFormat.SWISS,
                n_teams=32,
                skill_cap=0.5,  # Compressed scale
                swiss_rounds=6,
                start_offset_days=i * 3,
            )

        configs.append(config)

    # Add participation bias
    configs = add_participation_bias(circuit, configs)

    print(f"Created {len(configs)} tournaments")

    # Generate circuit
    print("\nSimulating tournaments...")
    circuit_results = circuit.generate_circuit(configs)

    # Analyze win probabilities
    analyze_win_probabilities(circuit, circuit_results)

    # Run PageRank evaluation
    print("\n=== PageRank Evaluation ===")

    evaluator = PageRankEvaluator(
        min_tournaments=10,
        beta=0.5,
    )

    evaluation = evaluator.evaluate_circuit(circuit, circuit_results)

    print(f"\nOverall Results:")
    print(f"  Players ranked: {len(evaluation.rank_by_player_id)}")
    print(f"  Spearman correlation: {evaluation.spearman_correlation:.3f}")
    print(f"  Weighted Spearman: {evaluation.weighted_spearman:.3f}")
    print(f"  Kendall's tau: {evaluation.kendall_tau:.3f}")

    print(f"\nTop-K Accuracy:")
    print(f"  Top 10: {evaluation.top_10_accuracy:.1%}")
    print(f"  Top 20: {evaluation.top_20_accuracy:.1%}")
    print(f"  Top 50: {evaluation.top_50_accuracy:.1%}")

    print(f"\nError Metrics:")
    print(f"  Mean rank error: {evaluation.mean_rank_error:.1f}")
    print(f"  Median rank error: {evaluation.median_rank_error:.1f}")
    print(f"  Top 10 mean error: {evaluation.top_10_mean_error:.1f}")

    # Analyze participation impact
    print(f"\n=== Participation Analysis ===")

    # Group by skill tier and check participation
    skill_tiers = [
        ("Elite (top 10%)", 0, 100),
        ("High (10-25%)", 100, 250),
        ("Mid (25-50%)", 250, 500),
        ("Low (bottom 50%)", 500, 1000),
    ]

    players_by_skill = sorted(
        circuit.player_pool, key=lambda p: p.true_skill, reverse=True
    )

    for tier_name, start_idx, end_idx in skill_tiers:
        tier_players = players_by_skill[start_idx:end_idx]
        participations = [
            len(circuit_results.player_participation[p.user_id])
            for p in tier_players
        ]

        if participations:
            print(f"{tier_name}:")
            print(f"  Avg tournaments: {np.mean(participations):.1f}")
            print(
                f"  Participation rate: {np.mean([p.participation_rate for p in tier_players]):.1%}"
            )

    # Show top 20 comparison
    print(f"\n=== Top 20 Comparison ===")
    print("PR# | True# | Skill | Tourn | Win% | Player")
    print("----|-------|-------|-------|------|-------")

    pr_sorted = sorted(
        evaluation.rank_by_player_id.items(), key=lambda x: x[1]
    )[:20]

    for pr_rank, (player_id, _) in enumerate(pr_sorted, 1):
        true_rank = evaluation.true_rank_by_player_id.get(player_id)
        player = next(p for p in circuit.player_pool if p.user_id == player_id)

        n_tourn = len(circuit_results.player_participation[player_id])
        matches = circuit_results.player_matches.get(player_id, 0)
        wins = circuit_results.player_wins.get(player_id, 0)
        win_rate = wins / matches if matches > 0 else 0

        print(
            f"{pr_rank:3d} | {true_rank:5d} | {player.true_skill:+5.2f} | "
            f"{n_tourn:5d} | {win_rate:4.1%} | Player_{player_id}"
        )

    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Skill distribution
    skills = [p.true_skill for p in circuit.player_pool]
    ax1.hist(skills, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
    ax1.set_xlabel("Compressed Skill")
    ax1.set_ylabel("Number of Players")
    ax1.set_title("Compressed Skill Distribution")
    ax1.axvline(np.median(skills), color="red", linestyle="--", label="Median")
    ax1.legend()

    # 2. Win probability distribution
    probs = []
    for _ in range(1000):
        # Sample random matchup
        p1, p2 = np.random.choice(circuit.player_pool, 2, replace=False)
        p_win = bt_prob(np.exp(p1.true_skill), np.exp(p2.true_skill), alpha=1.0)
        probs.append(max(p_win, 1 - p_win))  # Favorite's probability

    ax2.hist(probs, bins=30, alpha=0.7, color="green", edgecolor="black")
    ax2.set_xlabel("Favorite Win Probability")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of Match Win Probabilities")
    ax2.axvline(0.95, color="red", linestyle="--", label="95% threshold")
    ax2.legend()

    # 3. Rank correlation
    player_ids = list(
        set(evaluation.rank_by_player_id.keys())
        & set(evaluation.true_rank_by_player_id.keys())
    )
    pr_ranks = [evaluation.rank_by_player_id[pid] for pid in player_ids]
    true_ranks = [evaluation.true_rank_by_player_id[pid] for pid in player_ids]

    ax3.scatter(true_ranks, pr_ranks, alpha=0.5, s=15)
    ax3.plot([1, len(player_ids)], [1, len(player_ids)], "r--", label="Perfect")
    ax3.set_xlabel("True Rank")
    ax3.set_ylabel("PageRank")
    ax3.set_title(f"Rank Correlation (ρ={evaluation.spearman_correlation:.3f})")
    ax3.legend()
    ax3.set_xlim(0, min(300, len(player_ids)))
    ax3.set_ylim(0, min(300, len(player_ids)))

    # 4. Participation by skill
    skills_all = [p.true_skill for p in circuit.player_pool]
    participations_all = [
        len(circuit_results.player_participation[p.user_id])
        for p in circuit.player_pool
    ]

    scatter = ax4.scatter(
        skills_all,
        participations_all,
        alpha=0.5,
        s=20,
        c=participations_all,
        cmap="viridis",
    )
    ax4.set_xlabel("True Skill")
    ax4.set_ylabel("Tournaments Played")
    ax4.set_title("Participation vs Skill")
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label("Tournaments")

    plt.suptitle(
        "Realistic PageRank Evaluation with Compressed Skills", fontsize=16
    )
    plt.tight_layout()
    plt.savefig(
        "realistic_pagerank_evaluation.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print("\nSaved visualization to realistic_pagerank_evaluation.png")


if __name__ == "__main__":
    main()
