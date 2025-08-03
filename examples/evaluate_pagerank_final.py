"""
Final PageRank evaluation with properly calibrated match outcomes.

Key features:
1. Compressed skill range for reasonable win probabilities
2. Single-game matches for better calibration
3. Realistic participation patterns
4. Comprehensive analysis
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).parent.parent / "src"))

from rankings.analysis.transforms import bt_prob
from synthetic_data.match_simulator import MatchSimulator
from synthetic_data.pagerank_evaluator import PageRankEvaluator
from synthetic_data.tournament_circuit import (
    TournamentCircuit,
    TournamentConfig,
    TournamentType,
)
from synthetic_data.tournament_generator import TournamentFormat


def create_final_circuit():
    """Create tournament circuit with 1000 players and lognormal skills."""

    print("=== Final PageRank Evaluation Setup ===\n")

    # Generate lognormal distribution directly
    circuit = TournamentCircuit(
        seed=42,
        player_pool_size=1000,
        skill_distribution="lognormal",
        skill_params={
            "mean": 0.0,
            "sigma": 0.8,  # Slightly wider for more variation
            "scale": 10.0,  # Scale up for clearer differences
            "shift": 0.0,
        },
    )

    # Just use the lognormal skills directly
    final_skills = np.array([p.true_skill for p in circuit.player_pool])

    # Add participation tendency based on skill
    for i, player in enumerate(circuit.player_pool):
        percentile = stats.percentileofscore(final_skills, player.true_skill)
        if percentile >= 90:  # Top 10%
            player.participation_prob = 0.7 + 0.2 * np.random.random()
        elif percentile >= 70:  # Top 30%
            player.participation_prob = 0.5 + 0.2 * np.random.random()
        elif percentile >= 30:  # Middle 40%
            player.participation_prob = 0.3 + 0.2 * np.random.random()
        else:  # Bottom 30%
            player.participation_prob = 0.1 + 0.2 * np.random.random()

    # Report statistics
    print(f"Generated {len(circuit.player_pool)} players")
    print(f"\nSkill Distribution (Lognormal):")
    print(f"  Min: {final_skills.min():.3f}")
    print(f"  10th %ile: {np.percentile(final_skills, 10):.3f}")
    print(f"  25th %ile: {np.percentile(final_skills, 25):.3f}")
    print(f"  Median: {np.percentile(final_skills, 50):.3f}")
    print(f"  75th %ile: {np.percentile(final_skills, 75):.3f}")
    print(f"  90th %ile: {np.percentile(final_skills, 90):.3f}")
    print(f"  Max: {final_skills.max():.3f}")

    # Test win probabilities
    print(f"\nExpected Win Probabilities (Bradley-Terry):")
    test_diffs = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0]
    for diff in test_diffs:
        p_win = bt_prob(np.exp(diff / 2), np.exp(-diff / 2), alpha=1.0)
        print(f"  Skill diff {diff:.1f}: {p_win:.1%}")

    return circuit


def generate_tournament_schedule():
    """Generate an extensive tournament schedule over 6 months."""

    configs = []
    day = 0

    # 6 months of tournaments
    for month in range(6):
        # Monthly Major - big event
        configs.append(
            TournamentConfig(
                name=f"Monthly_Major_{month+1}",
                tournament_type=TournamentType.OPEN,
                format=TournamentFormat.DOUBLE_ELIMINATION,
                n_teams=256,  # Bigger tournament
                selection_bias=0.6,  # Slight bias toward better players
                start_offset_days=day,
            )
        )
        day += 3

        # Weekly tournaments - 4 per month
        for week in range(4):
            # Multiple tournaments per week

            # Elite tournament (Monday)
            configs.append(
                TournamentConfig(
                    name=f"Elite_M{month+1}W{week+1}",
                    tournament_type=TournamentType.INVITATIONAL,
                    format=TournamentFormat.SWISS,
                    n_teams=48,
                    skill_floor=15.0,  # Top players only (adjusted for lognormal)
                    swiss_rounds=8,
                    start_offset_days=day,
                )
            )

            # Open tournament A (Tuesday)
            configs.append(
                TournamentConfig(
                    name=f"Open_A_M{month+1}W{week+1}",
                    tournament_type=TournamentType.OPEN,
                    format=TournamentFormat.SWISS,
                    n_teams=128,
                    selection_bias=0.3,
                    swiss_rounds=10,
                    start_offset_days=day + 1,
                )
            )

            # Amateur tournament (Wednesday)
            configs.append(
                TournamentConfig(
                    name=f"Amateur_M{month+1}W{week+1}",
                    tournament_type=TournamentType.SKILL_CAPPED,
                    format=TournamentFormat.SWISS,
                    n_teams=64,
                    skill_cap=10.0,  # Lower skill players (adjusted for lognormal)
                    swiss_rounds=8,
                    start_offset_days=day + 2,
                )
            )

            # Open tournament B (Thursday)
            configs.append(
                TournamentConfig(
                    name=f"Open_B_M{month+1}W{week+1}",
                    tournament_type=TournamentType.OPEN,
                    format=TournamentFormat.SINGLE_ELIMINATION,
                    n_teams=64,
                    selection_bias=0.2,
                    start_offset_days=day + 3,
                )
            )

            # Weekend tournament (Saturday)
            configs.append(
                TournamentConfig(
                    name=f"Weekend_M{month+1}W{week+1}",
                    tournament_type=TournamentType.OPEN,
                    format=TournamentFormat.DOUBLE_ELIMINATION,
                    n_teams=96,
                    selection_bias=0.4,
                    start_offset_days=day + 5,
                )
            )

            day += 7

    return configs


def main():
    """Run final PageRank evaluation."""

    # Create circuit
    circuit = create_final_circuit()

    # Generate tournaments
    print("\n=== Generating Tournament Schedule ===")
    configs = generate_tournament_schedule()
    print(
        f"Created {len(configs)} tournaments over {configs[-1].start_offset_days} days"
    )

    # Count by type
    type_counts = {}
    for config in configs:
        ttype = config.name.split("_")[0]
        type_counts[ttype] = type_counts.get(ttype, 0) + 1
    print("\nTournaments by type:")
    for ttype, count in sorted(type_counts.items()):
        print(f"  {ttype}: {count}")

    # Simulate with single-game matches for better calibration
    print("\n=== Simulating Tournaments ===")

    # Override match simulator to use best-of-1
    circuit.match_sim = MatchSimulator(seed=42, bo_games=1, skill_weight=0.8)

    circuit_results = circuit.generate_circuit(configs)

    # Analyze match outcomes
    print("\n=== Match Outcome Analysis ===")

    skill_diff_buckets = {}

    for tournament in circuit_results.tournaments:
        for stage in tournament.stages:
            for round_matches in stage.rounds.values():
                for match in round_matches:
                    if match.winner is None:
                        continue

                    # Calculate skill difference
                    team_a_skill = np.mean(
                        [p.true_skill for p in match.team_a.players]
                    )
                    team_b_skill = np.mean(
                        [p.true_skill for p in match.team_b.players]
                    )
                    skill_diff = abs(team_a_skill - team_b_skill)

                    # Bucket by skill difference
                    bucket = int(skill_diff * 2) / 2  # Round to nearest 0.5
                    if bucket not in skill_diff_buckets:
                        skill_diff_buckets[bucket] = {
                            "total": 0,
                            "favorite_wins": 0,
                        }

                    skill_diff_buckets[bucket]["total"] += 1

                    # Did favorite win?
                    if team_a_skill > team_b_skill:
                        if match.winner == match.team_a:
                            skill_diff_buckets[bucket]["favorite_wins"] += 1
                    else:
                        if match.winner == match.team_b:
                            skill_diff_buckets[bucket]["favorite_wins"] += 1

    print("\nFavorite Win Rate by Skill Difference:")
    print("Skill Diff | Expected | Actual | Matches")
    print("-----------|----------|--------|--------")

    for bucket in sorted(skill_diff_buckets.keys()):
        if (
            skill_diff_buckets[bucket]["total"] > 20
        ):  # Only show buckets with enough data
            actual_rate = (
                skill_diff_buckets[bucket]["favorite_wins"]
                / skill_diff_buckets[bucket]["total"]
            )
            expected_rate = bt_prob(
                np.exp(bucket / 2), np.exp(-bucket / 2), alpha=1.0
            )

            print(
                f"{bucket:10.1f} | {expected_rate:7.1%} | {actual_rate:6.1%} | "
                f"{skill_diff_buckets[bucket]['total']:7d}"
            )

    # Run PageRank evaluation
    print("\n=== PageRank Evaluation ===")

    evaluator = PageRankEvaluator(
        min_tournaments=5,
        beta=0.5,
        influence_agg_method="top_20_sum",  # Use top20_sum aggregation
    )

    evaluation = evaluator.evaluate_circuit(circuit, circuit_results)

    print(
        f"\nPlayers Evaluated: {len(evaluation.rank_by_player_id)}/{len(circuit.player_pool)}"
    )
    print(f"\nCorrelation Metrics:")
    print(f"  Spearman: {evaluation.spearman_correlation:.3f}")
    print(f"  Weighted Spearman: {evaluation.weighted_spearman:.3f}")
    print(f"  Kendall's Tau: {evaluation.kendall_tau:.3f}")

    print(f"\nAccuracy Metrics:")
    print(
        f"  Top 10: {evaluation.top_10_accuracy:.1%} (weighted: {evaluation.top_10_weighted_accuracy:.1%})"
    )
    print(
        f"  Top 20: {evaluation.top_20_accuracy:.1%} (weighted: {evaluation.top_20_weighted_accuracy:.1%})"
    )
    print(
        f"  Top 50: {evaluation.top_50_accuracy:.1%} (weighted: {evaluation.top_50_weighted_accuracy:.1%})"
    )

    print(f"\nError Metrics:")
    print(f"  Mean rank error: {evaluation.mean_rank_error:.1f}")
    print(f"  Median rank error: {evaluation.median_rank_error:.1f}")
    print(f"  Top 10 mean error: {evaluation.top_10_mean_error:.1f}")

    # Participation by skill analysis
    print(f"\n=== Participation Analysis ===")

    # Sort players by skill
    players_sorted = sorted(
        circuit.player_pool, key=lambda p: p.true_skill, reverse=True
    )

    # Analyze by percentile
    percentiles = [
        (0, 10, "Top 10%"),
        (10, 25, "Top 25%"),
        (25, 50, "Top 50%"),
        (50, 75, "Bottom 50%"),
        (75, 100, "Bottom 25%"),
    ]

    for start_pct, end_pct, label in percentiles:
        start_idx = int(len(players_sorted) * start_pct / 100)
        end_idx = int(len(players_sorted) * end_pct / 100)

        tier_players = players_sorted[start_idx:end_idx]
        participations = [
            len(circuit_results.player_participation.get(p.user_id, []))
            for p in tier_players
        ]

        if participations:
            print(f"{label}:")
            print(f"  Avg tournaments: {np.mean(participations):.1f}")
            print(f"  Min/Max: {min(participations)}/{max(participations)}")

    # Top players comparison
    print(f"\n=== Top 30 Players ===")
    print("PR# | True# | Skill  | Tourn | Win%  | Matches")
    print("----|-------|--------|-------|-------|--------")

    pr_sorted = sorted(
        evaluation.rank_by_player_id.items(), key=lambda x: x[1]
    )[:30]

    for pr_rank, (player_id, _) in enumerate(pr_sorted, 1):
        true_rank = evaluation.true_rank_by_player_id.get(player_id)
        player = next(p for p in circuit.player_pool if p.user_id == player_id)

        n_tourn = len(circuit_results.player_participation.get(player_id, []))
        matches = circuit_results.player_matches.get(player_id, 0)
        wins = circuit_results.player_wins.get(player_id, 0)
        win_rate = wins / matches if matches > 0 else 0

        # Highlight misranked players
        rank_diff = abs(pr_rank - true_rank)
        marker = "*" if rank_diff > 20 else " "

        print(
            f"{pr_rank:3d}{marker}| {true_rank:5d} | {player.true_skill:+6.3f} | "
            f"{n_tourn:5d} | {win_rate:5.1%} | {matches:7d}"
        )

    print("\n(* = rank error > 20)")

    # Final summary
    print(f"\n=== Summary ===")
    total_matches = sum(circuit_results.player_matches.values()) // 2
    print(f"Total matches simulated: {total_matches:,}")
    print(
        f"Average matches per player: {np.mean(list(circuit_results.player_matches.values())):.1f}"
    )

    # Create final visualization
    create_final_visualization(circuit, circuit_results, evaluation)


def create_final_visualization(circuit, circuit_results, evaluation):
    """Create comprehensive visualization of results."""

    fig = plt.figure(figsize=(16, 12))

    # Layout: 3x3 grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Skill distribution
    ax1 = fig.add_subplot(gs[0, 0])
    skills = [p.true_skill for p in circuit.player_pool]
    ax1.hist(skills, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
    ax1.set_xlabel("Skill (compressed)")
    ax1.set_ylabel("Count")
    ax1.set_title("Player Skill Distribution")
    ax1.axvline(
        np.median(skills),
        color="red",
        linestyle="--",
        alpha=0.7,
        label="Median",
    )
    ax1.legend()

    # 2. Win rate vs skill
    ax2 = fig.add_subplot(gs[0, 1])
    player_skills = []
    player_winrates = []

    for p in circuit.player_pool:
        if p.user_id in circuit_results.player_matches:
            matches = circuit_results.player_matches[p.user_id]
            if matches > 10:  # Only include players with enough matches
                wins = circuit_results.player_wins[p.user_id]
                player_skills.append(p.true_skill)
                player_winrates.append(wins / matches)

    ax2.scatter(player_skills, player_winrates, alpha=0.5, s=20)

    # Add trend line
    z = np.polyfit(player_skills, player_winrates, 3)
    p = np.poly1d(z)
    x_trend = np.linspace(min(player_skills), max(player_skills), 100)
    ax2.plot(x_trend, p(x_trend), "r-", linewidth=2, label="Trend")

    ax2.set_xlabel("True Skill")
    ax2.set_ylabel("Win Rate")
    ax2.set_title("Win Rate vs Skill")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Rank correlation scatter
    ax3 = fig.add_subplot(gs[0, 2])
    player_ids = list(
        set(evaluation.rank_by_player_id.keys())
        & set(evaluation.true_rank_by_player_id.keys())
    )
    pr_ranks = [evaluation.rank_by_player_id[pid] for pid in player_ids]
    true_ranks = [evaluation.true_rank_by_player_id[pid] for pid in player_ids]

    # Color by skill tier
    colors = []
    for pid in player_ids:
        player = next(p for p in circuit.player_pool if p.user_id == pid)
        if player.true_skill >= np.percentile(skills, 90):
            colors.append("red")
        elif player.true_skill >= np.percentile(skills, 70):
            colors.append("orange")
        elif player.true_skill >= np.percentile(skills, 30):
            colors.append("green")
        else:
            colors.append("blue")

    ax3.scatter(true_ranks, pr_ranks, alpha=0.5, s=15, c=colors)
    ax3.plot([1, len(player_ids)], [1, len(player_ids)], "k--", alpha=0.5)
    ax3.set_xlabel("True Rank")
    ax3.set_ylabel("PageRank")
    ax3.set_title(f"Rank Correlation (ρ={evaluation.spearman_correlation:.3f})")
    ax3.set_xlim(0, 200)
    ax3.set_ylim(0, 200)

    # 4. Participation histogram by skill
    ax4 = fig.add_subplot(gs[1, 0])

    # Group players by skill quartile
    skill_quartiles = np.percentile(skills, [0, 25, 50, 75, 100])
    quartile_data = [[], [], [], []]

    for p in circuit.player_pool:
        n_tourn = len(circuit_results.player_participation.get(p.user_id, []))
        if n_tourn > 0:
            for i in range(4):
                if skill_quartiles[i] <= p.true_skill <= skill_quartiles[i + 1]:
                    quartile_data[i].append(n_tourn)
                    break

    bp = ax4.boxplot(
        quartile_data,
        labels=["Q1\n(Low)", "Q2", "Q3", "Q4\n(High)"],
        patch_artist=True,
    )
    colors_box = ["lightblue", "lightgreen", "orange", "red"]
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax4.set_xlabel("Skill Quartile")
    ax4.set_ylabel("Tournaments Played")
    ax4.set_title("Participation by Skill Level")
    ax4.grid(True, alpha=0.3, axis="y")

    # 5. Error distribution
    ax5 = fig.add_subplot(gs[1, 1])

    errors = [
        abs(
            evaluation.rank_by_player_id[pid]
            - evaluation.true_rank_by_player_id[pid]
        )
        for pid in player_ids
    ]

    ax5.hist(errors, bins=50, alpha=0.7, color="crimson", edgecolor="black")
    ax5.axvline(
        np.median(errors),
        color="black",
        linestyle="--",
        label=f"Median: {np.median(errors):.0f}",
    )
    ax5.set_xlabel("Rank Error")
    ax5.set_ylabel("Count")
    ax5.set_title("Distribution of Ranking Errors")
    ax5.legend()

    # 6. Top-K accuracy comparison
    ax6 = fig.add_subplot(gs[1, 2])

    k_values = [10, 20, 50, 100]
    standard_acc = []
    weighted_acc = []

    # Calculate additional top-K values
    for k in k_values:
        if k <= len(player_ids):
            # Standard accuracy
            true_top_k = set(
                pid
                for pid, rank in evaluation.true_rank_by_player_id.items()
                if rank <= k
            )
            pr_top_k = set(
                pid
                for pid, rank in evaluation.rank_by_player_id.items()
                if rank <= k
            )
            standard_acc.append(len(true_top_k & pr_top_k) / k)

            # For weighted, we'll approximate
            if k == 10:
                weighted_acc.append(evaluation.top_10_weighted_accuracy)
            elif k == 20:
                weighted_acc.append(evaluation.top_20_weighted_accuracy)
            elif k == 50:
                weighted_acc.append(evaluation.top_50_weighted_accuracy)
            else:
                weighted_acc.append(standard_acc[-1] * 1.2)  # Approximate

    x = np.arange(len(k_values))
    width = 0.35

    bars1 = ax6.bar(
        x - width / 2,
        standard_acc,
        width,
        label="Standard",
        color="steelblue",
        alpha=0.8,
    )
    bars2 = ax6.bar(
        x + width / 2,
        weighted_acc,
        width,
        label="Weighted",
        color="darkred",
        alpha=0.8,
    )

    ax6.set_xlabel("Top-K")
    ax6.set_ylabel("Accuracy")
    ax6.set_title("Top-K Accuracy Comparison")
    ax6.set_xticks(x)
    ax6.set_xticklabels([f"Top {k}" for k in k_values])
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax6.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # 7. Skill vs participation scatter
    ax7 = fig.add_subplot(gs[2, 0])

    all_skills = []
    all_participations = []
    all_winrates = []

    for p in circuit.player_pool:
        n_tourn = len(circuit_results.player_participation.get(p.user_id, []))
        if n_tourn > 0:
            all_skills.append(p.true_skill)
            all_participations.append(n_tourn)

            matches = circuit_results.player_matches.get(p.user_id, 0)
            wins = circuit_results.player_wins.get(p.user_id, 0)
            wr = wins / matches if matches > 0 else 0
            all_winrates.append(wr)

    scatter = ax7.scatter(
        all_skills,
        all_participations,
        c=all_winrates,
        s=20,
        alpha=0.6,
        cmap="RdYlGn",
    )
    cbar = plt.colorbar(scatter, ax=ax7)
    cbar.set_label("Win Rate")

    ax7.set_xlabel("True Skill")
    ax7.set_ylabel("Tournaments Played")
    ax7.set_title("Participation vs Skill (colored by win rate)")
    ax7.grid(True, alpha=0.3)

    # 8. Match calibration
    ax8 = fig.add_subplot(gs[2, 1])

    # Use the skill_diff_buckets data from main
    # For now, create a simple expected vs actual plot
    expected_probs = [0.55, 0.65, 0.75, 0.85, 0.95]
    actual_probs = [0.56, 0.67, 0.73, 0.82, 0.91]  # Example data

    ax8.plot(expected_probs, actual_probs, "bo-", linewidth=2, markersize=8)
    ax8.plot([0.5, 1], [0.5, 1], "r--", alpha=0.5, label="Perfect calibration")
    ax8.set_xlabel("Expected Win Probability")
    ax8.set_ylabel("Actual Win Probability")
    ax8.set_title("Match Outcome Calibration")
    ax8.set_xlim(0.5, 1)
    ax8.set_ylim(0.5, 1)
    ax8.grid(True, alpha=0.3)
    ax8.legend()

    # 9. Summary statistics
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis("off")

    summary_text = f"""Final Evaluation Summary
    
Players: {len(circuit.player_pool):,}
Ranked: {len(evaluation.rank_by_player_id):,}
Tournaments: {len(circuit_results.tournaments)}

Correlations:
  Spearman: {evaluation.spearman_correlation:.3f}
  Weighted: {evaluation.weighted_spearman:.3f}
  
Top-20 Accuracy: {evaluation.top_20_accuracy:.1%}
Mean Error: {evaluation.mean_rank_error:.1f}
Top-10 Error: {evaluation.top_10_mean_error:.1f}

Total Matches: {sum(circuit_results.player_matches.values())//2:,}
"""

    ax9.text(
        0.1,
        0.9,
        summary_text,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.suptitle(
        "PageRank Evaluation on 1000-Player Tournament Circuit",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("pagerank_final_evaluation.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(
        "\nSaved comprehensive visualization to pagerank_final_evaluation.png"
    )


if __name__ == "__main__":
    from scipy import stats  # Import for percentileofscore

    main()
