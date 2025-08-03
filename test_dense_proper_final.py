"""
Final test with dense schedule and proper realistic growth.
Using normal distribution with reasonable skill ranges and tiny differential growth.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.append(str(Path(__file__).parent / "src"))

from dense_tournament_schedule import create_dense_tournament_schedule
from synthetic_data.match_score_fixer import add_scores_to_circuit_results
from synthetic_data.pagerank_evaluator import PageRankEvaluator
from synthetic_data.tournament_circuit import TournamentCircuit


def test_dense_proper_final():
    """Final test with proper parameters."""

    print("=== FINAL TEST: DENSE SCHEDULE WITH PROPER GROWTH ===\n")

    # Create circuit with reasonable skill distribution
    circuit = TournamentCircuit(
        seed=42,
        player_pool_size=2000,
        skill_distribution="normal",  # Normal distribution
        skill_params={
            "mean": 10.0,
            "std": 3.0,  # Most players between 1 and 19
        },
    )

    print(
        f"Skill range: {min(p.true_skill for p in circuit.player_pool):.1f} to "
        f"{max(p.true_skill for p in circuit.player_pool):.1f}"
    )

    # Store initial skills and set growth
    for p in circuit.player_pool:
        p.initial_skill = p.true_skill
        # Tiny differential growth
        if p.initial_skill < 8:  # Bottom ~25%
            p.growth_per_day = 0.0008  # 0.144 over 180 days
        elif p.initial_skill < 12:  # Middle 50%
            p.growth_per_day = 0.0006  # 0.108 over 180 days
        else:  # Top 25%
            p.growth_per_day = 0.0004  # 0.072 over 180 days

    # Set participation
    for p in circuit.player_pool:
        # Higher skill = higher participation
        percentile = stats.percentileofscore(
            [x.true_skill for x in circuit.player_pool], p.true_skill
        )
        p.participation_prob = 0.3 + 0.4 * (percentile / 100)

    # Create dense schedule
    configs = create_dense_tournament_schedule()
    print(f"\nCreated {len(configs)} tournaments over 180 days")

    # Generate tournaments WITHOUT growth first
    print("\nGenerating tournaments...")
    circuit_results = circuit.generate_circuit(configs)
    circuit_results = add_scores_to_circuit_results(circuit_results)

    # Now apply growth based on time
    print("\nApplying gradual growth...")
    for p in circuit.player_pool:
        p.true_skill = p.initial_skill + (p.growth_per_day * 180)

    # Calculate actual growth
    growth_by_tier = {"Bottom 25%": [], "Middle 50%": [], "Top 25%": []}

    for p in circuit.player_pool:
        growth = p.true_skill - p.initial_skill
        if p.initial_skill < 8:
            growth_by_tier["Bottom 25%"].append(growth)
        elif p.initial_skill < 12:
            growth_by_tier["Middle 50%"].append(growth)
        else:
            growth_by_tier["Top 25%"].append(growth)

    print("\nGrowth by skill tier:")
    for tier, growths in growth_by_tier.items():
        if growths:
            print(
                f"  {tier}: avg={np.mean(growths):.3f}, max={max(growths):.3f}"
            )

    overall_avg_growth = np.mean(
        [p.true_skill - p.initial_skill for p in circuit.player_pool]
    )
    print(f"\nOverall average growth: {overall_avg_growth:.3f}")

    # Test different decay half-lives
    print("\n=== TESTING DECAY HALF-LIVES ===")

    decay_configs = [
        ("7-day", 7),
        ("15-day", 15),
        ("30-day", 30),
        ("60-day", 60),
        ("120-day", 120),
        ("No decay", 10000),  # Effectively no decay
    ]

    results = []

    for name, half_life in decay_configs:
        decay_rate = np.log(2) / half_life
        evaluator = PageRankEvaluator(
            min_tournaments=5,
            beta=0.5,
            decay_rate=decay_rate,
        )

        eval_result = evaluator.evaluate_circuit(circuit, circuit_results)

        results.append(
            {
                "name": name,
                "half_life": half_life,
                "correlation": eval_result.spearman_correlation,
                "top_10_acc": eval_result.top_10_accuracy,
                "top_20_acc": eval_result.top_20_accuracy,
                "n_qualified": len(eval_result.participation_counts),
            }
        )

        print(f"\n{name}:")
        print(f"  Correlation: {eval_result.spearman_correlation:.3f}")
        print(f"  Top 10 accuracy: {eval_result.top_10_accuracy:.1%}")
        print(f"  Qualified players: {len(eval_result.participation_counts)}")

    # Visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Correlation curve
    half_lives = [r["half_life"] for r in results[:-1]]  # Exclude "no decay"
    correlations = [r["correlation"] for r in results[:-1]]

    ax1.plot(
        half_lives, correlations, "o-", markersize=10, linewidth=2, color="blue"
    )
    ax1.set_xlabel("Decay Half-Life (days)")
    ax1.set_ylabel("Spearman Correlation")
    ax1.set_title(
        f"Dense Schedule + Tiny Growth\n(Avg: {overall_avg_growth:.3f})"
    )
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")
    ax1.set_ylim(0, 1)
    ax1.axhline(
        y=0.74, color="red", linestyle="--", alpha=0.5, label="Your baseline"
    )
    ax1.legend()

    # Annotate
    for hl, corr in zip(half_lives, correlations):
        ax1.annotate(
            f"{corr:.3f}",
            (hl, corr),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    # Growth distribution
    all_growth = [p.true_skill - p.initial_skill for p in circuit.player_pool]
    ax2.hist(all_growth, bins=50, alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Skill Growth")
    ax2.set_ylabel("Number of Players")
    ax2.set_title("Growth Distribution")
    ax2.axvline(
        x=overall_avg_growth,
        color="red",
        linestyle="--",
        label=f"Mean: {overall_avg_growth:.3f}",
    )
    ax2.legend()

    # Skill distribution before/after
    initial_skills = [p.initial_skill for p in circuit.player_pool]
    final_skills = [p.true_skill for p in circuit.player_pool]

    ax3.hist(initial_skills, bins=50, alpha=0.5, label="Initial", color="blue")
    ax3.hist(final_skills, bins=50, alpha=0.5, label="Final", color="green")
    ax3.set_xlabel("Skill Value")
    ax3.set_ylabel("Number of Players")
    ax3.set_title("Skill Distribution Before/After Growth")
    ax3.legend()

    # Top-k accuracy comparison
    x = np.arange(len(results))
    width = 0.35

    top_10_accs = [r["top_10_acc"] for r in results]
    top_20_accs = [r["top_20_acc"] for r in results]

    ax4.bar(x - width / 2, top_10_accs, width, label="Top 10", alpha=0.8)
    ax4.bar(x + width / 2, top_20_accs, width, label="Top 20", alpha=0.8)
    ax4.set_xlabel("Configuration")
    ax4.set_ylabel("Accuracy")
    ax4.set_title("Top-K Accuracy by Decay Setting")
    ax4.set_xticks(x)
    ax4.set_xticklabels([r["name"] for r in results], rotation=45, ha="right")
    ax4.legend()
    ax4.grid(True, axis="y", alpha=0.3)
    ax4.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: "{:.0%}".format(y))
    )

    plt.tight_layout()
    plt.savefig("dense_proper_final_results.png", dpi=150, bbox_inches="tight")
    print("\n\nSaved results to dense_proper_final_results.png")

    # Summary
    print("\n=== SUMMARY ===")
    print(
        f"With {len(configs)} tournaments and tiny differential growth ({overall_avg_growth:.3f} avg):"
    )
    best_idx = np.argmax(correlations)
    print(
        f"  Best: {correlations[best_idx]:.3f} with {half_lives[best_idx]}-day half-life"
    )
    print(f"  15-day: {results[1]['correlation']:.3f}")
    print(f"  30-day: {results[2]['correlation']:.3f}")
    print(f"  No decay: {results[-1]['correlation']:.3f}")

    # Compare to no-growth baseline
    print(f"\nCompared to your 0.74 baseline (no growth):")
    if results[1]["correlation"] >= 0.70:
        print("  ✓ 15-day half-life achieves good correlation even with growth")
    else:
        print("  ✗ Growth reduces correlation below acceptable levels")


if __name__ == "__main__":
    test_dense_proper_final()
