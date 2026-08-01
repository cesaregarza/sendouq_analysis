"""Command-line runner for the additive intended-use evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from analysis.part1.intended_use.evaluation import (
    adversarial_thinning_test,
    adversarial_transitive_chain_test,
    adversarial_volume_test,
    adversarial_weak_win_and_schedule_tests,
    audit_historical_seeds,
    comparative_important_tournament_removal,
    comparative_monthly_stability,
    decompose_plus_agreement,
    evaluate_models_against_seeds,
    graph_support_for_prediction_events,
    load_frozen_evaluation,
    prediction_by_graph_support,
    prediction_in_disjoint_support_bands,
    pooled_outcome_shuffle_null,
    summarize_seed_outcome_bands,
    summarize_seed_model_agreement,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _paths(repository_root: Path, output_dir: Path | None) -> tuple[Path, Path]:
    frozen_output = repository_root / "analysis" / "part1" / "output"
    intended_output = (
        output_dir
        if output_dir is not None
        else repository_root / "analysis" / "part1" / "intended_use" / "output"
    )
    intended_output.mkdir(parents=True, exist_ok=True)
    (intended_output / "tables").mkdir(parents=True, exist_ok=True)
    (intended_output / "results").mkdir(parents=True, exist_ok=True)
    return frozen_output, intended_output


def run_support(
    repository_root: Path, source_root: Path, intended_output: Path
) -> None:
    frozen_output = repository_root / "analysis" / "part1" / "output"
    data = load_frozen_evaluation(source_root, frozen_output)
    predictions = pd.read_parquet(
        frozen_output / "tables" / "walk_forward_predictions.parquet"
    )
    player_support, match_support = graph_support_for_prediction_events(
        data, predictions
    )
    player_support.to_parquet(
        intended_output / "tables" / "graph_support_player_tournament.parquet",
        index=False,
    )
    match_support.to_parquet(
        intended_output / "tables" / "graph_support_match.parquet", index=False
    )
    _write_support_summaries(frozen_output, intended_output, match_support)


def _write_support_summaries(
    frozen_output: Path, intended_output: Path, match_support: pd.DataFrame
) -> None:
    player_support_rows = len(
        pd.read_parquet(
            intended_output / "tables" / "graph_support_player_tournament.parquet",
            columns=["player_id"],
        )
    )
    predictions = pd.read_parquet(
        frozen_output / "tables" / "walk_forward_predictions.parquet"
    )
    summary, slopes = prediction_by_graph_support(predictions, match_support)
    bands = prediction_in_disjoint_support_bands(predictions, match_support)
    summary.to_csv(
        intended_output / "tables" / "prediction_by_graph_support.csv", index=False
    )
    slopes.to_csv(
        intended_output / "tables" / "prediction_support_trends.csv", index=False
    )
    bands.to_csv(
        intended_output / "tables" / "prediction_by_support_band.csv", index=False
    )
    _write_json(
        intended_output / "results" / "graph_support.json",
        {
            "player_tournament_rows": int(player_support_rows),
            "match_rows": int(len(match_support)),
            "report_matches": int(match_support["is_report_window"].sum()),
            "supported_thresholds": sorted(
                summary["minimum_core_threshold"].unique().astype(int).tolist()
            ),
        },
    )


def run_support_summary(
    repository_root: Path, source_root: Path, intended_output: Path
) -> None:
    del source_root
    frozen_output = repository_root / "analysis" / "part1" / "output"
    match_support = pd.read_parquet(
        intended_output / "tables" / "graph_support_match.parquet"
    )
    _write_support_summaries(frozen_output, intended_output, match_support)


def run_seed_audit(
    repository_root: Path, source_root: Path, intended_output: Path
) -> None:
    frozen_output = repository_root / "analysis" / "part1" / "output"
    data = load_frozen_evaluation(source_root, frozen_output)
    tournaments, teams, matches, summary = audit_historical_seeds(data)
    tournaments.to_csv(
        intended_output / "tables" / "seed_audit_tournaments.csv", index=False
    )
    teams.to_parquet(
        intended_output / "tables" / "seed_audit_teams.parquet", index=False
    )
    matches.to_csv(intended_output / "tables" / "seed_audit_matches.csv", index=False)
    summarize_seed_outcome_bands(matches, tournaments).to_csv(
        intended_output / "tables" / "seed_outcome_by_band.csv", index=False
    )
    _write_json(intended_output / "results" / "seed_audit.json", summary)


def run_seed_models(
    repository_root: Path, source_root: Path, intended_output: Path
) -> None:
    frozen_output = repository_root / "analysis" / "part1" / "output"
    data = load_frozen_evaluation(source_root, frozen_output)
    tournaments = pd.read_csv(intended_output / "tables" / "seed_audit_tournaments.csv")
    teams = pd.read_parquet(intended_output / "tables" / "seed_audit_teams.parquet")
    tournament_results, pairs, disagreements = evaluate_models_against_seeds(
        data, tournaments, teams
    )
    tournament_results.to_csv(
        intended_output / "tables" / "seed_model_tournament_results.csv",
        index=False,
    )
    pairs.to_parquet(
        intended_output / "tables" / "seed_model_pairs.parquet", index=False
    )
    disagreements.to_parquet(
        intended_output / "tables" / "seed_model_disagreements.parquet",
        index=False,
    )
    summary, disagreement_summary = summarize_seed_model_agreement(
        tournament_results, pairs, disagreements
    )
    summary.to_csv(intended_output / "tables" / "seed_model_summary.csv", index=False)
    disagreement_summary.to_csv(
        intended_output / "tables" / "seed_disagreement_summary.csv", index=False
    )


def run_plus(repository_root: Path, source_root: Path, intended_output: Path) -> None:
    frozen_output = repository_root / "analysis" / "part1" / "output"
    data = load_frozen_evaluation(source_root, frozen_output)
    plus = pd.read_parquet(frozen_output / "tables" / "plus_common_candidates.parquet")
    candidates, pairs, decomposition, baselines = decompose_plus_agreement(data, plus)
    candidates.to_parquet(
        intended_output / "tables" / "plus_candidates_with_support.parquet",
        index=False,
    )
    pairs.to_parquet(
        intended_output / "tables" / "plus_passer_failer_pairs.parquet", index=False
    )
    decomposition.to_csv(
        intended_output / "tables" / "plus_pooled_auc_decomposition.csv",
        index=False,
    )
    baselines.to_csv(
        intended_output / "tables" / "plus_baselines_and_support.csv", index=False
    )


def run_plus_shuffle(
    repository_root: Path, source_root: Path, intended_output: Path
) -> None:
    frozen_output = repository_root / "analysis" / "part1" / "output"
    data = load_frozen_evaluation(source_root, frozen_output)
    plus = pd.read_parquet(frozen_output / "tables" / "plus_common_candidates.parquet")
    draws, summary = pooled_outcome_shuffle_null(
        data,
        plus,
        checkpoint_path=intended_output / "cache" / "pooled_outcome_shuffle.csv",
    )
    draws.to_csv(
        intended_output / "tables" / "pooled_outcome_shuffle_null.csv", index=False
    )
    _write_json(intended_output / "results" / "pooled_outcome_shuffle.json", summary)


def run_stability(
    repository_root: Path, source_root: Path, intended_output: Path
) -> None:
    frozen_output = repository_root / "analysis" / "part1" / "output"
    data = load_frozen_evaluation(source_root, frozen_output)
    snapshots, transitions = comparative_monthly_stability(data)
    snapshots.to_parquet(
        intended_output / "tables" / "comparative_monthly_top100.parquet",
        index=False,
    )
    transitions.to_csv(
        intended_output / "tables" / "comparative_monthly_stability.csv",
        index=False,
    )


def run_removal(
    repository_root: Path, source_root: Path, intended_output: Path
) -> None:
    frozen_output = repository_root / "analysis" / "part1" / "output"
    data = load_frozen_evaluation(source_root, frozen_output)
    events, results = comparative_important_tournament_removal(data)
    events.to_csv(intended_output / "tables" / "important_tournaments.csv", index=False)
    results.to_csv(
        intended_output / "tables" / "comparative_tournament_removal.csv",
        index=False,
    )


def _write_attack_selection(intended_output: Path, selection: pd.DataFrame) -> None:
    selection.to_parquet(
        intended_output / "tables" / "attack_lineup_selection.parquet", index=False
    )


def run_attacks_farming(
    repository_root: Path, source_root: Path, intended_output: Path
) -> None:
    frozen_output = repository_root / "analysis" / "part1" / "output"
    data = load_frozen_evaluation(source_root, frozen_output)
    selection, player_results, schedule_summary = (
        adversarial_weak_win_and_schedule_tests(data)
    )
    _write_attack_selection(intended_output, selection)
    player_results.to_parquet(
        intended_output / "tables" / "farming_attack_player_results.parquet",
        index=False,
    )
    schedule_summary.to_csv(
        intended_output / "tables" / "schedule_attack_summary.csv", index=False
    )


def run_attacks_volume(
    repository_root: Path, source_root: Path, intended_output: Path
) -> None:
    frozen_output = repository_root / "analysis" / "part1" / "output"
    data = load_frozen_evaluation(source_root, frozen_output)
    selection, results = adversarial_volume_test(data)
    _write_attack_selection(intended_output, selection)
    results.to_parquet(
        intended_output / "tables" / "volume_attack_player_results.parquet",
        index=False,
    )


def run_attacks_thinning(
    repository_root: Path, source_root: Path, intended_output: Path
) -> None:
    frozen_output = repository_root / "analysis" / "part1" / "output"
    data = load_frozen_evaluation(source_root, frozen_output)
    selection, results = adversarial_thinning_test(data)
    _write_attack_selection(intended_output, selection)
    results.to_parquet(
        intended_output / "tables" / "thinning_attack_player_results.parquet",
        index=False,
    )


def run_attacks_transitive(
    repository_root: Path, source_root: Path, intended_output: Path
) -> None:
    frozen_output = repository_root / "analysis" / "part1" / "output"
    data = load_frozen_evaluation(source_root, frozen_output)
    selection, results = adversarial_transitive_chain_test(data)
    _write_attack_selection(intended_output, selection)
    results.to_csv(
        intended_output / "tables" / "transitive_chain_results.csv", index=False
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "support",
            "support-summary",
            "seed-audit",
            "seed-models",
            "plus",
            "plus-shuffle",
            "stability",
            "removal",
            "attacks-farming",
            "attacks-volume",
            "attacks-thinning",
            "attacks-transitive",
            "all",
        ],
    )
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Native-Linux source checkout containing data/tournaments; defaults to repository root",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repository_root = args.repository_root.resolve()
    source_root = (
        args.source_root.resolve() if args.source_root is not None else repository_root
    )
    _, intended_output = _paths(repository_root, args.output_dir)
    stages = (
        [
            "support",
            "seed-audit",
            "seed-models",
            "plus",
            "plus-shuffle",
            "stability",
            "removal",
            "attacks-farming",
            "attacks-volume",
            "attacks-thinning",
            "attacks-transitive",
        ]
        if args.stage == "all"
        else [args.stage]
    )
    runners = {
        "support": run_support,
        "support-summary": run_support_summary,
        "seed-audit": run_seed_audit,
        "seed-models": run_seed_models,
        "plus": run_plus,
        "plus-shuffle": run_plus_shuffle,
        "stability": run_stability,
        "removal": run_removal,
        "attacks-farming": run_attacks_farming,
        "attacks-volume": run_attacks_volume,
        "attacks-thinning": run_attacks_thinning,
        "attacks-transitive": run_attacks_transitive,
    }
    for stage in stages:
        runners[stage](repository_root, source_root, intended_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
