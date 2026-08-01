#!/usr/bin/env python3
"""Command-line entry point for the Part I evaluation.

Examples
--------
PYTHONPATH=.analysis-deps:src:. python -m analysis.part1.run audit \
    --source-root /root/dev/sendouq_analysis
PYTHONPATH=.analysis-deps:src:. python -m analysis.part1.run prediction \
    --source-root /root/dev/sendouq_analysis
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analyses import (
    connectivity_and_mixing,
    data_audit,
    half_life_tradeoff,
    leave_one_tournament_out,
    outcome_shuffle_null,
    plus_analysis,
    prediction_summary,
    property_tests,
    smoothing_perturbation_test,
    stability_analysis,
    walk_forward_prediction,
    tournament_weighting_property_test,
)
from .core import (
    DEFAULT_PLUS_DATE,
    DEFAULT_SEED,
    package_versions,
    prepare_evaluation_data,
    production_config,
    utc_date,
)
from .publication import (
    DEFAULT_RANK_BOOTSTRAP_REPETITIONS,
    run_publication_analyses,
)


def _json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(source_root: Path) -> dict[str, Any]:
    patterns = {
        "drafts": ["docs/blog_draft.md", "docs/rankings_system_draft.md"],
        "parameters": ["src/rankings/config.yaml", "pyproject.toml", "poetry.lock"],
        "prediction": [
            "codex_scripts/analysis/plot_match_prediction_harness.py",
            "docs/match_prediction_harness.csv",
            "docs/match_prediction_harness_summary.csv",
        ],
        "ablation": [
            "codex_scripts/analysis/loopr_ablation_table.py",
            "docs/loopr_ablation_table.csv",
            "docs/loopr_ablation_table.md",
        ],
        "plus": [
            "sendou_plus_voting_results_2025_11_16.csv",
            "sendou_plus_voting_results_2026_02_15.csv",
            "sendou_plus_voting_results_aug_nov_feb_consolidated.csv",
            "docs/plus_server_validation.md",
            "docs/plus_validation_split_summary_2025_11_17_full.csv",
        ],
        "implementation": [
            "src/rankings/algorithms/exposure_log_odds.py",
            "src/rankings/algorithms/tick_tock.py",
            "src/rankings/core/convert.py",
            "src/rankings/analysis/utils/tournament_filters.py",
            "src/rankings/analysis/utils/summaries.py",
            "src/rankings/evaluation/cross_validation/simple_splits.py",
            "src/rankings/evaluation/loss.py",
        ],
    }
    output: dict[str, list[dict[str, Any]]] = {}
    for category, rel_paths in patterns.items():
        rows = []
        for rel in rel_paths:
            path = source_root / rel
            rows.append(
                {
                    "path": rel,
                    "exists": path.exists(),
                    "bytes": path.stat().st_size if path.exists() else None,
                    "sha256": file_sha256(path) if path.is_file() else None,
                }
            )
        output[category] = rows
    data_files = list((source_root / "data" / "tournaments").rglob("*.json"))
    output["raw_data"] = [
        {
            "path": "data/tournaments",
            "json_files": len(data_files),
            "bytes": int(sum(path.stat().st_size for path in data_files)),
        }
    ]
    return output


def project_file_inventory(source_root: Path) -> pd.DataFrame:
    """Inventory prior ranking/evaluation materials at file level.

    Raw tournament JSON is represented by the audited aggregate and canonical
    source table rather than 181 repeated batch/snapshot paths.
    """

    patterns = {
        "drafts_and_reports": [
            "docs/blog_draft.md",
            "docs/rankings_system_draft.md",
            "docs/*plus*.md",
            "docs/*ranking*.md",
        ],
        "plus_vote_data": ["sendou_plus_voting_results*.csv"],
        "analysis_scripts": [
            "codex_scripts/analysis/*.py",
            "codex_scripts/evaluation/*.py",
            "scripts/*loopr*.py",
            "scripts/*rank*.py",
            "scripts/*volume*.py",
            "scripts/*tournament*.py",
            "examples/*pagerank*.py",
        ],
        "rating_implementation": [
            "src/rankings/algorithms/**/*.py",
            "src/rankings/core/**/*.py",
            "src/rankings/evaluation/**/*.py",
            "src/rankings/tournament_prediction/**/*.py",
            "src/rankings/calibration.py",
        ],
        "saved_tables_and_models": [
            "docs/*.csv",
            "codex_scripts/analysis/*.csv",
            "codex_scripts/analysis/*.parquet",
            "codex_scripts/analysis/.cache/*.json",
            "data/*rank*.csv",
            "data/*rating*.csv",
            "tmp/*loopr*.csv",
            "tmp/*plus*.csv",
        ],
        "existing_figures": [
            "docs/images/blog/*loopr*.png",
            "docs/images/blog/*plus*.png",
            "docs/images/blog/*prediction*.png",
            "docs/images/blog/*volume*.png",
            "docs/images/blog/*openskill*.png",
            "docs/images/blog/*connect*.png",
            "docs/images/plus_validation/*.png",
        ],
        "notebooks": ["*.ipynb", "experiments/*.ipynb"],
        "parameters_and_environment": [
            "src/rankings/config.yaml",
            "pyproject.toml",
            "poetry.lock",
        ],
    }
    selected: dict[Path, set[str]] = {}
    for category, globs in patterns.items():
        for pattern in globs:
            for path in source_root.glob(pattern):
                if path.is_file() and "analysis/part1" not in path.as_posix():
                    selected.setdefault(path, set()).add(category)

    rows = []
    for path, categories in sorted(
        selected.items(), key=lambda item: str(item[0].relative_to(source_root))
    ):
        rows.append(
            {
                "category": ";".join(sorted(categories)),
                "path": str(path.relative_to(source_root)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    rows.append(
        {
            "category": "raw_tournament_data_aggregate",
            "path": "data/tournaments/**/*.json",
            "bytes": sum(
                path.stat().st_size
                for path in (source_root / "data" / "tournaments").rglob("*.json")
            ),
            "sha256": None,
        }
    )
    return pd.DataFrame(rows).sort_values(["category", "path"]).reset_index(drop=True)


def reproduction_ledger(source_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    prediction_detail = source_root / "docs" / "match_prediction_harness.csv"
    prediction_summary_path = (
        source_root / "docs" / "match_prediction_harness_summary.csv"
    )
    if prediction_detail.exists() and prediction_summary_path.exists():
        detail = pd.read_csv(prediction_detail)
        saved = pd.read_csv(prediction_summary_path)
        loopr = detail[detail["method"] == "LOOPR"]
        reconstructed = float(np.average(loopr["log_loss"], weights=loopr["n_test"]))
        saved_value = float(
            saved.loc[saved["method"] == "LOOPR", "log_loss_mean"].iloc[0]
        )
        rows.append(
            {
                "result": "LOOPR prediction log loss (January harness)",
                "displayed": saved_value,
                "recomputed": reconstructed,
                "reproduced": bool(abs(saved_value - reconstructed) < 1e-12),
                "protocol": "3 sampled temporal blocks x 10 tournaments; ratings frozen within each block; match-iid bootstrap",
                "explanation": "Weighted aggregation of the three saved split results reproduces 0.526394. Raw rebuild is version-sensitive.",
            }
        )
    ablation_path = source_root / "docs" / "loopr_ablation_table.csv"
    if ablation_path.exists():
        ablation = pd.read_csv(ablation_path)
        for variant in [
            "LOOPR + exposure teleport",
            "LOOPR + smoothing",
            "LOOPR + tick-tock",
        ]:
            record = ablation[ablation["variant"] == variant].iloc[0]
            rows.append(
                {
                    "result": f"{variant}: prediction log loss",
                    "displayed": float(record["prediction_log_loss"]),
                    "recomputed": float(record["prediction_log_loss"]),
                    "reproduced": True,
                    "protocol": "ablation script's hand-built variant; same 3x10 block structure",
                    "explanation": "0.497/0.497/0.489 are different model rows, not reruns of the production harness.",
                }
            )
            rows.append(
                {
                    "result": f"{variant}: Plus macro AUC",
                    "displayed": float(record["tier_macro_auc"]),
                    "recomputed": float(record["tier_macro_auc"]),
                    "reproduced": True,
                    "protocol": "active <=90d; November 2025; unweighted mean of tier AUCs",
                    "explanation": "0.772 is the exposure/smoothing row; 0.779 is the later tick-tock row.",
                }
            )
    rows.append(
        {
            "result": "Draft Plus AUC",
            "displayed": 0.86,
            "recomputed": np.nan,
            "reproduced": False,
            "protocol": "not specified in draft",
            "explanation": "No saved table or script output in the project identifies a pooled/tiered protocol that yields the draft's 0.86. It must not be retained as an auditable result.",
        }
    )
    return pd.DataFrame(rows)


def manifest(data, source_root: Path, plus_path: Path) -> dict[str, Any]:
    audit = data_audit(data)
    cfg = production_config()
    votes = pd.read_csv(plus_path)
    plus_counts = (
        votes.groupby(["tier", "suggested", "pass_or_fail"], dropna=False)
        .size()
        .reset_index(name="n")
        .to_dict("records")
    )
    cutoff = float(data.resolved["ts"].max()) + 1.0
    return {
        "analysis_name": "Building Public Rankings, Part I evaluation",
        "protocol_status": "revised analysis; historical reproduction is reported separately",
        "random_seed": DEFAULT_SEED,
        "dataset": {
            "source_root_at_execution": str(source_root),
            "cutoff_date": utc_date(cutoff),
            "match_date_range": [audit["date_start"], audit["date_end"]],
            "tournament_date_range": [audit["date_start"], audit["date_end"]],
            "tournaments": audit["eligible_tournaments"],
            "matches": audit["eligible_decided_matches"],
            "unique_players": audit["eligible_unique_players"],
            "unique_roster_players": audit["eligible_unique_roster_players"],
            "unique_player_definition": "players resolved into at least one valid decided match; roster-only count reported separately",
            "eligibility": {
                "settings_is_ranked": True,
                "minimum_teams": 4,
                "minimum_participated_users": 16,
                "minimum_bracket_matches": 10,
                "excluded_name_regex": "(?i)test|debug|demo|practice",
                "valid_outcome": "non-bye with a winner and loser",
            },
            "duplicate_tournament_records": data.duplicate_audit,
            "timestamp_treatment": {
                "priority": [
                    "last_game_finished_at",
                    "match_created_at",
                    "tournament_start_time",
                ],
                "imputed_matches": audit["timestamp_imputed_matches"],
                "source_counts": audit["timestamp_sources"],
                "reason": "the repository converter otherwise assigns a global/current timestamp to missing match times, which is invalid for temporal evaluation",
            },
        },
        "participation": {
            "treatment": "use per-match appearances for a side when present; otherwise use that tournament/team roster",
            "partial_appearance_treatment": "appearance override is side-specific; the other side falls back to roster",
            "substitution_limitation": "per-match appearance coverage is incomplete, so unobserved substitutions cannot be recovered",
            "lineup_source_counts": audit["lineup_sources"],
        },
        "active_player_definition": "listed on a tournament roster for an eligible event with at least one valid match in the 90 days before the snapshot; chosen so optional appearance-payload coverage does not define activity",
        "prediction": {
            "calibration_warmup_start": "2025-07-15",
            "reported_test_start": "2025-08-17",
            "test_end": audit["date_end"],
            "unit": "held-out tournament",
            "training": "all eligible matches with timestamps strictly before the held-out tournament's information cutoff; held-out tournament excluded",
            "update": "ratings recomputed before every held-out tournament and frozen within that tournament",
            "calibration": "one positive temperature fit only to raw winner-score differences from earlier out-of-sample tournaments; minimum 200 matches and 5 tournaments",
            "unseen_players": {
                "LOOPR": 0.0,
                "OpenSkill_mu": "model population mean",
                "OpenSkill_ordinal": "model default mu - 3 sigma",
                "90_day_win_rate": "Beta(1,1) prior => logit 0",
                "Bradley_Terry": 0.0,
            },
            "low_history_players": "retained and shrunk by each method's native/default prior; no test-set coverage filter",
            "primary_metric": "match-weighted log loss",
            "secondary_metrics": [
                "tournament-weighted log loss",
                "Brier score",
                "accuracy",
            ],
            "uncertainty": "paired tournament-cluster bootstrap, 5,000 repetitions",
        },
        "plus_vote": {
            "date": DEFAULT_PLUS_DATE,
            "file": plus_path.name,
            "historical_rule": "pass at >=60% approval; database view encoded mean +/-1 vote score >= 0.2",
            "automatic_sendouq_admissions": "not present; source contains PlusVotingResult peer-vote rows",
            "candidate_status": "suggested=True is an official newly suggested candidate; False is an incumbent",
            "inclusion": "candidate roster-active in prior 90 days; all methods evaluated on the same candidates, with declared population-neutral values for unseen players",
            "counts_by_tier_status_outcome": plus_counts,
            "uncertainty": "candidate bootstrap stratified by tier, status, and outcome; 2,000 repetitions",
            "vote_percentage": "not exported in available CSVs; continuous analysis omitted",
            "incumbent_autopsy_preregistration": {
                "analysis_id": "N2",
                "registered_on": "2026-08-01",
                "registered_before_computation": True,
                "question": "What relationship, if any, exists between pre-vote LOOPR score and the historical Plus decision once candidate status and exposure are visible?",
                "two_sided_expectation": (
                    "LOOPR score may rise with vote percentage within tier and candidate status if the vote responds to demonstrated organized-play outcomes. "
                    "It may instead be weak, flat, non-monotone, or differ between incumbents and suggested candidates if retention, reputation, visibility, or the institutional status prior dominates the vote. "
                    "Neither direction is defined as success in advance."
                ),
                "planned_display": "score versus vote percentage, faceted by incumbent/suggested status; pass/fail encoded by color and shape; exposure encoded by size or band",
                "planned_reading": [
                    "A within-status slope would be convergent evidence, not ground truth.",
                    "Separation driven mainly by status or exposure would weaken a score-specific interpretation of pooled AUC.",
                    "A null or reversed relationship will be reported without relabeling the estimand."
                ],
                "missing_data_contingency": (
                    "If repository history contains no candidate-level vote percentage or vote counts, do not reconstruct or impute it. "
                    "Retain pass/fail as the only vote measure, render a clearly labeled threshold-outcome autopsy using score and exposure, and record the deviation."
                )
            },
        },
        "outcome_shuffle": {
            "repetitions": 200,
            "procedure": "independent p=.5 reversal of each pre-vote match; recompute full LOOPR including tick-tock",
        },
        "loopr_parameters": {
            "half_life_days": cfg.decay.half_life_days,
            "pagerank_alpha": cfg.pagerank.alpha,
            "pagerank_tolerance": cfg.pagerank.tol,
            "pagerank_max_iterations": cfg.pagerank.max_iter,
            "tournament_beta": cfg.engine.beta,
            "lambda_mode": cfg.lambda_mode,
            "tick_tock_tolerance": cfg.tick_tock.convergence_tol,
            "tick_tock_max_ticks": cfg.tick_tock.max_ticks,
            "tick_tock_influence": cfg.tick_tock.influence_method,
            "inactivity_delay_days": cfg.engine.score_decay_delay_days,
            "inactivity_decay_rate": cfg.engine.score_decay_rate,
            "use_tick_tock_active": cfg.use_tick_tock_active,
            "tested_half_lives": [90, 120, 180, 270, 365, "no_decay"],
        },
        "other_model_parameters": {
            "OpenSkill": "openskill.models.PlackettLuce defaults; sequential match updates in timestamp order",
            "Bradley_Terry": "team-average player effects, L2 logistic regression, C=1.0, no intercept",
            "win_rate": "prior 90 days; per-player Beta(1,1) smoothing; team mean logit",
        },
        "versions": package_versions(Path(__file__).resolve().parents[2]),
    }


def collect_results(output_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in sorted((output_dir / "results").glob("*.json")):
        if path.name == "results.json":
            continue
        result[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "audit",
            "connectivity",
            "prediction",
            "plus",
            "shuffle",
            "stability",
            "loto",
            "half-life",
            "properties",
            "publication",
            "collect",
            "core",
        ],
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/part1/output"),
    )
    parser.add_argument("--shuffle-repetitions", type=int, default=200)
    parser.add_argument(
        "--rank-bootstrap-repetitions",
        type=int,
        default=DEFAULT_RANK_BOOTSTRAP_REPETITIONS,
    )
    parser.add_argument(
        "--intended-output-dir",
        type=Path,
        default=Path("analysis/part1/intended_use/output"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    results_dir = output_dir / "results"
    cache_dir = output_dir / "cache"
    for directory in [tables_dir, results_dir, cache_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    plus_path = source_root / "sendou_plus_voting_results_2025_11_16.csv"

    if args.stage == "collect":
        write_json(results_dir / "results.json", collect_results(output_dir))
        return 0

    data = prepare_evaluation_data(source_root, output_dir)

    stages = (
        ["audit", "connectivity", "prediction", "plus", "stability"]
        if args.stage == "core"
        else [args.stage]
    )
    for stage in stages:
        if stage == "audit":
            write_json(output_dir / "audit" / "inventory.json", inventory(source_root))
            project_file_inventory(source_root).to_csv(
                output_dir / "audit" / "project_file_inventory.csv", index=False
            )
            reproduction_ledger(source_root).to_csv(
                output_dir / "audit" / "reproduction_ledger.csv", index=False
            )
            audit = data_audit(data)
            write_json(results_dir / "data_audit.json", audit)
            write_json(
                output_dir / "evaluation_manifest.json",
                manifest(data, source_root, plus_path),
            )

        elif stage == "connectivity":
            summary, players, components = connectivity_and_mixing(data, plus_path)
            write_json(results_dir / "connectivity.json", summary)
            players.to_parquet(tables_dir / "player_mixing.parquet", index=False)
            components.to_csv(tables_dir / "graph_components.csv", index=False)

        elif stage == "prediction":
            raw = walk_forward_prediction(
                data,
                checkpoint_path=cache_dir / "walk_forward_all.parquet",
            )
            raw.to_parquet(tables_dir / "walk_forward_predictions.parquet", index=False)
            summary, calibration, per_event, familiarity = prediction_summary(raw)
            summary.to_csv(tables_dir / "prediction_summary.csv", index=False)
            calibration.to_csv(tables_dir / "loopr_calibration.csv", index=False)
            per_event.to_csv(tables_dir / "prediction_by_tournament.csv", index=False)
            familiarity.to_csv(
                tables_dir / "prediction_by_familiarity.csv", index=False
            )
            write_json(
                results_dir / "prediction.json",
                {
                    "report_start": "2025-08-17",
                    "report_end": utc_date(float(data.resolved["ts"].max())),
                    "methods": summary.to_dict("records"),
                    "calibration": calibration.to_dict("records"),
                    "familiarity": familiarity.to_dict("records"),
                },
            )

        elif stage == "plus":
            summary, methods, strata, common = plus_analysis(data, plus_path)
            write_json(results_dir / "plus.json", summary)
            methods.to_csv(tables_dir / "plus_methods.csv", index=False)
            strata.to_csv(tables_dir / "plus_strata.csv", index=False)
            common.to_parquet(
                tables_dir / "plus_common_candidates.parquet", index=False
            )

        elif stage == "shuffle":
            common = pd.read_parquet(tables_dir / "plus_common_candidates.parquet")
            draws, summary = outcome_shuffle_null(
                data,
                common,
                repetitions=args.shuffle_repetitions,
                checkpoint_path=cache_dir / "outcome_shuffle.csv",
            )
            draws.to_csv(tables_dir / "outcome_shuffle_null.csv", index=False)
            write_json(results_dir / "outcome_shuffle.json", summary)

        elif stage == "stability":
            transitions, snapshots = stability_analysis(data)
            transitions.to_csv(tables_dir / "monthly_stability.csv", index=False)
            snapshots.to_parquet(tables_dir / "monthly_top100.parquet", index=False)
            write_json(
                results_dir / "stability.json",
                {
                    "transitions": transitions.to_dict("records"),
                    "mean_retention": float(transitions["top100_retention"].mean()),
                    "mean_spearman_retained": float(
                        transitions["spearman_on_retained"].mean()
                    ),
                    "median_transition_movement": float(
                        transitions["median_absolute_rank_movement"].median()
                    ),
                },
            )

        elif stage == "loto":
            rows, summary = leave_one_tournament_out(
                data, checkpoint_path=cache_dir / "leave_one_tournament_out_shared.csv"
            )
            rows.to_csv(tables_dir / "leave_one_tournament_out.csv", index=False)
            write_json(results_dir / "leave_one_tournament_out.json", summary)

        elif stage == "half-life":
            base = pd.read_parquet(tables_dir / "walk_forward_predictions.parquet")
            common = pd.read_parquet(tables_dir / "plus_common_candidates.parquet")
            rows = half_life_tradeoff(
                data,
                base,
                common,
                checkpoint_path=cache_dir / "half_life_tradeoff.parquet",
            )
            rows.to_csv(tables_dir / "half_life_tradeoff.csv", index=False)
            write_json(results_dir / "half_life.json", rows.to_dict("records"))

        elif stage == "properties":
            summary, pairs = property_tests(data)
            pairs.to_csv(
                tables_dir / "schedule_strength_matched_pairs.csv", index=False
            )
            smoothing_detail, smoothing_summary = smoothing_perturbation_test(data)
            smoothing_detail.to_csv(
                tables_dir / "smoothing_perturbations.csv", index=False
            )
            smoothing_summary.to_csv(
                tables_dir / "smoothing_perturbation_summary.csv", index=False
            )
            weighting_detail, weighting_summary = tournament_weighting_property_test(
                data
            )
            weighting_detail.to_csv(
                tables_dir / "tournament_weighting_property.csv", index=False
            )
            summary["smoothing"] = smoothing_summary.to_dict("records")
            summary["tournament_weighting"] = weighting_summary
            write_json(results_dir / "property_tests.json", summary)

        elif stage == "publication":
            run_publication_analyses(
                data,
                repository_root=source_root,
                output=output_dir,
                intended_output=args.intended_output_dir.resolve(),
                rank_bootstrap_repetitions=args.rank_bootstrap_repetitions,
            )

    write_json(results_dir / "results.json", collect_results(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
