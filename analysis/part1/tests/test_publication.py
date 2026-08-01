from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


OUTPUT = Path(__file__).resolve().parents[1] / "output"
FIGURE_NAMES = {
    "01_future_prediction",
    "02_loopr_calibration",
    "03_plus_and_shuffle",
    "04_monthly_stability",
    "05_leave_one_tournament_out",
    "A1_half_life_tradeoff",
    "A2_unfamiliar_lineups",
    "N1_rank_bootstrap_intervals",
    "N2_incumbent_autopsy",
    "N3_auc_by_exposure_band",
    "N4_core0_diagnostic",
    "N5_cross_system_event_deletion",
    "N6_ablation_walk",
}


def _json(relative_path: str) -> dict:
    return json.loads((OUTPUT / relative_path).read_text(encoding="utf-8"))


def test_render_contract_and_fixed_palette() -> None:
    validation = _json("results/figure_validation.json")
    manifest = _json("evaluation_manifest.json")

    assert validation["status"] == "pass"
    assert validation["figures"] == 13
    assert {row["figure"] for row in validation["surface_exports"]} == FIGURE_NAMES
    assert {row["surface"] for row in validation["surface_exports"]} == {
        "dark",
        "light",
    }
    assert all(row["png_width"] == 1520 for row in validation["surface_exports"])
    assert all(row["svg_parse_pass"] for row in validation["surface_exports"])
    assert all(row["background_pass"] for row in validation["surface_exports"])
    assert manifest["publication_figures"]["entity_colors"] == {
        "90-day win rate": "#9aa3b2",
        "Bradley-Terry (L2)": "#e0a03d",
        "Chance (0.5)": "#6b7280",
        "LOOPR": "#ec5fa8",
        "OpenSkill mu": "#8fc4ec",
        "OpenSkill ordinal": "#4d9de0",
    }


def test_existing_prediction_point_estimates_are_unchanged() -> None:
    with (OUTPUT / "tables" / "prediction_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = {row["method"]: row for row in csv.DictReader(handle)}

    assert float(rows["LOOPR"]["match_micro_log_loss"]) == pytest.approx(
        0.5129298101302047
    )
    assert float(rows["OpenSkill ordinal"]["match_micro_log_loss"]) == pytest.approx(
        0.486014551165572
    )
    assert all(row["individual_ci_cluster"] == "tournament" for row in rows.values())
    assert all(int(row["individual_ci_repetitions"]) == 5000 for row in rows.values())


def test_rank_bootstrap_and_n2_preregistration_are_auditable() -> None:
    manifest = _json("evaluation_manifest.json")
    rank = _json("results/rank_bootstrap.json")
    autopsy = _json("results/plus_incumbent_autopsy.json")
    preregistration = manifest["plus_vote"]["incumbent_autopsy_preregistration"]

    assert rank["seed"] == 20260731
    assert rank["repetitions"] == 200
    assert rank["resampling_unit"] == "tournament"
    assert preregistration["registered_before_computation"] is True
    assert preregistration["two_sided_expectation"]
    assert autopsy["vote_field_audit"]["vote_percentage_available"] is False
    assert autopsy["preregistered_before_computation_commit"] == "544984b"
