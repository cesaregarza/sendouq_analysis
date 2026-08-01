"""Focused invariance and artifact-integrity tests for the intended-use audit."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.part1.intended_use.evaluation import graph_metrics_for_players


OUTPUT = Path(__file__).resolve().parents[1] / "output"
TABLES = OUTPUT / "tables"


def _toy_matches() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"winners": (1,), "losers": (2,)},
            {"winners": (2,), "losers": (3,)},
            {"winners": (3,), "losers": (1,)},
            {"winners": (1,), "losers": (4,)},
        ]
    )


def test_graph_support_ignores_outcome_direction() -> None:
    matches = _toy_matches()
    reversed_matches = matches.rename(
        columns={"winners": "losers", "losers": "winners"}
    )
    columns = [
        "player_id",
        "prior_matches",
        "unique_opponents",
        "effective_opponents",
        "core_number",
        "component_size",
        "nonbridge_edge_share",
        "two_hop_redundancy",
    ]
    observed = graph_metrics_for_players(matches, range(1, 5))[columns]
    reversed_observed = graph_metrics_for_players(reversed_matches, range(1, 5))[
        columns
    ]
    pd.testing.assert_frame_equal(observed, reversed_observed)


def test_k_core_has_expected_topological_meaning() -> None:
    support = graph_metrics_for_players(_toy_matches(), range(1, 5)).set_index(
        "player_id"
    )
    assert set(support.loc[[1, 2, 3], "core_number"]) == {2}
    assert support.loc[4, "core_number"] == 1
    assert support["component_size"].eq(4).all()


def test_disjoint_support_bands_partition_report_matches() -> None:
    bands = pd.read_csv(TABLES / "prediction_by_support_band.csv")
    for method, group in bands.groupby("method"):
        assert group["matches"].sum() == 4_397, method


def test_plus_pair_decomposition_reconstructs_pooled_auc() -> None:
    decomposition = pd.read_csv(TABLES / "plus_pooled_auc_decomposition.csv")
    assert abs(decomposition["share_of_all_passer_failer_pairs"].sum() - 1.0) < 1e-12
    assert (
        abs(decomposition["contribution_to_pooled_auc"].sum() - 0.6509289321789321)
        < 1e-12
    )


def test_attack_targets_are_disjoint_from_each_other_and_opponents() -> None:
    selection = pd.read_parquet(TABLES / "attack_lineup_selection.parquet")
    targets = selection[selection["role"] == "target"]
    opponents = selection[selection["role"].isin(["weak opponent", "strong opponent"])]
    target_players: list[int] = [
        int(player_id) for lineup in targets["lineup"] for player_id in lineup
    ]
    opponent_players = {
        int(player_id) for lineup in opponents["lineup"] for player_id in lineup
    }
    assert len(targets) == 20
    assert len(target_players) == len(set(target_players)) == 80
    assert not (set(target_players) & opponent_players)


def test_every_comparative_profile_has_all_five_systems() -> None:
    expected = {
        "LOOPR",
        "OpenSkill μ",
        "OpenSkill ordinal",
        "Bradley–Terry (L2)",
        "90-day win rate",
    }
    profile = pd.read_csv(TABLES / "requirements_profile.csv")
    assert set(profile["method"]) == expected
    assert profile.notna().all().all()
