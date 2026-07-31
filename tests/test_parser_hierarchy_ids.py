import pytest

from rankings.core import parse_tournaments_data


def _tournament_payload(hierarchy_ids):
    return {
        "tournament": {
            "ctx": {
                "id": 4022,
                "name": "7 Seas #9",
                "teams": [],
            },
            "data": {
                "stage": [{"id": 6514, "number": 1}],
                "group": [
                    {
                        "id": 17979,
                        "number": 1,
                        **hierarchy_ids["group"],
                    }
                ],
                "round": [
                    {
                        "id": 53402,
                        "number": 1,
                        "maps": {"count": 3, "type": "BEST_OF"},
                        **hierarchy_ids["round"],
                    }
                ],
                "match": [
                    {
                        "id": 143319,
                        "number": 1,
                        "opponent1": {"id": 1},
                        "opponent2": {"id": 2},
                        **hierarchy_ids["match"],
                    }
                ],
            },
        }
    }


@pytest.mark.parametrize(
    "hierarchy_ids",
    [
        {
            "group": {"stageId": 6514},
            "round": {"stageId": 6514, "groupId": 17979},
            "match": {
                "stageId": 6514,
                "groupId": 17979,
                "roundId": 53402,
            },
        },
        {
            "group": {"stage_id": 6514},
            "round": {"stage_id": 6514, "group_id": 17979},
            "match": {
                "stage_id": 6514,
                "group_id": 17979,
                "round_id": 53402,
            },
        },
    ],
    ids=["current-camel-case", "legacy-snake-case"],
)
def test_parse_tournament_hierarchy_ids_supports_current_and_legacy_payloads(
    hierarchy_ids,
):
    tables = parse_tournaments_data([_tournament_payload(hierarchy_ids)])

    groups = tables["groups"]
    rounds = tables["rounds"]
    matches = tables["matches"]
    assert groups is not None
    assert rounds is not None
    assert matches is not None
    assert groups.select("stage_id").item() == 6514
    assert rounds.select("stage_id", "group_id").row(0) == (6514, 17979)
    assert matches.select("stage_id", "group_id", "round_id").row(0) == (
        6514,
        17979,
        53402,
    )
