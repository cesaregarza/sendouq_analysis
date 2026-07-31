import pytest

from rankings.core import parse_tournaments_data


def _tournament_payload(tags):
    return {
        "tournament": {
            "ctx": {
                "id": 4022,
                "name": "7 Seas #9",
                "tags": tags,
                "teams": [],
            },
            "data": {
                "stage": [],
                "group": [],
                "round": [],
                "match": [],
            },
        }
    }


@pytest.mark.parametrize(
    ("raw_tags", "expected"),
    [
        (["LOW"], ["LOW"]),
        ([" LOW ", "", "S+"], ["LOW", "S+"]),
        ("LOW, S+", ["LOW", "S+"]),
        ("", None),
        (None, None),
    ],
)
def test_parse_tournament_tags_supports_current_and_legacy_payloads(
    raw_tags, expected
):
    tables = parse_tournaments_data([_tournament_payload(raw_tags)])

    tournaments = tables["tournaments"]
    assert tournaments is not None
    assert tournaments.to_dicts()[0]["tags"] == expected
