from __future__ import annotations

from copy import deepcopy

import pytest

import rankings.cli.update as update_cli
import rankings.scraping.api as scraping_api
from rankings.scraping.turbo_stream import (
    TOURNAMENT_BRACKET_ROUTE_KEY,
    extract_tournament_bracket_data,
)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class _Session:
    def __init__(self, responses: dict[str, object]):
        self.responses = responses
        self.urls: list[str] = []

    def get(self, url: str, timeout: float):
        self.urls.append(url)
        return _Response(self.responses[url])


def _encode_turbo_route(route_key: str, route_data: dict) -> list:
    data: list = [{}]

    def add(value) -> int:
        idx = len(data)
        data.append(None)
        if isinstance(value, dict):
            encoded = {}
            for key, item in value.items():
                key_idx = add(key)
                item_idx = add(item)
                encoded[f"_{key_idx}"] = item_idx
            data[idx] = encoded
        elif isinstance(value, list):
            data[idx] = [add(item) for item in value]
        else:
            data[idx] = value
        return idx

    route_idx = add(route_key)
    value_idx = add({"data": route_data})
    data[0] = {f"_{route_idx}": value_idx}
    return data


def test_extract_tournament_bracket_data_decodes_route_payload():
    route_data = {
        "bracketIdx": 0,
        "bracket": {
            "data": {"stage": [], "group": [], "round": [], "match": []}
        },
    }

    payload = _encode_turbo_route(TOURNAMENT_BRACKET_ROUTE_KEY, route_data)

    assert extract_tournament_bracket_data(payload) == route_data


def test_scrape_tournament_reconstructs_split_payload_and_full_rosters(
    monkeypatch,
):
    tournament_id = 4212
    layout = {
        "tournament": {
            "ctx": {
                "id": tournament_id,
                "name": "Craft Testers",
                "startsAt": 1_785_888_000,
                "teams": [
                    {
                        "id": 84167,
                        "name": "STARCRASHER",
                        "seed": 1,
                        "memberUserIds": [50551],
                    }
                ],
            },
            "bracketsMeta": [{"id": 6591}, {"id": 6592}],
        }
    }
    bracket_payloads = [
        {
            "bracketIdx": 0,
            "bracket": {
                "data": {
                    "stage": [{"id": 1}],
                    "group": [{"id": 2}],
                    "round": [{"id": 3}],
                    "match": [{"id": 144654}],
                }
            },
        },
        {
            "bracketIdx": 1,
            "bracket": {
                "data": {
                    "stage": [{"id": 4}],
                    "group": [{"id": 5}],
                    "round": [{"id": 6}],
                    "match": [{"id": 144655}],
                }
            },
        },
    ]
    results_url = scraping_api.build_tournament_url(tournament_id)
    bracket_urls = [
        scraping_api.build_tournament_bracket_url(tournament_id, idx)
        for idx in range(2)
    ]
    session = _Session(
        {
            results_url: {"layout": layout},
            bracket_urls[0]: {"route": bracket_payloads[0]},
            bracket_urls[1]: {"route": bracket_payloads[1]},
        }
    )
    monkeypatch.setattr(
        scraping_api,
        "extract_tournament_data",
        lambda payload: deepcopy(payload["layout"]),
    )
    monkeypatch.setattr(
        scraping_api,
        "extract_tournament_bracket_data",
        lambda payload: payload["route"],
    )
    monkeypatch.setattr(
        scraping_api,
        "fetch_tournament_teams",
        lambda _tournament_id: [
            {
                "id": 84167,
                "name": "STARCRASHER",
                "members": [
                    {
                        "userId": 50551,
                        "name": "howl",
                        "country": "US",
                        "captain": True,
                    }
                ],
            }
        ],
    )

    result = scraping_api.scrape_tournament(
        tournament_id, max_retries=1, session=session
    )

    tournament = result["tournament"]
    assert tournament["ctx"]["startTime"] == 1_785_888_000
    assert tournament["ctx"]["teams"][0]["seed"] == 1
    assert tournament["ctx"]["teams"][0]["members"] == [
        {
            "userId": 50551,
            "username": "howl",
            "discordId": None,
            "discordAvatar": None,
            "customUrl": None,
            "country": "US",
            "twitch": None,
            "plusTier": None,
            "isOwner": 1,
            "createdAt": None,
            "inGameName": None,
        }
    ]
    assert [match["id"] for match in tournament["data"]["match"]] == [
        144654,
        144655,
    ]
    assert session.urls == [results_url, *bracket_urls]


def test_scrape_tournament_skips_unmaterialized_preview_bracket(monkeypatch):
    tournament_id = 4289
    layout = {
        "tournament": {
            "ctx": {
                "id": tournament_id,
                "name": "Moo Moo Cup #8",
                "teams": [{"id": 1, "members": []}],
            },
            "bracketsMeta": [
                {"id": 6739, "preview": False},
                {"id": -1, "preview": True},
            ],
        }
    }
    bracket_payload = {
        "bracketIdx": 0,
        "bracket": {
            "data": {
                "stage": [{"id": 1}],
                "group": [{"id": 2}],
                "round": [{"id": 3}],
                "match": [{"id": 147887}],
            }
        },
    }
    results_url = scraping_api.build_tournament_url(tournament_id)
    bracket_url = scraping_api.build_tournament_bracket_url(tournament_id, 0)
    session = _Session(
        {
            results_url: {"layout": layout},
            bracket_url: {"route": bracket_payload},
        }
    )
    monkeypatch.setattr(
        scraping_api,
        "extract_tournament_data",
        lambda payload: deepcopy(payload["layout"]),
    )
    monkeypatch.setattr(
        scraping_api,
        "extract_tournament_bracket_data",
        lambda payload: payload["route"],
    )

    result = scraping_api.scrape_tournament(
        tournament_id, max_retries=1, session=session
    )

    assert result["tournament"]["data"]["match"] == [{"id": 147887}]
    assert session.urls == [results_url, bracket_url]


def test_scrape_tournament_preserves_legacy_complete_payload(monkeypatch):
    tournament_id = 4000
    full_payload = {
        "tournament": {
            "ctx": {
                "id": tournament_id,
                "name": "Existing format",
                "startTime": 1_700_000_000,
                "teams": [{"id": 1, "members": []}],
            },
            "data": {"match": [{"id": 10}]},
        }
    }
    results_url = scraping_api.build_tournament_url(tournament_id)
    session = _Session({results_url: full_payload})
    monkeypatch.setattr(
        scraping_api, "extract_tournament_data", lambda payload: payload
    )
    monkeypatch.setattr(
        scraping_api,
        "fetch_tournament_teams",
        lambda _tournament_id: pytest.fail(
            "full rosters should not be fetched"
        ),
    )

    result = scraping_api.scrape_tournament(
        tournament_id, max_retries=1, session=session
    )

    assert result == full_payload
    assert session.urls == [results_url]


def test_scrape_tournament_reports_endpoint_failure_reasons(monkeypatch):
    tournament_id = 4212
    results_url = scraping_api.build_tournament_url(tournament_id)
    legacy_url = scraping_api.build_tournament_url(tournament_id, legacy=True)
    session = _Session(
        {
            results_url: {"layout": {"tournament": {"ctx": {}}}},
            legacy_url: ValueError("HTML response is not JSON"),
        }
    )
    monkeypatch.setattr(
        scraping_api,
        "extract_tournament_data",
        lambda payload: payload["layout"],
    )

    with pytest.raises(RuntimeError) as exc_info:
        scraping_api.scrape_tournament(
            tournament_id, max_retries=1, session=session
        )

    message = str(exc_info.value)
    assert "no match data or bracket metadata" in message
    assert "HTML response is not JSON" in message


def test_scrape_tournament_rejects_split_payload_without_full_rosters(
    monkeypatch,
):
    tournament_id = 4212
    layout = {
        "tournament": {
            "ctx": {
                "id": tournament_id,
                "name": "Craft Testers",
                "teams": [{"id": 1, "memberUserIds": [10]}],
            },
            "data": {"match": [{"id": 1}]},
        }
    }
    results_url = scraping_api.build_tournament_url(tournament_id)
    legacy_url = scraping_api.build_tournament_url(tournament_id, legacy=True)
    session = _Session(
        {
            results_url: layout,
            legacy_url: ValueError("HTML response is not JSON"),
        }
    )
    monkeypatch.setattr(
        scraping_api, "extract_tournament_data", lambda payload: payload
    )

    def raise_missing_key(_tournament_id):
        raise RuntimeError("SENDOU_KEY missing")

    monkeypatch.setattr(
        scraping_api,
        "fetch_tournament_teams",
        raise_missing_key,
    )

    with pytest.raises(RuntimeError) as exc_info:
        scraping_api.scrape_tournament(
            tournament_id, max_retries=1, session=session
        )

    assert "failed to enrich team rosters" in str(exc_info.value)


def test_rankings_update_rejects_partial_scrape_results():
    with pytest.raises(RuntimeError, match=r"4212.*4235"):
        update_cli._require_complete_scrape(
            {"scraped": 1, "failed": 2, "failed_ids": [4212, 4235]}
        )


def test_rankings_update_accepts_complete_scrape_results():
    update_cli._require_complete_scrape(
        {"scraped": 3, "failed": 0, "failed_ids": []}
    )
