from __future__ import annotations

import json
import logging
import os
import time
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from typing import Literal

base_url = "https://sendou.ink/q/match/"
url_suffix = r"?_data=features%2Fsendouq%2Froutes%2Fq.match.%24id"

logger = logging.getLogger(__name__)


def scrape_match(match_id: int) -> dict:
    """Scrapes a single match from sendou.ink

    Args:
        match_id (int): The id of the match to scrape

    Returns:
        dict: The match data as a dictionary
    """
    logger.info("Scraping match id: %s", match_id)
    url = base_url + str(match_id) + url_suffix
    response = requests.get(url)
    return response.json()


def scrape_matches(
    start_id: int,
    end_id: int | Literal[False] = False,
    debounce_amt: float = 0.0,
) -> list[dict]:
    """Scrapes a range of matches from sendou.ink

    Args:
        start_id (int): The id of the first match to scrape
        end_id (int | Literal[False]): The id of the last match to scrape. If
            False, scrape all matches from start_id until it errors out.
        debounce_amt (float): The amount of time to wait between requests

    Returns:
        list[dict]: A list of match data dictionaries
    """
    matches = []
    if not end_id or end_id < start_id:
        logger.info("Scraping all matches from id: %s", start_id)
        end_id = int(1e12)  # 1 trillion, should hold if IDs remain sequential
    else:
        logger.info("Scraping matches from id: %s to %s", start_id, end_id)

    try:
        for match_id in range(start_id, end_id + 1):
            match_json = scrape_match(match_id)
            if match_json["match"]["reportedAt"] is None:
                logger.info(
                    "Match in progress, ending scrape. Last match id: %s",
                    match_id - 1,
                )
                break
            matches.append(match_json)
            if debounce_amt > 0.0:
                time.sleep(debounce_amt)
    except json.JSONDecodeError:
        logger.warning("All matches scraped, last match id: %s", match_id - 1)
    return matches


def scrape_matches_to_files(
    start_id: int,
    save_path: str,
    end_id: int | Literal[False] = False,
    major_directory_count: int = 1_000,
    minor_directory_count: int = 100,
) -> None:
    """Scrapes a range of matches from sendou.ink and saves them to files

    Args:
        start_id (int): The id of the first match to scrape
        save_path (str): The path to save the matches to
        end_id (int | Literal[False]): The id of the last match to scrape. If
            False, scrape all matches from start_id until it errors out.
        major_directory_count (int): The number of matches to save in each
            major directory. Defaults to 1,000.
        minor_directory_count (int): The number of matches to save in each
            minor directory. Defaults to 100.
    """
    matches = scrape_matches(start_id, end_id)
    create_directory_structure(
        start_id,
        start_id + len(matches) - 1,
        save_path,
        major_directory_count,
        minor_directory_count,
    )
    for i, match in enumerate(matches):
        write_match_to_file(match, save_path, i)


def create_directory_structure(
    start_id: int,
    end_id: int,
    save_path: str,
    major_directory_count: int = 1_000,
    minor_directory_count: int = 100,
) -> None:
    """Creates the directory structure for saving matches

    Args:
        start_id (int): The id of the first match to scrape
        end_id (int): The id of the last match to scrape
        save_path (str): The path to save the matches to
        major_directory_count (int): The number of matches to save in each
            major directory. Defaults to 1,000.
        minor_directory_count (int): The number of matches to save in each
            minor directory. Defaults to 100.
    """
    num_matches = end_id - start_id + 1
    max_major_directory = num_matches // major_directory_count
    max_minor_directory = (
        num_matches % major_directory_count
    ) // minor_directory_count

    for major_directory in range(max_major_directory + 1):
        if major_directory < max_major_directory:
            minor_directory_end = major_directory_count // minor_directory_count
        else:
            minor_directory_end = max_minor_directory + 1

        for minor_directory in range(minor_directory_end):
            directory_path = os.path.join(
                save_path, str(major_directory), str(minor_directory)
            )
            os.makedirs(directory_path, exist_ok=True)


def write_match_to_file(match: dict, save_path: str, match_id: int) -> None:
    """Writes a match to a file

    Args:
        match (dict): The match data to write
        save_path (str): The path to save the match to
        match_id (int): The id of the match, used to determine the file path
    """
    major_directory = match_id // 1_000
    minor_directory = match_id // 100
    base_path = os.path.join(
        save_path, str(major_directory), str(minor_directory)
    )
    path = os.path.join(base_path, f"sendouq_{match_id}.json")
    with open(path, "w") as f:
        json.dump(match, f)
