"""
Data storage and management functions for tournament data.

This module handles loading, saving, and managing scraped tournament data
across different storage formats and locations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import polars as pl
from tqdm import tqdm

logger = logging.getLogger(__name__)


def save_tournament_batch(
    batch_data: List[Dict], batch_idx: int, output_dir: str
) -> None:
    """
    Save a batch of tournament data to a JSON file.

    Parameters
    ----------
    batch_data : list of dict
        Tournament data to save
    batch_idx : int
        Batch index for filename
    output_dir : str
        Output directory path
    """
    if not batch_data:
        return

    batch_filename = f"tournament_{batch_idx}.json"
    batch_path = Path(output_dir) / batch_filename

    with open(batch_path, "w") as f:
        json.dump(batch_data, f, indent=2)

    logger.info(f"Saved {len(batch_data)} tournaments to {batch_filename}")


def load_scraped_tournaments(data_dir: str = "data/tournaments") -> List[Dict]:
    """
    Load all scraped tournament data from JSON files, recursively.

    - Recurses into subdirectories of `data_dir`.
    - Includes per-ID files (e.g., `tournament_2090.json`).
    - Includes batch files (e.g., `tournament_0.json`, lists of tournaments).
    - Includes snapshot files (e.g., `tournaments_*.json` and
      `tournaments_continuous_*.json`).

    Parameters
    ----------
    data_dir : str, optional
        Directory containing tournament JSON files (root; searched recursively)

    Returns
    -------
    list of dict
        Combined tournament data from all files
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning(f"Data directory {data_dir} does not exist")
        return []

    all_tournaments: List[Dict] = []

    # Recursively gather files by pattern
    per_id_files = list(data_path.rglob("tournament_*.json"))
    snapshot_files = list(data_path.rglob("tournaments_*.json"))
    continuous_files = list(data_path.rglob("tournaments_continuous_*.json"))

    # Avoid double counting continuous files in snapshot_files if patterns overlap
    snapshot_only = [
        p
        for p in snapshot_files
        if p.name.startswith("tournaments_")
        and not p.name.startswith("tournaments_continuous_")
    ]

    json_files = sorted(set(per_id_files + snapshot_only + continuous_files))

    if not json_files:
        logger.warning(
            f"No tournament JSON files found under {data_dir} (recursive)"
        )
        return []

    logger.info(
        "Loading %d JSON files recursively (%d per-id/batch, %d snapshots, %d continuous)"
        % (
            len(json_files),
            len(per_id_files),
            len(snapshot_only),
            len(continuous_files),
        )
    )

    for json_file in tqdm(json_files, desc="Loading files"):
        try:
            with open(json_file, "r") as f:
                payload = json.load(f)
            if isinstance(payload, list):
                all_tournaments.extend(payload)
            else:
                all_tournaments.append(payload)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load %s: %s", json_file, e)

    logger.info(f"Loaded {len(all_tournaments)} tournaments total")
    return all_tournaments


def get_tournament_summary(data_dir: str = "data/tournaments") -> pl.DataFrame:
    """
    Get a summary of scraped tournaments using the enhanced parser.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing tournament JSON files

    Returns
    -------
    pl.DataFrame
        Summary with comprehensive tournament metadata including organization,
        settings, staff, and match/team counts
    """
    from rankings.core.parser import parse_tournaments_data

    tournaments = load_scraped_tournaments(data_dir)

    if not tournaments:
        return pl.DataFrame([])

    # Use the enhanced parser to get full tournament metadata
    tables = parse_tournaments_data(tournaments)
    tournament_df = tables.get("tournaments")

    if tournament_df is None:
        return pl.DataFrame([])

    # Return the comprehensive tournament metadata
    return tournament_df
