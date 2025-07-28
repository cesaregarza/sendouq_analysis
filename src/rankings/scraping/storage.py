"""
Data storage and management functions for tournament data.

This module handles loading, saving, and managing scraped tournament data
across different storage formats and locations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import polars as pl
from tqdm import tqdm


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

    print(f"Saved {len(batch_data)} tournaments to {batch_filename}")


def load_scraped_tournaments(data_dir: str = "data/tournaments") -> List[Dict]:
    """
    Load all scraped tournament data from JSON files.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing tournament JSON files

    Returns
    -------
    list of dict
        Combined tournament data from all files
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Data directory {data_dir} does not exist")
        return []

    all_tournaments = []
    json_files = list(data_path.glob("tournament_*.json"))

    if not json_files:
        print(f"No tournament JSON files found in {data_dir}")
        return []

    print(f"Loading {len(json_files)} tournament files...")

    for json_file in tqdm(sorted(json_files), desc="Loading files"):
        try:
            with open(json_file, "r") as f:
                tournaments = json.load(f)
                if isinstance(tournaments, list):
                    all_tournaments.extend(tournaments)
                else:
                    all_tournaments.append(tournaments)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Failed to load {json_file}: {e}")

    print(f"Loaded {len(all_tournaments)} tournaments total")
    return all_tournaments


def get_tournament_summary(data_dir: str = "data/tournaments") -> pl.DataFrame:
    """
    Get a summary of scraped tournaments.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing tournament JSON files

    Returns
    -------
    pl.DataFrame
        Summary with tournament IDs, names, and metadata
    """
    tournaments = load_scraped_tournaments(data_dir)

    if not tournaments:
        return pl.DataFrame([])

    summary_data = []
    for entry in tournaments:
        tournament = entry.get("tournament", {})
        ctx = tournament.get("ctx", {})

        summary_data.append(
            {
                "tournament_id": ctx.get("id"),
                "name": ctx.get("name", "Unknown"),
                "start_time": ctx.get("startTime"),
                "is_finalized": ctx.get("isFinalized", False),
                "team_count": len(ctx.get("teams", [])),
                "has_matches": bool(
                    tournament.get("data", {}).get("match", [])
                ),
            }
        )

    return pl.DataFrame(summary_data)
