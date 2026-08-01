#!/usr/bin/env python3
"""Run one half-life setting so the six predeclared settings can be parallelized."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .analyses import half_life_tradeoff
from .core import prepare_evaluation_data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--setting", required=True)
    args = parser.parse_args()
    setting = None if args.setting == "no_decay" else float(args.setting)
    output = args.output_dir.resolve()
    data = prepare_evaluation_data(args.source_root.resolve())
    base = pd.read_parquet(output / "tables" / "walk_forward_predictions.parquet")
    common = pd.read_parquet(output / "tables" / "plus_common_candidates.parquet")
    result = half_life_tradeoff(
        data,
        base,
        common,
        settings=[setting],
        checkpoint_path=output / "cache" / f"half_life_fullcal_{args.setting}.parquet",
    )
    result.to_csv(output / "tables" / f"half_life_{args.setting}.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
