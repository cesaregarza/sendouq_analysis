from __future__ import annotations

"""
Pull player rankings from the database using the notebook_db helpers.

This CLI loads the `player_rankings` table (optionally the latest run only),
optionally joins player display names, and writes to CSV/NDJSON/Parquet or
prints a preview to stdout.
"""

import argparse
import os
from typing import Optional

import polars as pl


def _import_nb():
    try:
        from codex_scripts import notebook_db as nb  # type: ignore

        return nb
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "codex_scripts.notebook_db is required for this command."
        ) from e


def _load_env(dotenv: Optional[str]) -> None:
    if dotenv and os.path.exists(dotenv):
        try:
            with open(dotenv, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    os.environ.setdefault(k, v)
        except Exception:
            pass


def _latest_run(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty() or "calculated_at_ms" not in df.columns:
        return df
    max_ts = df.select(pl.max("calculated_at_ms")).item()
    return df.filter(pl.col("calculated_at_ms") == max_ts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pull player rankings from DB")
    parser.add_argument(
        "--dotenv",
        type=str,
        default=".env",
        help="Path to .env with DB creds (default: .env)",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Restrict to the latest rankings run (by calculated_at_ms)",
    )
    parser.add_argument(
        "--build-version",
        type=str,
        default=None,
        help="Filter to a specific build_version",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of rows"
    )
    parser.add_argument(
        "--join-players",
        action="store_true",
        help="Join player display names from players table",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filepath (infers format by extension: .csv, .json, .ndjson, .parquet)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "json", "ndjson", "parquet"],
        default=None,
        help="Explicit output format (overrides extension inference)",
    )

    args = parser.parse_args(argv)

    _load_env(args.dotenv)
    nb = _import_nb()

    pr = nb.load_player_rankings()
    if args.build_version:
        pr = pr.filter(pl.col("build_version") == args.build_version)
    if args.latest and not pr.is_empty():
        pr = _latest_run(pr)

    if args.join_players and not pr.is_empty():
        players = nb.load_players(columns=["player_id", "display_name"])  # type: ignore[arg-type]
        if not players.is_empty():
            pr = pr.join(
                players, left_on="player_id", right_on="player_id", how="left"
            )

    if args.limit is not None and args.limit > 0:
        pr = pr.head(args.limit)

    # Output
    if args.output:
        out = args.output
        fmt = args.format
        if fmt is None:
            if out.endswith(".csv"):
                fmt = "csv"
            elif out.endswith(".ndjson"):
                fmt = "ndjson"
            elif out.endswith(".json"):
                fmt = "json"
            elif out.endswith(".parquet"):
                fmt = "parquet"
            else:
                fmt = "csv"
        if fmt == "csv":
            pr.write_csv(out)
        elif fmt == "ndjson":
            pr.write_ndjson(out)  # type: ignore[attr-defined]
        elif fmt == "json":
            try:
                pr.write_json(out)  # polars >=0.20
            except Exception:
                pr.write_ndjson(out)
        elif fmt == "parquet":
            pr.write_parquet(out)
        print(f"Wrote {pr.height} rows to {out}")
    else:
        # Print a small preview to stdout
        print(pr.head(20))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
