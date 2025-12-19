from __future__ import annotations

"""Backfill daily rankings_update runs.

This command replays `rankings_update` over a range of UTC dates, producing one
rankings snapshot per day (persisted to DB, and optionally uploaded to S3 via
the existing `rankings_update` upload behavior).

Typical usage (production container):
  poetry run rankings_backfill_daily --start-date 2025-11-09 --end-date 2025-12-18
"""

import argparse
import logging
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import text

from rankings.cli import update as update_cli
from rankings.sql import create_engine
from rankings.sql.constants import SCHEMA


def _parse_date(value: str) -> date:
    return date.fromisoformat(value.strip())


def _iter_dates(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("end-date must be >= start-date")
    out: list[date] = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur = cur + timedelta(days=1)
    return out


def _day_anchor_ms(d: date, *, cutoff: str) -> int:
    if cutoff == "start":
        dt = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc)
    else:
        dt = datetime(
            d.year, d.month, d.day, 23, 59, 59, 999000, tzinfo=timezone.utc
        )
    return int(dt.timestamp() * 1000)


def _format_build_version(template: str, d: date) -> str:
    return template.format(date=d.isoformat(), date_compact=d.strftime("%Y%m%d"))


def _run_exists(engine, calculated_at_ms: int, build_version: str) -> bool:  # noqa: ANN001
    q = text(
        f"""
        SELECT 1
        FROM {SCHEMA}.player_rankings
        WHERE calculated_at_ms = :ts
          AND build_version = :bv
        LIMIT 1
        """
    )
    with engine.connect() as conn:
        row = conn.execute(
            q, {"ts": int(calculated_at_ms), "bv": str(build_version)}
        ).fetchone()
    return row is not None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill one rankings_update run per day"
    )
    parser.add_argument(
        "--start-date",
        required=True,
        type=str,
        help="Start date in UTC (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        type=str,
        help="End date in UTC (YYYY-MM-DD). Defaults to today (UTC).",
    )
    parser.add_argument(
        "--cutoff",
        choices=["end", "start"],
        default="end",
        help="Anchor within each day used for --as-of and calculated_at (UTC).",
    )
    parser.add_argument(
        "--since-days",
        type=int,
        default=540,
        help="Retrieval window in days (passed to rankings_update --since-days).",
    )
    parser.add_argument(
        "--build-version-template",
        type=str,
        default="daily-{date}",
        help=(
            "Template for build_version; supports {date}=YYYY-MM-DD and "
            "{date_compact}=YYYYMMDD."
        ),
    )
    parser.add_argument(
        "--compiled-out",
        type=str,
        default=None,
        help="Override rankings_update --compiled-out for outputs (optional).",
    )
    parser.add_argument(
        "--include-unranked",
        action="store_true",
        help="Include unranked tournaments (passed to rankings_update --include-unranked).",
    )
    parser.add_argument(
        "--no-upload-s3",
        action="store_true",
        help="Disable S3 upload (passed to rankings_update --no-upload-s3).",
    )
    parser.add_argument(
        "--persist-appearance-teams",
        action="store_true",
        help=(
            "Persist appearance team assignments to DB on each day "
            "(default: skip DB upserts to reduce backfill load)."
        ),
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not skip days already present for the chosen build_version.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to the next day if a run fails (default: stop).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level for stdout output",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("rankings.cli.backfill_daily")

    start = _parse_date(args.start_date)
    end = _parse_date(args.end_date) if args.end_date else datetime.now(
        timezone.utc
    ).date()

    days = _iter_dates(start, end)
    log.info("Backfill days: %s -> %s (%d days)", start, end, len(days))

    # Shared engine for existence checks (update_cli uses its own engine per run)
    engine = create_engine()
    try:

        for idx, d in enumerate(days, 1):
            anchor_ms = _day_anchor_ms(d, cutoff=args.cutoff)
            build_version = _format_build_version(
                args.build_version_template, d
            )
            run_ts = datetime.fromtimestamp(
                anchor_ms / 1000, tz=timezone.utc
            ).strftime("%Y%m%d_%H%M%S")

            if not args.no_skip_existing:
                try:
                    if _run_exists(engine, anchor_ms, build_version):
                        log.info(
                            "[%d/%d] %s already present (build=%s); skipping",
                            idx,
                            len(days),
                            d.isoformat(),
                            build_version,
                        )
                        continue
                except Exception as e:
                    log.warning(
                        "[%d/%d] %s run existence check failed; proceeding: %s",
                        idx,
                        len(days),
                        d.isoformat(),
                        e,
                    )

            cmd = [
                "--skip-discovery",
                "--save-to-db",
                "--since-days",
                str(int(args.since_days)),
                "--as-of",
                str(anchor_ms),
                "--calculated-at",
                str(anchor_ms),
                "--run-ts",
                run_ts,
                "--build-version",
                build_version,
            ]
            if args.include_unranked:
                cmd.append("--include-unranked")
            else:
                cmd.append("--only-ranked")
            if args.no_upload_s3:
                cmd.append("--no-upload-s3")
            if not args.persist_appearance_teams:
                cmd.append("--no-persist-appearance-teams")
            if args.compiled_out:
                cmd.extend(["--compiled-out", str(args.compiled_out)])

            log.info(
                "[%d/%d] Running %s build=%s anchor=%s",
                idx,
                len(days),
                d.isoformat(),
                build_version,
                anchor_ms,
            )
            rc = update_cli.main(cmd)
            if rc != 0:
                msg = f"rankings_update failed for {d.isoformat()} (rc={rc})"
                if args.continue_on_error:
                    log.error("%s; continuing", msg)
                    continue
                log.error("%s; stopping", msg)
                return rc

        log.info("Backfill complete.")
        return 0
    finally:
        try:
            engine.dispose()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
