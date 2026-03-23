"""Reusable utilities for LOOPR Plus-server drift diagnostics."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score

DEFAULT_DISPLAY_SCORE_MULTIPLIER = 25.0
DEFAULT_DISPLAY_SCORE_OFFSET = 6.0


def parse_cutoff_ts(cutoff_date: str) -> int:
    dt = datetime.strptime(cutoff_date, "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    )
    return int(dt.timestamp())


def ts_expr_to_seconds(expr: pl.Expr) -> pl.Expr:
    return (
        pl.when(expr.cast(pl.Float64, strict=False) > 1e11)
        .then(expr.cast(pl.Float64, strict=False) / 1000.0)
        .otherwise(expr.cast(pl.Float64, strict=False))
    )


def normalize_rankings_schema(
    rankings: pl.DataFrame,
    *,
    display_score_multiplier: float = DEFAULT_DISPLAY_SCORE_MULTIPLIER,
    display_score_offset: float = DEFAULT_DISPLAY_SCORE_OFFSET,
) -> pl.DataFrame:
    normalized = rankings
    if "player_id" not in normalized.columns and "id" in normalized.columns:
        normalized = normalized.rename({"id": "player_id"})
    if (
        "display_score" not in normalized.columns
        and "score" in normalized.columns
    ):
        normalized = normalized.with_columns(
            (
                pl.col("score") * display_score_multiplier
                + display_score_offset
            ).alias("display_score")
        )
    required = {"player_id", "display_score"}
    missing = required - set(normalized.columns)
    if missing:
        raise ValueError(
            "Rankings missing required columns: "
            + ", ".join(sorted(missing))
        )
    return (
        normalized.select(["player_id", "display_score"])
        .drop_nulls(["player_id", "display_score"])
        .group_by("player_id")
        .agg(pl.col("display_score").max().alias("display_score"))
    )


def normalize_suggested(votes: pl.DataFrame) -> pl.DataFrame:
    if "suggested" not in votes.columns:
        return votes.with_columns(pl.lit(False).alias("suggested"))
    if votes.schema["suggested"] == pl.Boolean:
        return votes.with_columns(pl.col("suggested").fill_null(False))
    return votes.with_columns(
        pl.col("suggested")
        .cast(pl.Utf8, strict=False)
        .fill_null("0")
        .str.to_lowercase()
        .str.strip_chars()
        .is_in(["1", "true", "t", "yes"])
        .alias("suggested")
    )


def derive_last_active_from_snapshot(
    *,
    repo_root: Path,
    rankings_parquet: Path,
    cutoff_date: str,
    tournaments_ref: Path | None = None,
) -> pl.DataFrame:
    cutoff_ts = parse_cutoff_ts(cutoff_date)
    base_dir = rankings_parquet.parent
    matches_path = base_dir / "matches.parquet"
    players_path = base_dir / "players.parquet"
    if not players_path.exists():
        return pl.DataFrame(
            schema={"player_id": pl.Int64, "last_active": pl.Int64}
        )

    players = (
        pl.read_parquet(players_path)
        .select(
            [
                "tournament_id",
                pl.col("team_id").cast(pl.Int64).alias("team_id"),
                pl.col("user_id").cast(pl.Int64).alias("player_id"),
            ]
        )
        .drop_nulls(["player_id"])
        .unique(subset=["tournament_id", "team_id", "player_id"])
    )

    lookups: list[pl.DataFrame] = []

    if matches_path.exists():
        matches = pl.read_parquet(matches_path).select(
            [
                "tournament_id",
                "team1_id",
                "team2_id",
                "last_game_finished_at",
                "match_created_at",
            ]
        )
        if not matches.is_empty():
            matches = matches.with_columns(
                ts_expr_to_seconds(
                    pl.coalesce(
                        [
                            pl.col("last_game_finished_at"),
                            pl.col("match_created_at"),
                        ]
                    )
                ).alias("last_active")
            ).drop_nulls(["last_active"])
            matches = matches.filter(pl.col("last_active") <= cutoff_ts)
            team1 = matches.select(
                [
                    "tournament_id",
                    pl.col("team1_id").cast(pl.Int64).alias("team_id"),
                    pl.col("last_active"),
                ]
            )
            team2 = matches.select(
                [
                    "tournament_id",
                    pl.col("team2_id").cast(pl.Int64).alias("team_id"),
                    pl.col("last_active"),
                ]
            )
            teams_long = pl.concat([team1, team2], how="vertical_relaxed").drop_nulls(
                ["team_id", "last_active"]
            )
            match_lookup = (
                teams_long.join(
                    players.drop_nulls(["team_id"]),
                    on=["tournament_id", "team_id"],
                    how="inner",
                )
                .group_by("player_id")
                .agg(pl.col("last_active").max().alias("last_active"))
            )
            lookups.append(match_lookup)

    tournaments_ref = (
        tournaments_ref
        if tournaments_ref is not None
        else (
            repo_root
            / "data"
            / "embeddings_window_540d_all"
            / "tournaments.parquet"
        )
    )
    if tournaments_ref.exists():
        tdf = pl.read_parquet(tournaments_ref)
        ts_col = (
            "start_time_ms" if "start_time_ms" in tdf.columns else "start_time"
        )
        if ts_col in tdf.columns:
            tlookup = tdf.select(
                [
                    pl.col("tournament_id").cast(pl.Int64),
                    ts_expr_to_seconds(pl.col(ts_col))
                    .floor()
                    .cast(pl.Int64, strict=False)
                    .alias("t_last_active"),
                ]
            ).drop_nulls(["t_last_active"])
            tlookup = tlookup.filter(pl.col("t_last_active") <= cutoff_ts)
            roster_lookup = (
                players.join(tlookup, on="tournament_id", how="inner")
                .group_by("player_id")
                .agg(pl.col("t_last_active").max().alias("last_active"))
            )
            lookups.append(roster_lookup)

    if not lookups:
        return pl.DataFrame(
            schema={"player_id": pl.Int64, "last_active": pl.Int64}
        )

    return (
        pl.concat(lookups, how="vertical_relaxed")
        .group_by("player_id")
        .agg(pl.col("last_active").max().alias("last_active"))
        .with_columns(pl.col("last_active").cast(pl.Int64, strict=False))
    )


def load_last_active_lookup(path: Path | None) -> pl.DataFrame:
    if path is None:
        return pl.DataFrame(
            schema={"player_id": pl.Int64, "last_active": pl.Int64}
        )
    if not path.exists():
        raise FileNotFoundError(f"user-meta path not found: {path}")

    if path.suffix.lower() == ".parquet":
        df = pl.read_parquet(path)
    else:
        df = pl.read_csv(path)

    id_col = None
    for candidate in ["player_id", "user_id", "id"]:
        if candidate in df.columns:
            id_col = candidate
            break
    if id_col is None:
        raise ValueError(f"{path} needs one of: player_id/user_id/id")

    ts_col = None
    for candidate in ["last_active", "last_active_ms", "last_active_sec"]:
        if candidate in df.columns:
            ts_col = candidate
            break
    if ts_col is None:
        raise ValueError(
            f"{path} needs one of: last_active/last_active_ms/last_active_sec"
        )

    sec_expr = pl.col(ts_col).cast(pl.Float64, strict=False)
    if ts_col.endswith("_ms"):
        sec_expr = sec_expr / 1000.0
    else:
        sec_expr = ts_expr_to_seconds(pl.col(ts_col))

    return (
        df.select(
            [
                pl.col(id_col).cast(pl.Int64).alias("player_id"),
                sec_expr.floor().cast(pl.Int64, strict=False).alias(
                    "last_active"
                ),
            ]
        )
        .drop_nulls(["player_id", "last_active"])
        .group_by("player_id")
        .agg(pl.col("last_active").max().alias("last_active"))
    )


def build_last_active_lookup(
    *,
    repo_root: Path,
    rankings_parquet: Path,
    cutoff_date: str,
    user_meta_path: Path | None = None,
    tournaments_ref: Path | None = None,
) -> pl.DataFrame:
    derived = derive_last_active_from_snapshot(
        repo_root=repo_root,
        rankings_parquet=rankings_parquet,
        cutoff_date=cutoff_date,
        tournaments_ref=tournaments_ref,
    )
    external = load_last_active_lookup(user_meta_path)
    if derived.is_empty() and external.is_empty():
        return pl.DataFrame(
            schema={"player_id": pl.Int64, "last_active": pl.Int64}
        )
    if derived.is_empty():
        return external
    if external.is_empty():
        return derived
    return (
        pl.concat([derived, external], how="vertical_relaxed")
        .group_by("player_id")
        .agg(pl.col("last_active").max().alias("last_active"))
    )


def split_map(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    has_true = df.filter(pl.col("suggested") == True).height > 0
    has_false = df.filter(pl.col("suggested") == False).height > 0
    splits = {"all": df}
    if has_true and has_false:
        splits["incumbents"] = df.filter(pl.col("suggested") == False)
        splits["suggests"] = df.filter(pl.col("suggested") == True)
    return splits


def safe_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    if len(y) < 2 or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def auc_bootstrap_ci(
    y: np.ndarray,
    s: np.ndarray,
    *,
    n_boot: int = 500,
    seed: int = 42,
) -> tuple[float | None, float | None]:
    if len(y) < 2 or len(np.unique(y)) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        sb = s[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(float(roc_auc_score(yb, sb)))
    if not aucs:
        return None, None
    return (
        float(np.quantile(aucs, 0.025)),
        float(np.quantile(aucs, 0.975)),
    )


def kde_mode_and_peaks(values: np.ndarray) -> tuple[float | None, int]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return None, 0
    if np.ptp(x) == 0:
        return float(x[0]), 1

    try:
        kde = gaussian_kde(x)
        spread = max(0.5, float(np.std(x, ddof=0)))
        grid = np.linspace(x.min() - 2 * spread, x.max() + 2 * spread, 1600)
        dens = kde.evaluate(grid)
        area = np.trapezoid(dens, grid)
        if area > 0:
            dens = dens / area
        mode = float(grid[int(np.argmax(dens))])
        peaks, _ = find_peaks(dens, prominence=np.max(dens) * 0.05)
        return mode, int(len(peaks))
    except Exception:
        hist, edges = np.histogram(x, bins="auto")
        if hist.sum() == 0:
            return None, 0
        idx = int(np.argmax(hist))
        center = float((edges[idx] + edges[idx + 1]) / 2.0)
        return center, 1


def distribution_overlap(
    pass_scores: np.ndarray, fail_scores: np.ndarray
) -> float | None:
    p = np.asarray(pass_scores, dtype=float)
    f = np.asarray(fail_scores, dtype=float)
    p = p[np.isfinite(p)]
    f = f[np.isfinite(f)]
    if p.size < 2 or f.size < 2:
        return None
    if np.ptp(p) == 0 and np.ptp(f) == 0:
        return 1.0 if float(p[0]) == float(f[0]) else 0.0

    lo = float(min(np.min(p), np.min(f)))
    hi = float(max(np.max(p), np.max(f)))
    span = max(1.0, hi - lo)
    grid = np.linspace(lo - 0.2 * span, hi + 0.2 * span, 1800)
    try:
        kp = gaussian_kde(p)
        kf = gaussian_kde(f)
        dp = kp.evaluate(grid)
        df = kf.evaluate(grid)
        ap = np.trapezoid(dp, grid)
        af = np.trapezoid(df, grid)
        if ap <= 0 or af <= 0:
            return None
        dp = dp / ap
        df = df / af
        overlap = np.trapezoid(np.minimum(dp, df), grid)
        return float(overlap)
    except Exception:
        return None


def cohen_d(pass_scores: np.ndarray, fail_scores: np.ndarray) -> float | None:
    p = np.asarray(pass_scores, dtype=float)
    f = np.asarray(fail_scores, dtype=float)
    p = p[np.isfinite(p)]
    f = f[np.isfinite(f)]
    if p.size < 2 or f.size < 2:
        return None
    sd_p = float(np.std(p, ddof=1))
    sd_f = float(np.std(f, ddof=1))
    pooled = math.sqrt(
        ((p.size - 1) * sd_p**2 + (f.size - 1) * sd_f**2)
        / max(1, p.size + f.size - 2)
    )
    if pooled == 0:
        return None
    return float((np.mean(p) - np.mean(f)) / pooled)


def top_decile_pass_rate(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[int | None, float | None]:
    if len(scores) < 10:
        return None, None
    q90 = float(np.quantile(scores, 0.9))
    mask = scores >= q90
    top_n = int(np.sum(mask))
    if top_n == 0:
        return 0, None
    return top_n, float(np.mean(labels[mask]))


__all__ = [
    "DEFAULT_DISPLAY_SCORE_MULTIPLIER",
    "DEFAULT_DISPLAY_SCORE_OFFSET",
    "parse_cutoff_ts",
    "ts_expr_to_seconds",
    "normalize_rankings_schema",
    "normalize_suggested",
    "derive_last_active_from_snapshot",
    "load_last_active_lookup",
    "build_last_active_lookup",
    "split_map",
    "safe_auc",
    "auc_bootstrap_ci",
    "kde_mode_and_peaks",
    "distribution_overlap",
    "cohen_d",
    "top_decile_pass_rate",
]
