from __future__ import annotations

import polars as pl


def expand_players_for_legacy_matches(
    matches: pl.DataFrame,
    players: pl.DataFrame,
) -> pl.DataFrame:
    """Backfill missing tournament/team roster rows using existing team rosters.

    Historical callers sometimes passed a single tournament's roster table while
    ranking many tournaments that reused the same team IDs. Older code paths were
    permissive about this. `loopr` is stricter, so we clone a team's first-seen
    roster into missing tournament/team pairs before delegating.
    """

    required_player_cols = {"tournament_id", "team_id", "user_id"}
    required_match_cols = {"tournament_id", "winner_team_id", "loser_team_id"}
    if not required_player_cols.issubset(players.columns):
        return players
    if not required_match_cols.issubset(matches.columns):
        return players

    tournament_teams = (
        pl.concat(
            [
                matches.select(
                    [
                        "tournament_id",
                        pl.col("winner_team_id").alias("team_id"),
                    ]
                ),
                matches.select(
                    [
                        "tournament_id",
                        pl.col("loser_team_id").alias("team_id"),
                    ]
                ),
            ]
        )
        .drop_nulls()
        .unique()
    )
    existing = players.select(["tournament_id", "team_id"]).unique()
    missing = tournament_teams.join(
        existing,
        on=["tournament_id", "team_id"],
        how="anti",
    )
    if missing.is_empty():
        return players

    first_seen = (
        players.group_by("team_id")
        .agg(pl.col("tournament_id").min().alias("template_tournament_id"))
    )
    template_users = (
        players.join(first_seen, on="team_id", how="inner")
        .filter(pl.col("tournament_id") == pl.col("template_tournament_id"))
        .select(["team_id", "user_id"])
        .unique()
    )
    cloned = (
        missing.join(template_users, on="team_id", how="inner")
        .select(["tournament_id", "team_id", "user_id"])
        .unique()
    )
    if cloned.is_empty():
        return players

    return pl.concat([players, cloned], how="vertical_relaxed").unique()


def filter_legacy_appearances(
    matches: pl.DataFrame,
    players: pl.DataFrame | None,
    appearances: pl.DataFrame | None,
) -> pl.DataFrame | None:
    """Drop appearance rows that fall outside the current ranking slice.

    Older sendouq_analysis callers commonly loaded a broad appearances table and
    then filtered matches/players separately. The previous local pipeline
    effectively ignored those orphaned appearance rows. `loopr` validates them
    strictly, so we scope appearances to the active match keys and participant
    set before delegating.
    """

    if appearances is None or appearances.is_empty():
        return appearances

    required_appearance_cols = {"tournament_id", "match_id", "user_id"}
    if not required_appearance_cols.issubset(appearances.columns):
        return appearances

    scoped = appearances

    if {"tournament_id", "match_id"}.issubset(matches.columns):
        scoped = scoped.join(
            matches.select(["tournament_id", "match_id"]).drop_nulls().unique(),
            on=["tournament_id", "match_id"],
            how="semi",
        )
        if scoped.is_empty():
            return scoped

    if players is not None and {"tournament_id", "user_id"}.issubset(
        players.columns
    ):
        player_keys = (
            players.select(["tournament_id", "user_id"]).drop_nulls().unique()
        )
        scoped = scoped.join(
            player_keys,
            on=["tournament_id", "user_id"],
            how="semi",
        )

    if players is not None and {
        "tournament_id",
        "user_id",
        "team_id",
    }.issubset(players.columns):
        needs_team_id = ("team_id" not in scoped.columns) or scoped.select(
            pl.col("team_id").is_null().any()
        ).item()
        if needs_team_id:
            roster_lookup = (
                players.select(["tournament_id", "user_id", "team_id"])
                .drop_nulls()
                .unique()
            )
            scoped = scoped.join(
                roster_lookup,
                on=["tournament_id", "user_id"],
                how="left",
            )
            if "team_id_right" in scoped.columns:
                scoped = scoped.with_columns(
                    pl.coalesce(
                        [pl.col("team_id"), pl.col("team_id_right")]
                    ).alias("team_id")
                ).drop("team_id_right")

        if "team_id" in scoped.columns:
            scoped = scoped.drop_nulls(["team_id"])

    return scoped
