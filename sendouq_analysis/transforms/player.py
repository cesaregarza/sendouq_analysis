import logging

import numpy as np
import pandas as pd
from scipy.stats import kstest, lognorm

from sendouq_analysis.constants import (
    COLUMNS,
    ROLLING_MATCH_WINDOWS,
    ROLLING_TIME_WINDOWS,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def build_player_df(
    matches_df: pd.DataFrame,
    user_memento_df: pd.DataFrame,
    groups_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct a DataFrame containing player data by merging match, user memento,
    and group data.

    Parameters
    ----------
    matches_df : pd.DataFrame
        DataFrame containing match data.
    user_memento_df : pd.DataFrame
        DataFrame containing user memento data.
    groups_df : pd.DataFrame
        DataFrame containing group data.

    Returns
    -------
    pd.DataFrame
        The resulting DataFrame after merging and processing player data.
    """
    # Base merges
    logging.info("Building player_df")
    player_df = base_merges(
        matches_df=matches_df,
        user_memento_df=user_memento_df,
        groups_df=groups_df,
    )
    player_latest_df = generate_latest_df(player_df)
    shape, loc, scale = fit_lognorm(player_latest_df)
    player_df = calculate_logz_values(
        player_df=player_df,
        shape=shape,
        loc=loc,
        scale=scale,
    )
    player_df = calculate_teammate_enemy_values(player_df)
    player_df = calculate_rolling_data(player_df)
    return player_df


def build_team_enemy_xref(
    matches_df: pd.DataFrame,
) -> pd.Series:
    """
    Create a cross-reference Series mapping match and group IDs to enemy group
    IDs.

    Parameters
    ----------
    matches_df : pd.DataFrame
        DataFrame containing match data with team IDs.

    Returns
    -------
    pd.Series
        A Series indexed by match_id and group_id with enemy_group_id as values.
    """
    return pd.concat(
        [
            matches_df[
                [
                    COLUMNS.MATCHES.MATCH_ID,
                    COLUMNS.MATCHES.ALPHA_TEAM_ID,
                    COLUMNS.MATCHES.BRAVO_TEAM_ID,
                ]
            ].rename(
                columns={
                    COLUMNS.MATCHES.ALPHA_TEAM_ID: COLUMNS.GROUPS.GROUP_ID,
                    COLUMNS.MATCHES.BRAVO_TEAM_ID: COLUMNS.PLAYER.ENEMY_GROUP_ID,
                }
            ),
            matches_df[
                [
                    COLUMNS.MATCHES.MATCH_ID,
                    COLUMNS.MATCHES.BRAVO_TEAM_ID,
                    COLUMNS.MATCHES.ALPHA_TEAM_ID,
                ]
            ].rename(
                columns={
                    COLUMNS.MATCHES.BRAVO_TEAM_ID: COLUMNS.GROUPS.GROUP_ID,
                    COLUMNS.MATCHES.ALPHA_TEAM_ID: COLUMNS.PLAYER.ENEMY_GROUP_ID,
                }
            ),
        ],
        axis=0,
        ignore_index=True,
    ).set_index([COLUMNS.MATCHES.MATCH_ID, COLUMNS.GROUPS.GROUP_ID])[
        COLUMNS.PLAYER.ENEMY_GROUP_ID
    ]


def correct_sp(
    player_df: pd.DataFrame,
) -> pd.Series:
    """
    Correct the 'sp' values in the player DataFrame by considering the changes
    and differences over time.

    Parameters
    ----------
    player_df : pd.DataFrame
        DataFrame containing player data with 'sp' and 'sp_diff' columns.

    Returns
    -------
    pd.Series
        A Series with the corrected 'sp' values, indexed as 'after_sp'.
    """
    sorted_scores = player_df.copy().sort_values(
        [COLUMNS.PLAYER.USER_ID, COLUMNS.PLAYER.CREATED_AT_DT], ascending=True
    )
    sorted_scores["changed"] = (
        sorted_scores[COLUMNS.PLAYER.SP]
        != sorted_scores[COLUMNS.PLAYER.SP].shift()
    )
    sorted_scores.iloc[0, -1] = False
    sorted_scores["sp_id"] = sorted_scores.groupby(COLUMNS.PLAYER.USER_ID)[
        "changed"
    ].cumsum()
    sorted_scores[COLUMNS.PLAYER.SP_DIFF] = sorted_scores[
        COLUMNS.PLAYER.SP_DIFF
    ].fillna(0)
    return (
        sorted_scores[COLUMNS.PLAYER.SP]
        .add(
            sorted_scores.groupby([COLUMNS.PLAYER.USER_ID, "sp_id"])[
                COLUMNS.PLAYER.SP_DIFF
            ].cumsum()
        )
        .rename(COLUMNS.PLAYER.AFTER_SP)
        .sort_index()
    )


def base_merges(
    matches_df: pd.DataFrame,
    user_memento_df: pd.DataFrame,
    groups_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform base merges of match, user memento, and group data to form a
    preliminary player DataFrame.

    Parameters
    ----------
    matches_df : pd.DataFrame
        DataFrame containing match data.
    user_memento_df : pd.DataFrame
        DataFrame containing user memento data.
    groups_df : pd.DataFrame
        DataFrame containing group data.

    Returns
    -------
    pd.DataFrame
        The merged DataFrame with additional computed columns.
    """
    team_enemy_xref = build_team_enemy_xref(matches_df)

    player_df = (
        user_memento_df.copy()
        .merge(
            matches_df[
                [
                    COLUMNS.MATCHES.MATCH_ID,
                    COLUMNS.MATCHES.CREATED_AT,
                    COLUMNS.MATCHES.REPORTED_AT,
                    COLUMNS.MATCHES.WINNER,
                ]
            ],
            how="left",
            on=COLUMNS.MATCHES.MATCH_ID,
        )
        .merge(
            groups_df[
                [
                    COLUMNS.GROUPS.USER_ID,
                    COLUMNS.GROUPS.GROUP_ID,
                    COLUMNS.GROUPS.DISCORD_NAME,
                    COLUMNS.GROUPS.IN_GAME_NAME,
                    COLUMNS.GROUPS.TEAM,
                ]
            ],
            how="left",
            on=[COLUMNS.GROUPS.USER_ID, COLUMNS.GROUPS.GROUP_ID],
        )
        .merge(
            team_enemy_xref,
            how="left",
            on=[COLUMNS.MATCHES.MATCH_ID, COLUMNS.GROUPS.GROUP_ID],
        )
        .assign(
            is_winner=lambda df: df[COLUMNS.GROUPS.TEAM] == df[COLUMNS.MATCHES.WINNER],
        )
    )
    time_cols = [COLUMNS.MATCHES.CREATED_AT, COLUMNS.MATCHES.REPORTED_AT]
    for col in time_cols:
        player_df[f"{col}_dt"] = pd.to_datetime(player_df[col], unit="s")

    player_df[COLUMNS.PLAYER.AFTER_SP] = correct_sp(player_df)
    return player_df


def generate_latest_df(
    player_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate a DataFrame containing the latest player data, excluding cancelled
    matches.

    Parameters
    ----------
    player_df : pd.DataFrame
        DataFrame containing player data.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the latest player data for each user.
    """
    no_cancelled = player_df.copy().query("winner != 'cancelled'")

    max_reported_at = no_cancelled.groupby("user_id")[
        "reported_at_dt"
    ].transform("max")
    latest_mask = no_cancelled["reported_at_dt"] == max_reported_at
    return (
        no_cancelled[latest_mask]
        .copy()
        .query("(calculated == 1) | (matches_count == matches_count_needed)")
    )


def fit_lognorm(
    player_latest_df: pd.DataFrame,
) -> tuple[float, float, float]:
    """
    Fit a log-normal distribution to the 'after_sp' column of the player
    DataFrame and perform a KS test.

    Parameters
    ----------
    player_latest_df : pd.DataFrame
        DataFrame containing the latest player data.

    Returns
    -------
    tuple[float, float, float]
        The shape, location, and scale parameters of the fitted log-normal
        distribution.
    """
    # Lognorm fit
    shape, loc, scale = lognorm.fit(player_latest_df["after_sp"].dropna())
    logging.info(f"Lognorm fit: shape={shape}, loc={loc}, scale={scale}")
    ks_stat, p_value = kstest(
        player_latest_df["after_sp"].dropna(),
        "lognorm",
        args=(shape, loc, scale),
    )
    logging.info(f"KS test: stat={ks_stat}, p_value={p_value}")
    return shape, loc, scale


def calculate_logz_values(
    player_df: pd.DataFrame,
    shape: float,
    loc: float,
    scale: float,
) -> pd.DataFrame:
    """
    Calculate the log-z values for 'sp' and 'after_sp' columns in the player
    DataFrame.

    Parameters
    ----------
    player_df : pd.DataFrame
        DataFrame containing player data.
    shape : float
        The shape parameter of the log-normal distribution.
    loc : float
        The location parameter of the log-normal distribution.
    scale : float
        The scale parameter of the log-normal distribution.

    Returns
    -------
    pd.DataFrame
        The player DataFrame with added 'logz' columns for 'sp' and 'after_sp'.
    """
    for col in ["sp", "after_sp"]:
        player_df[f"{col}_logz"] = (
            player_df[col].pipe(np.log).sub(np.log(scale)).div(shape)
        )

    player_df["sp_diff_logz"] = player_df["after_sp_logz"].sub(
        player_df["sp_logz"]
    )
    return player_df


def calculate_teammate_enemy_values(
    player_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate teammate and enemy log-z values for each player in the DataFrame.

    Parameters
    ----------
    player_df : pd.DataFrame
        DataFrame containing player data with 'sp_logz' and 'sp_diff_logz'
        values.

    Returns
    -------
    pd.DataFrame
        The player DataFrame with added columns for teammate and enemy log-z
        values.
    """
    group_sp_logz = (
        player_df.groupby(["group_id", "match_id"])
        .agg(
            {
                "sp_logz": ["sum", "std"],
                "sp_diff_logz": ["sum", "std"],
            }
        )
        .set_axis(
            [
                "sp_logz_sum",
                "sp_logz_std",
                "sp_diff_logz_sum",
                "sp_diff_logz_std",
            ],
            axis=1,
        )
    )

    player_df = player_df.merge(
        group_sp_logz,
        how="left",
        on=["group_id", "match_id"],
    ).merge(
        group_sp_logz.rename(
            columns={
                "sp_logz_sum": "enemy_sp_logz_sum",
                "sp_logz_std": "enemy_sp_logz_std",
                "sp_diff_logz_sum": "enemy_sp_diff_logz_sum",
                "sp_diff_logz_std": "enemy_sp_diff_logz_std",
            }
        ),
        how="left",
        left_on=["enemy_group_id", "match_id"],
        right_on=["group_id", "match_id"],
    )
    teammate_sp_logz = player_df["sp_logz_sum"].sub(player_df["sp_logz"]).div(3)
    enemy_sp_logz = player_df["enemy_sp_logz_sum"].div(4)
    player_df["teammate_sp_logz_diff"] = player_df["sp_logz"].sub(
        teammate_sp_logz
    )
    player_df["enemy_sp_logz_diff"] = player_df["sp_logz"].sub(enemy_sp_logz)
    return player_df


def calculate_rolling_data(
    player_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate rolling statistics for player win rates over specified time
    windows.

    Parameters
    ----------
    player_df : pd.DataFrame
        DataFrame containing player data with 'is_winner' and 'created_at_dt'
        columns.

    Returns
    -------
    pd.DataFrame
        The player DataFrame with added rolling statistics for win rates.
    """
    for window in ROLLING_TIME_WINDOWS:
        rolling_data = (
            player_df.sort_values("created_at_dt")
            .groupby("user_id")
            .rolling(window, on="created_at_dt")["is_winner"]
            .agg(["count", "sum"])
            .set_axis([f"count_{window}", f"won_matches_{window}"], axis=1)
        )
        player_df = player_df.merge(
            rolling_data,
            how="left",
            on=["user_id", "created_at_dt"],
        )

    for window in ROLLING_MATCH_WINDOWS:
        for column in [
            "after_sp",
            "after_sp_logz",
            "teammate_sp_logz_diff",
            "enemy_sp_logz_diff",
        ]:
            rolling_data = (
                player_df.sort_values("created_at_dt")
                .groupby("user_id")
                .rolling(window, on="created_at_dt")[column]
                .std()
                .rename(f"{column}_std_{window}")
            )
            player_df = player_df.merge(
                rolling_data,
                how="left",
                on=["user_id", "created_at_dt"],
            )
        player_df[f"has_calculated_{window}"] = player_df[
            f"after_sp_std_{window}"
        ].notna()
    return player_df


def calculate_cumulative_data(
    player_df: pd.DataFrame,
) -> pd.DataFrame:
    cumulative_matches = (
        player_df.query("winner != 'cancelled'")
        .sort_values("created_at_dt")
        .groupby("user_id")
        .cumcount()
        .rename("cumulative_matches")
    )

    return player_df.merge(
        cumulative_matches, left_index=True, right_index=True, how="left"
    )
