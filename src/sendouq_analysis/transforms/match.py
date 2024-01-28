import logging

import numpy as np
import pandas as pd

from sendouq_analysis.constants import COLUMNS, SQ_DATA

MATCHCOLS = COLUMNS.MATCHES


def build_match_df(
    matches_df: pd.DataFrame,
) -> pd.DataFrame:
    """Transforms the input DataFrame of matches by performing several
    preprocessing steps.

    The function performs the following operations on the matches DataFrame:
    - Converts the 'created_at' and 'reported_at' columns to datetime objects.
    - Calculates the season for each match based on the time difference between
      matches.
    - Assigns a time slot for each match for use in a Gantt chart visualization.

    Args:
        matches_df (pd.DataFrame): A DataFrame containing match data.

    Returns:
        pd.DataFrame: The transformed DataFrame with additional columns for
        season and time slot.
    """
    logging.info("Building match df")
    matches_df = matches_df.copy()
    time_cols = [MATCHCOLS.CREATED_AT, MATCHCOLS.REPORTED_AT]
    for col in time_cols:
        matches_df[col] = pd.to_datetime(matches_df[col], unit="s")

    matches_df = calculate_seasons(matches_df)
    matches_df[MATCHCOLS.TIME_SLOT] = calculate_gantt_row(matches_df)
    return matches_df


def calculate_seasons(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the season for each match.

    Args:
        matches_df (pd.DataFrame): A DataFrame of matches

    Returns:
        pd.DataFrame: A DataFrame of matches with a season column
    """
    logging.info("Calculating seasons")
    created = matches_df[MATCHCOLS.CREATED_AT]
    previous_created = (
        created.shift(1).rename("previous_created").fillna(created.iloc[0])
    )
    time_between_matches = created.sub(previous_created).rename(
        "time_between_matches"
    )
    big_gap = time_between_matches > SQ_DATA.SEASON_BREAK_THRESHOLD
    matches_df[MATCHCOLS.SEASON] = big_gap.cumsum()
    return matches_df


def calculate_gantt_row(matches_df: pd.DataFrame) -> pd.Series:
    """Calculates the row index for a Gantt chart.

    Gantt charts are a way of visualizing the duration of a series of events.
    Since they usually have each row correspond to a unique event, we instead
    want to have each row correspond to the next available time slot. The way
    this works is that for each match, we find the first available time slot
    after the match's start time. If there is no available time slot, we create
    a new one.

    Args:
        matches_df (pd.DataFrame): A DataFrame of matches

    Returns:
        pd.Series: A Series of row indices for a Gantt chart
    """
    matches_df = matches_df.copy()
    matches_df = matches_df.sort_values(MATCHCOLS.CREATED_AT)

    def check_time_slot(row: pd.Series, gantt_rows: list[pd.Series]):
        for i, gantt_row in enumerate(gantt_rows):
            if row[MATCHCOLS.CREATED_AT] > gantt_row[MATCHCOLS.REPORTED_AT]:
                gantt_rows[i] = row
                return i
        gantt_rows.append(row)
        return len(gantt_rows) - 1

    gantt_rows = []
    time_slot_series = pd.Series(np.nan, index=matches_df.index)
    for i, row in matches_df.iterrows():
        time_slot_series.loc[i] = check_time_slot(row, gantt_rows)
    return time_slot_series