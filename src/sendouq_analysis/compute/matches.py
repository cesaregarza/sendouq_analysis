from __future__ import annotations

from typing import TYPE_CHECKING

from sendouq_analysis.constants.columns import AGGREGATE, MATCHES

if TYPE_CHECKING:
    import pandas as pd


def get_matches_metadata(match_df: pd.DataFrame) -> pd.DataFrame:
    match_df = match_df.groupby(MATCHES.SEASON).agg(
        {
            MATCHES.MATCH_ID: ["min", "max", "count"],
            MATCHES.CREATED_AT: ["min", "max"],
        }
    )
    match_df.columns = AGGREGATE.MATCH_COLUMNS
    match_df = match_df.reset_index()
    return match_df
