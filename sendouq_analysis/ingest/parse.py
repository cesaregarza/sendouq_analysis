from __future__ import annotations

import pandas as pd

from sendouq_analysis.constants import PREFERENCES
from sendouq_analysis.constants.columns import MAP_LIST, MATCHES
from sendouq_analysis.constants.columns import user_memento as um
from sendouq_analysis.constants.json_keys import MATCH
from sendouq_analysis.utils import camel_to_snake


def parse_memento(memento: dict) -> tuple[pd.DataFrame, ...]:
    """Parses the "memento" field from a match, which contains the match data
    related to SP. Returns two DataFrames, one for the group data and one for
    the user data. If "Map Preferences" is part of the memento, a third
    DataFrame is returned.

    Args:
        memento (dict): The "memento" field from a match

    Returns:
        tuple:
            - pd.DataFrame: The group data
            - pd.DataFrame: The user data
            - pd.DataFrame: The map preferences data, if present
    """
    groups_data = []
    for group_id, details in memento["groups"].items():
        group_info = details.copy()
        group_info["groupId"] = group_id
        groups_data.append(group_info)

    groups_df = pd.json_normalize(groups_data, sep="_").rename(
        columns=camel_to_snake
    )

    # Parse users
    users_data = []
    for user_id, details in memento["users"].items():
        user_info = details.copy()
        user_info["userId"] = user_id
        users_data.append(user_info)

    users_df = pd.json_normalize(users_data, sep="_").rename(
        columns=camel_to_snake
    )
    users_df[um.USER_ID] = users_df[um.USER_ID].astype(int)

    user_ids = users_df[um.USER_ID].unique()

    if "mapPreferences" not in memento:
        return groups_df, users_df

    # Parse map preferences
    map_data = []
    for map_id, details in enumerate(memento["mapPreferences"]):
        map_info = pd.json_normalize(details).rename(columns=camel_to_snake)
        map_info["map_id"] = map_id
        map_info[um.USER_ID] = map_info[um.USER_ID].astype(int)
        # Add missing users
        missing_users = set(user_ids) - set(map_info[um.USER_ID])
        if missing_users:
            missing_df = pd.DataFrame(
                [
                    {
                        um.USER_ID: user,
                        "map_id": map_id,
                        "preference": PREFERENCES.IMPLICIT_INDIFFERENT,
                    }
                    for user in missing_users
                ]
            )
            map_info = pd.concat([map_info, missing_df])
        map_data.append(map_info)

    map_df = pd.concat(map_data).fillna(PREFERENCES.EXPLICIT_INDIFFERENT)
    map_df[um.USER_ID] = map_df[um.USER_ID].astype(int)

    return groups_df, users_df, map_df


def parse_match_json(
    match_json: dict,
) -> tuple[
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | None,
]:
    """Parses a match JSON to return the match data, the group data, the user
    data and the map preferences data.

    Args:
        match_json (dict): The match JSON

    Returns:
        tuple:
            - pd.Series: The match data
            - pd.DataFrame: The group data
            - pd.DataFrame: The user data
            - pd.DataFrame: The map data
            - pd.DataFrame | None: The map preferences data, if present
    """
    id = match_json[MATCH.ID]
    alpha_team_id = match_json[MATCH.ALPHA_GROUP_ID]
    bravo_team_id = match_json[MATCH.BRAVO_GROUP_ID]
    created_at = match_json[MATCH.CREATED_AT]
    reported_at = match_json[MATCH.REPORTED_AT]
    reported_by = match_json[MATCH.REPORTED_BY]
    try:
        mementos = parse_memento(match_json[MATCH.MEMENTO])
        if len(mementos) == 2:
            group_memento, user_memento = mementos
            map_memento = None
        else:
            group_memento, user_memento, map_memento = mementos
    except (KeyError, TypeError):
        group_memento = pd.DataFrame()
        user_memento = pd.DataFrame()
        map_memento = None

    map_df = pd.DataFrame(match_json[MATCH.MAP_LIST])

    match_df = pd.Series(
        {
            MATCHES.MATCH_ID: id,
            MATCHES.ALPHA_TEAM_ID: alpha_team_id,
            MATCHES.BRAVO_TEAM_ID: bravo_team_id,
            MATCHES.CREATED_AT: created_at,
            MATCHES.REPORTED_AT: reported_at,
            MATCHES.REPORTED_BY_USER_ID: reported_by,
        }
    )
    group_memento[MATCHES.MATCH_ID] = id
    user_memento[MATCHES.MATCH_ID] = id
    map_df[MATCHES.MATCH_ID] = id

    winner = calculate_winner(map_df)
    match_df[MATCHES.WINNER_ID] = winner
    if winner == "cancelled":
        match_df[MATCHES.WINNER] = "cancelled"
    elif int(winner) == alpha_team_id:
        match_df[MATCHES.WINNER] = "alpha"
    elif int(winner) == bravo_team_id:
        match_df[MATCHES.WINNER] = "bravo"

    return match_df, group_memento, user_memento, map_df, map_memento


def calculate_winner(map_list: pd.DataFrame) -> str:
    """
    Calculates the winner based on the provided map list.

    Args:
        map_list (pd.DataFrame): The DataFrame containing the map list.

    Returns:
        str: The ID of the winner as a string. If the match was cancelled, the
            string "cancelled" is returned.
    """
    try:
        return (
            map_list.groupby(MAP_LIST.WINNER_GROUP_ID)[MAP_LIST.ID]
            .count()
            .idxmax()
            .astype(int)
            .astype(str)
        )
    except ValueError:
        return "cancelled"
