import pandas as pd

from sendouq_analysis.constants.columns import user_memento as um
from sendouq_analysis.constants import PREFERENCES
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
