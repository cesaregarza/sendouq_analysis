from __future__ import annotations

import logging
import warnings

import pandas as pd
import tqdm

from sendouq_analysis.constants import DATA, JSON_KEYS, PREFERENCES
from sendouq_analysis.constants.columns import (
    GROUPS,
    MAP_LIST,
    MATCHES,
    WEAPONS,
)
from sendouq_analysis.constants.columns import user_memento as um
from sendouq_analysis.utils import camel_to_snake

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="pandas")


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
    logger.info("Parsing memento")
    groups_data = []
    for group_id, details in memento[JSON_KEYS.GROUPS].items():
        group_info = details.copy()
        group_info[JSON_KEYS.GROUP_ID] = group_id
        groups_data.append(group_info)

    groups_df = pd.json_normalize(groups_data, sep="_").rename(
        columns=camel_to_snake
    )

    # Parse users
    users_data = []
    for user_id, details in memento[JSON_KEYS.USERS].items():
        user_info = details.copy()
        user_info[JSON_KEYS.USER_ID] = user_id
        users_data.append(user_info)

    users_df = pd.json_normalize(users_data, sep="_").rename(
        columns=camel_to_snake
    )
    users_df[um.USER_ID] = users_df[um.USER_ID].astype(int)

    user_ids = users_df[um.USER_ID].unique()

    if JSON_KEYS.MAP_PREFERENCES not in memento:
        return groups_df, users_df

    # Parse map preferences
    map_data = []
    for map_index, details in enumerate(memento[JSON_KEYS.MAP_PREFERENCES]):
        map_info = pd.json_normalize(details).rename(columns=camel_to_snake)
        map_info[um.MAP_INDEX] = map_index
        map_info[um.USER_ID] = map_info[um.USER_ID].astype(int)
        # Add missing users
        missing_users = set(user_ids) - set(map_info[um.USER_ID])
        if missing_users:
            missing_df = pd.DataFrame(
                [
                    {
                        um.USER_ID: user,
                        um.MAP_INDEX: map_index,
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
    logger.info("Parsing match JSON")
    id = match_json[JSON_KEYS.ID]
    alpha_team_id = match_json[JSON_KEYS.ALPHA_GROUP_ID]
    bravo_team_id = match_json[JSON_KEYS.BRAVO_GROUP_ID]
    created_at = match_json[JSON_KEYS.CREATED_AT]
    reported_at = match_json[JSON_KEYS.REPORTED_AT]
    reported_by = match_json[JSON_KEYS.REPORTED_BY]
    try:
        mementos = parse_memento(match_json[JSON_KEYS.MEMENTO])
        if len(mementos) == 2:
            group_memento, user_memento = mementos
            map_memento = None
        else:
            group_memento, user_memento, map_memento = mementos
    except (KeyError, TypeError):
        group_memento = pd.DataFrame()
        user_memento = pd.DataFrame()
        map_memento = None

    map_df = pd.DataFrame(match_json[JSON_KEYS.MAP_LIST]).rename(
        columns=camel_to_snake
    )

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
    if winner == DATA.CANCELLED:
        match_df[MATCHES.WINNER] = DATA.CANCELLED
    elif int(winner) == alpha_team_id:
        match_df[MATCHES.WINNER] = DATA.ALPHA
    elif int(winner) == bravo_team_id:
        match_df[MATCHES.WINNER] = DATA.BRAVO

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
    logger.info("Calculating winner")
    try:
        return (
            map_list.groupby(MAP_LIST.WINNER_GROUP_ID)[MAP_LIST.ID]
            .count()
            .idxmax()
            .astype(int)
            .astype(str)
        )
    except ValueError:
        return DATA.CANCELLED


def parse_group_members(group_data: dict) -> pd.DataFrame:
    """
    Parse the group members data and return it as a pandas DataFrame.

    Args:
        group_data (dict): The group data containing information about the group members.

    Returns:
        pd.DataFrame: A DataFrame containing the parsed group members data.
    """
    logger.info("Parsing group members")
    group_id = group_data[JSON_KEYS.ID]
    members_list = []

    for member in group_data[JSON_KEYS.MEMBERS]:
        member_data: dict = member.copy()

        member_data.pop(JSON_KEYS.DISCORD_AVATAR, None)
        member_data.pop(JSON_KEYS.WEAPONS, None)
        member_data.pop(JSON_KEYS.SKILL, None)
        member_data.pop(JSON_KEYS.SKILL_DIFFERENCE, None)

        member_data[JSON_KEYS.GROUP_ID] = group_id
        members_list.append(member_data)

    return pd.DataFrame(members_list)


def parse_groups(full_json: dict) -> pd.DataFrame:
    """
    Parse the group members from the given JSON and return a DataFrame
    containing the parsed data.

    Args:
        full_json (dict): The JSON containing the group data.

    Returns:
        pd.DataFrame: A DataFrame containing the parsed group members.
    """
    logger.info("Parsing groups")
    alpha = parse_group_members(full_json[JSON_KEYS.GROUP_ALPHA])
    bravo = parse_group_members(full_json[JSON_KEYS.GROUP_BRAVO])
    alpha[GROUPS.TEAM] = DATA.ALPHA
    bravo[GROUPS.TEAM] = DATA.BRAVO
    return pd.concat([alpha, bravo])


def parse_json(
    full_json: dict,
) -> tuple[
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | None,
    pd.DataFrame | None,
]:
    """Parses a full JSON from the API to return the match data, the group
    data, the user data and the map preferences data.

    Args:
        full_json (dict): The full JSON from the API

    Returns:
        tuple:
            - pd.Series: The match data
            - pd.DataFrame: The group memento data
            - pd.DataFrame: The user memento data
            - pd.DataFrame: The map data
            - pd.DataFrame: The group data
            - pd.DataFrame | None: The map preferences data, if present
            - pd.DataFrame | None: The weapons data, if present
    """
    id = full_json[JSON_KEYS.MATCH][JSON_KEYS.ID]
    logger.info("Parsing JSON, id: %s", id)
    (
        match_df,
        group_memento,
        user_memento,
        map_df,
        map_memento,
    ) = parse_match_json(full_json[JSON_KEYS.MATCH])
    group_df = (
        parse_groups(full_json)
        .rename(columns=camel_to_snake)
        .rename(columns={JSON_KEYS.ID: GROUPS.USER_ID})
    )
    try:
        user_memento = user_memento.merge(
            group_df[[GROUPS.GROUP_ID, GROUPS.USER_ID]],
            on=GROUPS.USER_ID,
            how="left",
        )
    except KeyError:
        user_memento = None

    try:
        weapons_df = pd.DataFrame(
            full_json[JSON_KEYS.RAW_REPORTED_WEAPONS]
        ).rename(columns=camel_to_snake)
        weapons_df[WEAPONS.MATCH_ID] = id
    except KeyError:
        weapons_df = None

    return (
        match_df,
        group_memento,
        user_memento,
        map_df,
        group_df,
        map_memento,
        weapons_df,
    )


def parse_all(
    jsons: list[dict],
    *,
    disable_tqdm: bool = False,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | None,
    pd.DataFrame | None,
]:
    """Parses a list of JSONs from the API to return the match data, the group
    data, the user data and the map preferences data.

    Args:
        jsons (list[dict]): The list of JSONs from the API
        disable_tqdm (bool): Whether to disable the tqdm progress bar

    Returns:
        tuple:
            - pd.DataFrame: The match data
            - pd.DataFrame: The group memento data
            - pd.DataFrame: The user memento data
            - pd.DataFrame: The map data
            - pd.DataFrame: The group data
            - pd.DataFrame | None: The map preferences data, if present
            - pd.DataFrame | None: The weapons data, if present
    """
    logger.info("Parsing JSONs")
    data = [
        list(parse_json(json))
        for json in tqdm.tqdm(jsons, disable=disable_tqdm)
    ]
    logger.info("Transposing data")
    data = list(zip(*data))  # Transpose the list of lists
    # return tuple(pd.concat(d, ignore_index=True) for d in data)
    logger.info("Concatenating data")
    out = []
    for i, d in enumerate(data):
        if i == 0:
            concat = pd.concat(d, axis=1).T
        else:
            try:
                concat = pd.concat(d, ignore_index=True)
            except ValueError:
                concat = pd.DataFrame()
        out.append(concat)
    return tuple(out)
