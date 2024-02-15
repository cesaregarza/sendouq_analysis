from sendouq_analysis.constants.columns import matches as match_c
from sendouq_analysis.constants.columns import user_memento as user_c
from sendouq_analysis.constants.table_names import (
    AGGREGATE_LATEST_PLAYER_STATS,
    AGGREGATE_PLAYER_STATS,
    AGGREGATE_SCHEMA,
)

create_latest_player_stats_query = f"""
INSERT INTO {AGGREGATE_SCHEMA}.{AGGREGATE_LATEST_PLAYER_STATS} (
    {match_c.SEASON},
    {user_c.USER_ID},
    {user_c.SP},
    {match_c.CREATED_AT}
)
SELECT
    {match_c.SEASON},
    {user_c.USER_ID},
    {user_c.SP},
    {match_c.CREATED_AT}
FROM
    (
        SELECT
            {match_c.SEASON},
            {user_c.USER_ID},
            {user_c.SP},
            {match_c.CREATED_AT},
            ROW_NUMBER() OVER (
                                PARTITION BY {user_c.USER_ID}
                                ORDER BY {match_c.CREATED_AT}
                                DESC
                                ) AS rn
        FROM
            {AGGREGATE_SCHEMA}.{AGGREGATE_PLAYER_STATS}
            ) AS sub
WHERE
    rn = 1
ON CONFLICT ({user_c.USER_ID}) DO UPDATE
SET
    {match_c.SEASON} = EXCLUDED.{match_c.SEASON},
    {user_c.SP} = EXCLUDED.{user_c.SP},
    {match_c.CREATED_AT} = EXCLUDED.{match_c.CREATED_AT};
"""
