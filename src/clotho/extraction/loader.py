"""
This module loads the data from cloudburst tables
"""
from clotho.extraction.cloudburst_connection import cloudburst_db_connection


# TODO add scrapper user_PID to an ignore list

### Query for extracting channels signals from cloudburst data#########

COLUMNS_NAMES_SIGNALS = [
    "pid",
    "entity_id",
    "signal_type",
    "source_datetime",
    "commodity",
    "channel_participants",
    "message_text",
    "channel_crowd_score",
    "channel_time_score",
]

QUERY_SIGNALS = (
    "SELECT "
    + ", ".join(COLUMNS_NAMES_SIGNALS)
    + """
FROM cloudburst_signals WHERE message_text IS NOT NULL
"""
)


def load_cloudburst_signals(signals_query: str = QUERY_SIGNALS) -> list[tuple]:
    """
    This function loads the signals from cloudburst database
    :param QUERY_SIGNALS: Query to extract the signals from cloudburst
    :return: List of tuples with the signals
    """
    cloudbust_signals = cloudburst_db_connection(signals_query)

    return cloudbust_signals


### Query for extracting channels data from cloudburst data###############

COLUMN_NAMES_CHANNELS = [
    "pid",
    "entity_id",
    "username",
    "title",
    "entity_type",
    "last_profile_updated",
]

QUERY_CHANNELS = (
    "SELECT "
    + ", ".join(COLUMN_NAMES_CHANNELS)
    + """
FROM telegram_chats
"""
)


def load_cloudburst_channels(channels_query: str = QUERY_CHANNELS) -> list[tuple]:
    """
    This function loads the signals from cloudburst database
    :param QUERY_CHANNELS: Query to extract the signals from cloudburst
    :return: List of tuples with the signals
    """
    cloudbust_channels = cloudburst_db_connection(channels_query)

    return cloudbust_channels


### Query for extracting users data from cloudburst data###############

COLUMNS_NAMES_USERS = [
    "pid",
    "entity_id",
    "username",
    "first_name",
    "last_name",
    "phone_number",
    "created_at",
    "updated_at",
    "bot",
    "premium",
    "bio",
    "last_online_at",
]

MEMBERS_QUERY = (
    "SELECT "
    + ", ".join(COLUMNS_NAMES_USERS)
    + """
FROM telegram_users
"""
)


def load_cloudburst_users(query_members: str = MEMBERS_QUERY) -> list[tuple]:
    """
    This function loads the signals from cloudburst database
    :param MEMBERS_QUERY: Query to extract the signals from cloudburst
    :return: List of tuples with the signals
    """
    cloudbust_members = cloudburst_db_connection(query_members)

    return cloudbust_members


### Query for extracting channel members data from cloudburst data###############

COLUMNS_NAMES_CHANNEL_MEMBERS = [
    "pid",
    "user_PID",
    "chat_PID",
    "joined_at",
    "updated_at",
    "created_at",
    "flair",
]

# Add to user_PID and chat_PID "" inside before query
# This is because the columns names have "" in the cloudburst table
# TODO Ask Alex to remove the "" from the cloudburst table when he has time
columns_names_for_quering = COLUMNS_NAMES_CHANNEL_MEMBERS.copy()
columns_names_for_quering[1] = f'"{COLUMNS_NAMES_CHANNEL_MEMBERS[1]}"'
columns_names_for_quering[2] = f'"{COLUMNS_NAMES_CHANNEL_MEMBERS[2]}"'

CHANNEL_MEMBERS_QUERY = (
    "SELECT "
    + ", ".join(columns_names_for_quering)
    + """
FROM telegram_chats_members
"""
)


def load_cloudburst_channel_members(
    query_channel_members: str = CHANNEL_MEMBERS_QUERY,
) -> list[tuple]:
    """
    This function loads the signals from cloudburst database
    :param CHANNEL_MEMBERS_QUERY: Query to extract the signals from cloudburst
    :return: List of tuples with the signals
    """
    cloudbust_channel_members = cloudburst_db_connection(query_channel_members)

    return cloudbust_channel_members


USERS_SCORES_COLUMNS = [
    "pid",
    "admin_score",
    "owner_score",
    "member_score",
    "time_score",
    "crowd_score",
    "total_score",
]

import os

# get path to sql file
sql_path = os.path.join(os.path.dirname(__file__), "users_score.sql")

with open(sql_path, "r") as f:
    QUERY_USERS_SCORE = f.read()


def load_cloudburst_users_score(
    users_score_query: str = QUERY_USERS_SCORE,
) -> list[tuple]:
    """
    This function use a query to extract the users scores from cloudburst db
    :param sql: Query to calculate the users scores
    :return: List of tuples with the users scores
    """
    # TODO see how to load only the column names from the query
    # Or we can just have them in a list fixed
    # because it's a really specific query

    cloudbust_users_score = cloudburst_db_connection(users_score_query)

    return cloudbust_users_score
