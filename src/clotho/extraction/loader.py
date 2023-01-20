"""
This module loads the data from cloudburst tables
"""
from clotho.extraction.cloudburst_connection import CloudburstDataBaseConnection

# TODO add scrapper user_PID to an ignore list

# Query for extracting and filtering pumps from cloudburst data
COLUMNS_NAMES_SIGNALS = [
    "pid",
    "entity_id",
    "signal_type",
    "source_datetime",
    "commodity",
    "channel_participants",
    "message_text",
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
    cloudbust_signals = CloudburstDataBaseConnection.fetch_data(signals_query)

    return cloudbust_signals


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
    "last_online_at",
]

MEMBERS_QUERY = (
    "SELECT "
    + ", ".join(COLUMNS_NAMES_USERS)
    + """
FROM telegram_users
"""
)


def load_cloudburst_members(MEMBERS_QUERY: str) -> list[tuple]:
    """
    This function loads the signals from cloudburst database
    :param MEMBERS_QUERY: Query to extract the signals from cloudburst
    :return: List of tuples with the signals
    """
    cloudbust_members = CloudburstDataBaseConnection().fetch_data(MEMBERS_QUERY)

    return cloudbust_members
