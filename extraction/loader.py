"""
This module loads the data from cloudburst tables
"""
import pandas as pd
from clotho_env.database_connection import CloudburstDataBaseConnection

def load_cloudburst_signals() -> pd.DataFrame:
    """
    This function loads the data from cloudburst and preprocess a bit

    Parameter:
    None

    Return:
    Return the retrieved and preprocessed data as pandas dataframe
    """
    # Query for extracting and filtering pumps from cloudburst data
    COLUMNS_NAMES_SIGNALS = [
        "pid",
        "entity_id",
        "signal_type",
        "source_datetime",
        "commodity",
        "channel_participants",
    ]

    QUERY_SIGNALS = ("SELECT " 
    + ", ".join(COLUMNS_NAMES_SIGNALS) 
    + """
    FROM cloudburst_signals WHERE signal_type IS NOT NULL and commodity IS NOT NULL
    """)

    # Query for extracting chats table
    COLUMNS_NAMES_CHATS = ["entity_id", "username"]

    query_chats =("SELECT "
    + ", ".join(COLUMNS_NAMES_CHATS)
    + """
    FROM telegram_chats
    """)

    # Creating the df with the column names
    tbl_signals = pd.DataFrame(
        list(CloudburstDataBaseConnection.fetch_data(QUERY_SIGNALS)),
        columns=COLUMNS_NAMES_SIGNALS,
    )

    tbl_chats = pd.DataFrame(
        list(CloudburstDataBaseConnection.fetch_data(query_chats)),
        columns=COLUMNS_NAMES_CHATS,
    )

    # Add name to the signals
    tbl_id_to_username = tbl_chats[["entity_id", "username"]]
    tbl_signals_named = tbl_signals.merge(
        tbl_id_to_username, left_on="entity_id", right_on="entity_id"
    )

    # Detected pump from cloudburst (late using dash to filter pumpdump and crowdpump)
    tbl_pump_detected = tbl_signals_named.loc[
        (tbl_signals_named["commodity"].notnull())
        & (tbl_signals_named["username"].notnull())
    ]

    processed_signals = tbl_pump_detected[
        [
            "pid",
            "entity_id",
            "signal_type",
            "source_datetime",
            "commodity",
            "username",
            "channel_participants",
        ]]
    
    return processed_signals


def load_cloudburst_members_data() -> pd.DataFrame:
    """
    This function loads the data from cloudburst and preprocess a bit

    Parameter:
    None

    Return:
    Return the retrieved and preprocessed data as pandas dataframe
    """

    MEMBERS_COLS = [
        "pid",
        '"chat_PID"',
        '"user_PID"',
        "meta_data",
        "joined_at",
        "created_at",
        "updated_at",
        "flair",
    ]

    MEMBERS_QUERY = (
        "SELECT "
        + ", ".join(MEMBERS_COLS)
        + """
    FROM telegram_chats_members
    """
    )
    tbl_members = pd.DataFrame(
        list(CloudburstDataBaseConnection.fetch_data(MEMBERS_QUERY)),
        columns=MEMBERS_COLS,
    )
    # Remove the double quotes from the column names
    tbl_members.columns = tbl_members.columns.str.replace('"', "")

    return tbl_members

def load_cloudburst_users_data() -> pd.DataFrame:
    """
    This function loads the data from cloudburst and preprocess a bit

    Parameter:
    None

    Return:
    Return the retrieved and preprocessed data as pandas dataframe
    """

    MEMBERS_COLS = [
        "pid",
        "entity_id",
        "username",
        "first_name",
        "last_name",
        "phone_number",
        "meta_data",
        "created_at",
        "updated_at",
        "bot",
        "premium",
        "last_online_at",
    ]

    MEMBERS_QUERY = (
        "SELECT "
        + ", ".join(MEMBERS_COLS)
        + """
    FROM telegram_users
    """
    )
    tbl_members = pd.DataFrame(
        list(CloudburstDataBaseConnection.fetch_data(MEMBERS_QUERY)),
        columns=MEMBERS_COLS,
    )

    return tbl_members
