"""
Download data from Cloudburst.
"""

import os
import psycopg2
import pandas as pd
from perseus.config import (
    GAIA_DB_DB,
    GAIA_DB_HOST,
    GAIA_DB_PASSWORD,
    GAIA_DB_PORT,
    GAIA_DB_USER,
)


all_scored_sql_path = os.path.join(os.path.dirname(__file__), "all.sql")
direct_link = os.path.join(os.path.dirname(__file__), "direct_link.sql")
volume = os.path.join(os.path.dirname(__file__), "volume.sql")
comparison = os.path.join(os.path.dirname(__file__), "timevscrowd.sql")
new_detection = os.path.join(os.path.dirname(__file__), "new_detection.sql")
group_name = os.path.join(os.path.dirname(__file__), "group_name.sql")
signals_channel = os.path.join(os.path.dirname(__file__), "timecrowd_channel.sql")
btc_base = os.path.join(os.path.dirname(__file__), "btc_query.sql")


with open(all_scored_sql_path, "r") as f:
    ALL_QUERY_SCORED_PUMPS = f.read()
with open(direct_link, "r") as f:
    QUERY_DIRECT_LINK = f.read()
with open(volume, "r") as f:
    QUERY_VOLUME = f.read()
with open(comparison, "r") as f:
    QUERY_COMPARISON = f.read()
with open(new_detection, "r") as f:
    QUERY_NEW_DETECTION = f.read()
with open(group_name, "r") as f:
    QUERY_GROUP_NAME = f.read()
with open(signals_channel, "r") as f:
    QUERY_SIGNALS_CHANNEL = f.read()
with open(btc_base, "r") as f:
    QUERY_BTC_BASE = f.read()


def get_direct_link(query: str = QUERY_DIRECT_LINK):
    """
    Get direct link from the database for pre-pump scoring.
    """

    conn = psycopg2.connect(
        f"dbname={GAIA_DB_DB} user={GAIA_DB_USER} password={GAIA_DB_PASSWORD} host={GAIA_DB_HOST} port={GAIA_DB_PORT}"
    )

    links = pd.read_sql(query, conn)  # type: ignore

    conn.close()

    return links


def get_all_scored_signals(query: str = ALL_QUERY_SCORED_PUMPS):
    """
    Get signals from the database for pre-pump scoring.
    """

    conn = psycopg2.connect(
        f"dbname={GAIA_DB_DB} user={GAIA_DB_USER} password={GAIA_DB_PASSWORD} host={GAIA_DB_HOST} port={GAIA_DB_PORT}"
    )

    signals = pd.read_sql(query, conn)  # type: ignore

    conn.close()

    result = signals[~signals["telegram_chat_id"].isna()]

    return result


def get_new_detection(query: str = QUERY_NEW_DETECTION):
    """
    Get signals from the database for pre-pump scoring.
    """

    conn = psycopg2.connect(
        f"dbname={GAIA_DB_DB} user={GAIA_DB_USER} password={GAIA_DB_PASSWORD} host={GAIA_DB_HOST} port={GAIA_DB_PORT}"
    )

    signals = pd.read_sql(query, conn)  # type: ignore

    conn.close()

    result = signals[~signals["telegram_chat_id"].isna()]

    return result


def get_volumes(query: str = QUERY_VOLUME):
    """
    Get volumes from the database for pre-pump scoring.
    """

    conn = psycopg2.connect(
        f"dbname={GAIA_DB_DB} user={GAIA_DB_USER} password={GAIA_DB_PASSWORD} host={GAIA_DB_HOST} port={GAIA_DB_PORT}"
    )

    result = pd.read_sql(query, conn)  # type: ignore

    conn.close()

    return result


def get_comparison(query: str = QUERY_COMPARISON):
    """
    Get volumes from the database for pre-pump scoring.
    """

    conn = psycopg2.connect(
        f"dbname={GAIA_DB_DB} user={GAIA_DB_USER} password={GAIA_DB_PASSWORD} host={GAIA_DB_HOST} port={GAIA_DB_PORT}"
    )

    result = pd.read_sql(query, conn)  # type: ignore

    conn.close()

    return result


def get_groupname(query: str = QUERY_GROUP_NAME):
    """
    Get volumes from the database for pre-pump scoring.
    """

    conn = psycopg2.connect(
        f"dbname={GAIA_DB_DB} user={GAIA_DB_USER} password={GAIA_DB_PASSWORD} host={GAIA_DB_HOST} port={GAIA_DB_PORT}"
    )

    result = pd.read_sql(query, conn)  # type: ignore

    conn.close()

    return result


def get_signals_channel(query: str = QUERY_SIGNALS_CHANNEL):
    """
    Get volumes from the database for pre-pump scoring.
    """

    conn = psycopg2.connect(
        f"dbname={GAIA_DB_DB} user={GAIA_DB_USER} password={GAIA_DB_PASSWORD} host={GAIA_DB_HOST} port={GAIA_DB_PORT}"
    )

    result = pd.read_sql(query, conn)  # type: ignore

    conn.close()

    return result


def get_btc_base(query: str = QUERY_BTC_BASE):
    """
    Get BTC signals from the database.
    """

    conn = psycopg2.connect(
        f"dbname={GAIA_DB_DB} user={GAIA_DB_USER} password={GAIA_DB_PASSWORD} host={GAIA_DB_HOST} port={GAIA_DB_PORT}"
    )

    result = pd.read_sql(query, conn)  # type: ignore

    conn.close()

    return result


if __name__ == "__main__":
    signal_channel = get_btc_base()
