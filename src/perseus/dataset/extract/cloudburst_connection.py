"""
Download data from Cloudburst.
"""

from os import path
import os
import pickle
import psycopg2
import pandas as pd
from perseus.config import (
    GAIA_DB_DB,
    GAIA_DB_HOST,
    GAIA_DB_PASSWORD,
    GAIA_DB_PORT,
    GAIA_DB_USER,
)
from perseus.settings import PROJECT_ROOT


all_scored_sql_path = os.path.join(os.path.dirname(__file__), "all.sql")
direct_link = os.path.join(os.path.dirname(__file__), "direct_link.sql")
volume = os.path.join(os.path.dirname(__file__), "volume.sql")
comparison = os.path.join(os.path.dirname(__file__), "timevscrowd.sql")
new_detection = os.path.join(os.path.dirname(__file__), "new_detection.sql")

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


if __name__ == "__main__":
    all = get_volumes()
    # with open(path.join(PROJECT_ROOT, "data", "new_detection.pkl"), "wb") as file:
    #     pickle.dump(all, file)
