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


with open(all_scored_sql_path, "r") as f:
    ALL_QUERY_SCORED_PUMPS = f.read()
with open(direct_link, "r") as f:
    QUERY_DIRECT_LINK = f.read()
with open(volume, "r") as f:
    QUERY_VOLUME = f.read()


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


if __name__ == "__main__":
    volume = get_volumes()
    with open(path.join(PROJECT_ROOT, "data", "volume.pkl"), "wb") as file:
        pickle.dump(volume, file)
