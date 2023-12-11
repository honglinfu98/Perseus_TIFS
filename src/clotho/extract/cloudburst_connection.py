"""
Download signals from Cloudburst database.
"""
import pickle
import os
import psycopg2
import pandas as pd

GAIA_DB_HOST = os.getenv("GAIA_DB_HOST")
GAIA_DB_PORT = os.getenv("GAIA_DB_PORT")
GAIA_DB_USER = os.getenv("GAIA_DB_USER")
GAIA_DB_PASSWORD = os.getenv("GAIA_DB_PASSWORD")
GAIA_DB_DB = os.getenv("GAIA_DB_DB")


# get path to sql file
sql_path = os.path.join(os.path.dirname(__file__), "scored_pumps.sql")

mastermind_path = os.path.join(os.path.dirname(__file__), "mastermind.sql")

with open(sql_path, "r") as f:
    QUERY_SCORED_PUMPS = f.read()


with open(mastermind_path, "r") as f:
    QUERY_MASTERMIND = f.read()


def get_scored_signals(query: str = QUERY_SCORED_PUMPS):
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


def get_masterminds(query: str = QUERY_MASTERMIND):
    """
    Get masterminds from the database.
    """

    conn = psycopg2.connect(
        f"dbname={GAIA_DB_DB} user={GAIA_DB_USER} password={GAIA_DB_PASSWORD} host={GAIA_DB_HOST} port={GAIA_DB_PORT}"
    )

    mastermind = pd.read_sql(query, conn)  # type: ignore

    conn.close()

    result = mastermind[~mastermind["telegram_chat_id"].isna()]

    return result


if __name__ == "__main__":
    # signals = get_scored_signals()
    mastermind = get_masterminds()

    # with open("signals.pkl", "wb") as file:
    #     pickle.dump(signals, file)
