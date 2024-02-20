"""
Download signals from Cloudburst database.
"""
import pickle
import os
import psycopg2
import pandas as pd
from clotho.config import (
    GAIA_DB_DB,
    GAIA_DB_HOST,
    GAIA_DB_PASSWORD,
    GAIA_DB_PORT,
    GAIA_DB_USER,
)
from clotho.settings import PROJECT_ROOT


# get path to sql file
sql_path = os.path.join(os.path.dirname(__file__), "scored_pumps.sql")
train_sql_path = os.path.join(os.path.dirname(__file__), "old_scored_pumps.sql")
valid_sql_path = os.path.join(os.path.dirname(__file__), "validate_scored_pumps.sql")
all_scored_sql_path = os.path.join(os.path.dirname(__file__), "all.sql")

mastermind_path = os.path.join(os.path.dirname(__file__), "old_mastermind.sql")
train_mastermind_path = os.path.join(os.path.dirname(__file__), "mastermind.sql")
valid_mastermind_path = os.path.join(os.path.dirname(__file__), "mastermind.sql")


direct_link = os.path.join(os.path.dirname(__file__), "direct_link.sql")

with open(sql_path, "r") as f:
    QUERY_SCORED_PUMPS = f.read()

with open(train_sql_path, "r") as f:
    TRAIN_QUERY_SCORED_PUMPS = f.read()

with open(valid_sql_path, "r") as f:
    VALID_QUERY_SCORED_PUMPS = f.read()

with open(all_scored_sql_path, "r") as f:
    ALL_QUERY_SCORED_PUMPS = f.read()



with open(mastermind_path, "r") as f:
    QUERY_MASTERMIND = f.read()

with open(train_mastermind_path, "r") as f:
    TRAIN_QUERY_MASTERMIND = f.read()

with open(direct_link, "r") as f:
    QUERY_DIRECT_LINK = f.read()


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

    # with open(os.path.join(PROJECT_ROOT, "src","clotho","dataset","extract","download","train_signals.pkl"), "rb") as file:
    #     result = pickle.load(file)

    return result



def get_scored_signals(query: str = QUERY_SCORED_PUMPS):
    """
    Get signals from the database for pre-pump scoring.
    """

    # conn = psycopg2.connect(
    #     f"dbname={GAIA_DB_DB} user={GAIA_DB_USER} password={GAIA_DB_PASSWORD} host={GAIA_DB_HOST} port={GAIA_DB_PORT}"
    # )

    # signals = pd.read_sql(query, conn)  # type: ignore

    # conn.close()

    # result = signals[~signals["telegram_chat_id"].isna()]
    with open(os.path.join(PROJECT_ROOT, "src","clotho","dataset","extract","download","test_signals.pkl"), "rb") as file:
        signals = pickle.load(file)
    result = signals[~signals["telegram_chat_id"].isna()]

    return result


def get_train_scored_signals(query: str = TRAIN_QUERY_SCORED_PUMPS):
    """
    Get signals from the database for pre-pump scoring.
    """

    # conn = psycopg2.connect(
    #     f"dbname={GAIA_DB_DB} user={GAIA_DB_USER} password={GAIA_DB_PASSWORD} host={GAIA_DB_HOST} port={GAIA_DB_PORT}"
    # )

    # signals = pd.read_sql(query, conn)  # type: ignore

    # conn.close()

    with open(os.path.join(PROJECT_ROOT, "src","clotho","dataset","extract","download","train_signals.pkl"), "rb") as file:
        signals = pickle.load(file)      
    result = signals[~signals["telegram_chat_id"].isna()]

    return result


def get_valid_scored_signals(query: str = VALID_QUERY_SCORED_PUMPS):
    """
    Get signals from the database for pre-pump scoring.
    """

    # conn = psycopg2.connect(
    #     f"dbname={GAIA_DB_DB} user={GAIA_DB_USER} password={GAIA_DB_PASSWORD} host={GAIA_DB_HOST} port={GAIA_DB_PORT}"
    # )

    # signals = pd.read_sql(query, conn)  # type: ignore

    # conn.close()

    with open(os.path.join(PROJECT_ROOT, "src","clotho","dataset","extract","download","validate_signals.pkl"), "rb") as file:
        signals = pickle.load(file)       
    result = signals[~signals["telegram_chat_id"].isna()]


    return result


def get_train_masterminds(query: str = TRAIN_QUERY_MASTERMIND):
    """
    Get masterminds from the database.
    """

    conn = psycopg2.connect(
        f"dbname={GAIA_DB_DB} user={GAIA_DB_USER} password={GAIA_DB_PASSWORD} host={GAIA_DB_HOST} port={GAIA_DB_PORT}"
    )

    mastermind = pd.read_sql(query, conn)  # type: ignore

    conn.close()
    # with open("train_mastermind.pkl", "rb") as file:
    #     mastermind = pickle.load(file)


    result = mastermind[~mastermind["telegram_chat_id"].isna()]
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
    # with open("mastermind.pkl", "rb") as file:
    #     mastermind = pickle.load(file)

    result = mastermind[~mastermind["telegram_chat_id"].isna()]

    return result


if __name__ == "__main__":
    # signals1 = get_scored_signals()
    # signals2 = get_train_masterminds()
    # signals3 = get_valid_scored_signals()    
    # mastermind = get_masterminds()
    # # train_signals = get_train_scored_signals()
    # train_mastermind = get_train_masterminds()

    # c = get_direct_link()


    signals1 = get_all_scored_signals()
    with open("all_scored.pkl", "wb") as file:
        pickle.dump(signals1, file)


    # with open("signals2.pkl", "wb") as file:
    #     pickle.dump(signals2, file)


    # with open("signals3.pkl", "wb") as file:
    #     pickle.dump(signals3, file)


    # with open("mastermind.pkl", "wb") as file:
    #     pickle.dump(mastermind, file)


    # with open("train_mastermind.pkl", "wb") as file:
    #     pickle.dump(train_mastermind, file)

