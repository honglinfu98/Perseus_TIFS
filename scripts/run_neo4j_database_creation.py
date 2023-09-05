"""
This script is used to create the neo4j database from the json files

"""

import logging
import os
import pandas as pd
import psycopg2

logger = logging.getLogger()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

MAIN_DB_HOST = os.getenv('MAIN_DB_HOST')
MAIN_DB_PORT = os.getenv('MAIN_DB_PORT')
MAIN_DB_DIONYSUS_USER = os.getenv('MAIN_DB_DIONYSUS_USER')
MAIN_DB_DIONYSUS_PASSWORD = os.getenv('MAIN_DB_DIONYSUS_PASSWORD')
MAIN_DB_DB = os.getenv('MAIN_DB_DB') 

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

from clotho.neo4j.database_creation import (
    filter_nodes,
    filter_shared_values,
    read_json_file,
)
from clotho.neo4j.neo4j_conexion_and_methods import Neo4jConnection

USERS_QUERY = """
    WITH entity_score_mapping AS (
        SELECT entity_id, chat_crowd_score, chat_time_score
        FROM signals_signal s
        LEFT JOIN signals_pumpsignal ps ON ps.signal_id = s.id
    ),
    avg_scores AS (
        SELECT entity_id,
            AVG(
                CASE
                    WHEN chat_time_score IS NOT NULL then (chat_time_score)
                    ELSE 0
                END
            ) AS time_score,
            AVG(
                CASE
                    WHEN chat_crowd_score IS NOT NULL then (chat_crowd_score)
                    ELSE 0
                END
            ) AS crowd_score
        FROM entity_score_mapping
        GROUP BY entity_id
    ),
    ind_scores AS (
        SELECT cm.id as user_id,
            SUM(
                CASE
                    WHEN Lower(cm.flair) = 'administrator' THEN 0.5
                    ELSE 0
                END
            ) AS score_admin,
            SUM(
                CASE
                    WHEN Lower(cm.flair) = 'owner' THEN 0.25
                    ELSE 0
                END
            ) AS score_owner,
            SUM(
                CASE
                    WHEN Lower(cm.flair) = 'member' THEN 0.1
                    ELSE 0
                END
            ) AS score_member,
            sum(sc.time_score) as time_score,
            sum(sc.crowd_score) as crowd_score
        FROM community_data_telegramchatmember cm
            JOIN community_data_telegramchat tc ON cm.chat_id = tc.id
            JOIN avg_scores sc ON tc.entity_id = sc.entity_id
        GROUP BY 1
    )
    SELECT m."user_id" as id, u.name as username, u.phone_number as phone, 
        (
            s.score_admin + s.score_owner + s.score_member + s.time_score + s.crowd_score
        ) AS total_score,
        COUNT(distinct m.chat_id) as subscriptions,
        STRING_AGG(distinct m."chat_id"::text, ',')  as chat_list,
        COUNT(distinct uw.walletaddress_id) as wallets,
        STRING_AGG(distinct w.address || '|' || w.blockchain, ',') as wallet_list
    FROM community_data_telegramuser u
    INNER JOIN community_data_telegramchatmember m on u.user_id = m.user_id
    LEFT JOIN attribution_walletaddress_users uw ON m.user_id = uw.user_id
    LEFT JOIN attribution_walletaddress w ON uw.walletaddress_id = w.id
    LEFT JOIN ind_scores s on m.id = s.user_id
    GROUP BY m."user_id", u.name, phone, total_score
    ORDER BY wallets DESC, total_score, subscriptions DESC
    LIMIT 100
    """

CHATS_QUERY = """
    SELECT m.chat_id as id, c.username, c.title as channel, s.crowd_score, s.time_score,
        COUNT(m.chat_id) as subscribers
    FROM community_data_telegramchatmember m
    INNER JOIN community_data_telegramchat c ON m.chat_id = c.id
    LEFT JOIN community_data_telegramchatscore s ON m.chat_id = s.telegram_chat_id
    GROUP BY m.chat_id, c.username, c.title, s.crowd_score, s.time_score
    ORDER BY subscribers DESC
    LIMIT 100
    """

current_dir = os.getcwd()

#edges_file = os.path.abspath(
#    os.path.join(current_dir, "../data/channel_members_dict.json")
#)
#nodes_file = os.path.abspath(os.path.join(current_dir, "../data/users_featured.json"))
users_file = os.path.abspath(os.path.join(current_dir, "../data/users_subscription.json"))
chats_file = os.path.abspath(os.path.join(current_dir, "../data/chats_subscribers.json"))


def query_gaia(query, params={}) -> pd.DataFrame:
    with psycopg2.connect(f'dbname={MAIN_DB_DB} user={MAIN_DB_DIONYSUS_USER} password={MAIN_DB_DIONYSUS_PASSWORD} host={MAIN_DB_HOST} port={MAIN_DB_PORT}') as gaia_conn:
        return pd.read_sql_query(query, gaia_conn, params=params)

def add_nodes_user(conn: Neo4jConnection, dictionary_nodes: list, batch_size: int = 1000):
    for batch_start in range(0, len(dictionary_nodes), batch_size):
        batch = dictionary_nodes[batch_start : batch_start + batch_size]
        query = """
        UNWIND $nodes_batch AS node
        MERGE (n:User {id: node.id})
        SET n += {props: node.properties}.props
        """
        properties_batch = [
            {
                "id": node["id"],
                "properties": {k: v for k, v in node.items() if k != "id"},
            }
            for node in batch
        ]
        conn.query(query, parameters={"nodes_batch": properties_batch})
        logger.info(
            f"User Nodes {batch_start + 1}-{min(batch_start + batch_size, len(dictionary_nodes))}/{len(dictionary_nodes)} added to the database"
        )

def add_nodes(conn: Neo4jConnection, node_type: str, dictionary_nodes: list, batch_size: int = 1000):
    for batch_start in range(0, len(dictionary_nodes), batch_size):
        batch = dictionary_nodes[batch_start : batch_start + batch_size]
        query = """
        UNWIND $nodes_batch AS node
        MERGE (n:%s {id: node.id})
        SET n += {props: node.properties}.props
        """ % node_type
        properties_batch = [
            {
                "id": node["id"],
                "properties": {k: v for k, v in node.items() if k != "id"},
            }
            for node in batch
        ]
        conn.query(query, parameters={"nodes_batch": properties_batch})
        logger.info(
            f"{node_type} Nodes {batch_start + 1}-{min(batch_start + batch_size, len(dictionary_nodes))}/{len(dictionary_nodes)} added to the database"
        )

def add_edges_chats(conn: Neo4jConnection, dictionary_edges: list, batch_size: int = 1000):
    for batch_start in range(0, len(dictionary_edges), batch_size):
        batch = dictionary_edges[batch_start : batch_start + batch_size]
        query = """
        UNWIND $edges_batch AS edge
        MATCH (a:User {id: edge['user_id']}), (b:Chat {id: edge['chat_id']})
        MERGE (a)-[:SUBSCRIBES]-(b)
        """
        conn.query(query, parameters={"edges_batch": batch})
        logger.info(
            f"Chat Edges {batch_start + 1}-{min(batch_start + batch_size, len(dictionary_edges))}/{len(dictionary_edges)} added to the database"
    )

def add_edges_wallets(conn: Neo4jConnection, dictionary_edges: list, batch_size: int = 1000):
    for batch_start in range(0, len(dictionary_edges), batch_size):
        batch = dictionary_edges[batch_start : batch_start + batch_size]
        query = """
        UNWIND $edges_batch AS edge
        MATCH (a:User {id: edge['user_id']}), (b:Wallet {id: edge['wallet_id']})
        MERGE (a)-[:LINKED]-(b)
        """
        conn.query(query, parameters={"edges_batch": batch})
        logger.info(
            f"Wallet Edges {batch_start + 1}-{min(batch_start + batch_size, len(dictionary_edges))}/{len(dictionary_edges)} added to the database"
    )


if __name__ == "__main__":
    #edges = read_json_file(edges_file)
    #nodes = read_json_file(nodes_file)
    #users = read_json_file(users_file)
    #chats = read_json_file(chats_file)

    # load user and chat data from main db
    logger.info("Pulling user data from Gaia DB")
    users = query_gaia(USERS_QUERY).to_dict('records')
    logger.info("Pulling chat data from Gaia DB")
    chats = query_gaia(CHATS_QUERY).to_dict('records')

    #edges_filtered = filter_shared_values(edges, 7)
    #nodes_filtered = filter_nodes(nodes, edges_filtered)
    edges = []
    wallets = []
    edges_wallets = []
    for u in users:
        for c in u['chat_list'].split(','):
            edges.append({'user_id': u['id'], 'chat_id': int(c)})
        del u['chat_list']
        if u['wallet_list'] is not None:
            for w in u['wallet_list'].split(','):
                (wallet, blockchain) = w.split('|')
                id = len(wallets) + 1
                wallets.append({'id': id, 'wallet_id': id, 'address': wallet, 'blockchain': blockchain})
                edges_wallets.append({'user_id': u['id'], 'wallet_id': id})
            del u['wallet_list']

    with Neo4jConnection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD) as conn:  # type: ignore
        #conn.add_nodes(nodes_filtered, 1000)
        #conn.add_edges(edges_filtered, 2000)
        add_nodes_user(conn, users, 1000)
        add_nodes(conn, "Chat", chats, 1000)
        add_nodes(conn, "Wallet", wallets, 1000)
        add_edges_chats(conn, edges, 2000)
        add_edges_wallets(conn, edges_wallets, 2000)
        #conn.delete_nodes_by_pid(["1"])
