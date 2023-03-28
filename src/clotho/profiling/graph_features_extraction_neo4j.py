"""
In this module, we extract the features from the Neo4j graph and add it to the user information.
This module only work as a local conection to the Neo4j graph.
Neo4j must be open and the graph must be created.
"""
from neo4j import GraphDatabase
from py2neo import Node
from clotho.config import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    URL_NODES,
    URL_EDGES,
    AURA_INSTANCENAME,
)
from neo4j import GraphDatabase
import logging
from neo4j.exceptions import ServiceUnavailable

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()
from neo4j import GraphDatabase
from typing import Dict
import csv
import json


# read a jsonfile and store it
def read_json_file(json_file: str) -> Dict:
    """
    This function reads a json file and returns a dictionary with the information.
    :param json_file: the path of the json file
    :return: the dictionary with the information
    """
    with open(json_file, "r") as f:
        return json.load(f)


def filter_shared_values(dictionary_list, min_shared):
    filtered_list = []
    for d in dictionary_list:
        if d["shared"] >= min_shared:
            filtered_list.append(d)
    return filtered_list


def filter_nodes(nodes, edges):
    pid_set = set()
    for edge in edges:
        pid_set.add(edge["user1_PID"])
        pid_set.add(edge["user2_PID"])
    nodes_filtered = [node for node in nodes if node["pid"] in pid_set]
    return nodes_filtered


class Neo4jConnection:
    """
    This class is used to create a connection to the graph database.
    """

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        # Don't forget to close the driver connection when you are finished with it
        self.driver.close()

    def query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return result.data()

    def add_nodes(self, nodes_filtered, batch_size=100):
        for batch_start in range(0, len(nodes_filtered), batch_size):
            batch = nodes_filtered[batch_start : batch_start + batch_size]
            query = """
            UNWIND $nodes_batch AS node
            MERGE (n:User {pid: node.pid})
            SET n += {props: node.properties}.props
            """
            properties_batch = [
                {
                    "pid": node["pid"],
                    "properties": {k: v for k, v in node.items() if k != "pid"},
                }
                for node in batch
            ]
            self.query(query, parameters={"nodes_batch": properties_batch})
            logger.info(
                f"Nodes {batch_start + 1}-{min(batch_start + batch_size, len(nodes_filtered))}/{len(nodes_filtered)} added to the database"
            )

    def add_edges(self, edges_filtered, batch_size=1000):
        for batch_start in range(0, len(edges_filtered), batch_size):
            batch = edges_filtered[batch_start : batch_start + batch_size]
            query = """
            UNWIND $edges_batch AS edge
            MATCH (a:User {pid: edge.user1_PID}), (b:User {pid: edge.user2_PID})
            MERGE (a)-[r:SHARED {shared: edge.shared}]-(b)
            """
            self.query(query, parameters={"edges_batch": batch})
            logger.info(
                f"Edges {batch_start + 1}-{min(batch_start + batch_size, len(edges_filtered))}/{len(edges_filtered)} added to the database"
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


if __name__ == "__main__":
    edges = read_json_file(
        "C:/Users/nagge/Desktop/Nico/Cloudburst/clotho/data/channel_members_dict.json"
    )
    nodes = read_json_file(
        "C:/Users/nagge/Desktop/Nico/Cloudburst/clotho/data/users_featured.json"
    )
    edges_filtered = filter_shared_values(edges, 7)
    nodes_filtered = filter_nodes(nodes, edges_filtered)

    with Neo4jConnection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD) as conn:
        conn.add_nodes(nodes_filtered)
        conn.add_edges(edges_filtered)
