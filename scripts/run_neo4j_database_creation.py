"""
This script is used to create the neo4j database from the json files

"""

import logging
import os

logger = logging.getLogger()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
from clotho.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME
from clotho.neo4j.database_creation import (
    filter_nodes,
    filter_shared_values,
    read_json_file,
)
from clotho.neo4j.neo4j_conexion_and_methods import Neo4jConnection

current_dir = os.getcwd()

edges_file = os.path.abspath(
    os.path.join(current_dir, "../data/channel_members_dict.json")
)
nodes_file = os.path.abspath(os.path.join(current_dir, "../data/users_featured.json"))


if __name__ == "__main__":
    edges = read_json_file(edges_file)
    nodes = read_json_file(nodes_file)

    edges_filtered = filter_shared_values(edges, 7)
    nodes_filtered = filter_nodes(nodes, edges_filtered)

    with Neo4jConnection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD) as conn:  # type: ignore
        conn.add_nodes(nodes_filtered, 1000)
        conn.add_edges(edges_filtered, 2000)
        conn.delete_nodes_by_pid(["1"])
