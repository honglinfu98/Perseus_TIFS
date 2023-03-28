"""
In this module, we extract the features from the Neo4j graph and add it to the user information.
This module only work as a local conection to the Neo4j graph.
Neo4j must be open and the graph must be created.
"""
import logging


from clotho.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME
from clotho.neo4j.neo4j_conexion_and_methods import Neo4jConnection

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()
import json


# read a jsonfile and store it
def read_json_file(json_file: str) -> list:
    """
    This function reads a json file and returns a list of dictionaries with the information.
    :param json_file: the path of the json file
    :return: the list of dictionaries with the information
    """
    with open(json_file, "r", encoding="utf-8") as document:
        return json.load(document)


def filter_shared_values(dictionary_list: list, min_shared: int) -> list:
    filtered_list = []
    for dictionary in dictionary_list:
        if dictionary["shared"] >= min_shared:
            filtered_list.append(dictionary)
    return filtered_list


def filter_nodes(nodes_list: list, edges: list) -> list:
    pid_set = set()
    for edge in edges:
        pid_set.add(edge["user1_PID"])
        pid_set.add(edge["user2_PID"])
    filtered_nodes = [node for node in nodes_list if node["pid"] in pid_set]
    return filtered_nodes


if __name__ == "__main__":
    edges = read_json_file(
        "C:/Users/nagge/Desktop/Nico/Cloudburst/clotho/data/channel_members_dict.json"
    )
    nodes = read_json_file(
        "C:/Users/nagge/Desktop/Nico/Cloudburst/clotho/data/users_featured.json"
    )
    edges_filtered = filter_shared_values(edges, 7)
    nodes_filtered = filter_nodes(nodes, edges_filtered)

    with Neo4jConnection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD) as conn:  # type: ignore
        conn.add_nodes(nodes_filtered)
        conn.add_edges(edges_filtered)
