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


edges = read_json_file(
    "C:/Users/nagge/Desktop/Nico/Cloudburst/clotho/data/channel_members_dict.json"
)
nodes = read_json_file(
    "C:/Users/nagge/Desktop/Nico/Cloudburst/clotho/data/users_featured.json"
)


def filter_shared_values(dictionary_list, min_shared):
    filtered_list = []
    for d in dictionary_list:
        if d["shared"] >= min_shared:
            filtered_list.append(d)
    return filtered_list


edges_filtered = filter_shared_values(edges, 7)


def filter_nodes(nodes, edges):
    pid_set = set()
    for edge in edges:
        pid_set.add(edge["user1_PID"])
        pid_set.add(edge["user2_PID"])
    print(len(pid_set))
    print(pid_set)
    nodes_filtered = [node for node in nodes if node["pid"] in pid_set]
    return nodes_filtered


nodes_filtered = filter_nodes(nodes, edges_filtered)
# how many nodes are in the nodes_filtered list?
print(len(nodes_filtered))

# filter the nodes that are not in the edges
nodes_filtered = []
for node in nodes:
    for edge in edges_filtered:
        if node["pid"] == edge["user1_PID"] or node["pid"] == edge["user2_PID"]:
            nodes_filtered.append(node)
            break


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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def create_user_nodes_and_edges_from_csv(
    nodes: str = URL_NODES, edges: str = URL_EDGES  # type: ignore
):
    """
    This function creates the nodes from a csv file.
    :param csv_url: the url of the csv file
    """
    # Crea una consulta para crear los nodos

    query = """LOAD CSV WITH HEADERS FROM https://drive.google.com/file/d/1QN644Rw4E-RBultyOsBxhdrC50p6S7nS/view?usp=share_link AS row CREATE (:User {user_id: toInteger(row.pid), admin_score: row.admin_score, owner_score: row.owner_score, member_score: row.member_score, time_score: row.time_score, crowd_score: row.crowd_score, total_score: row.total_score}) """

    # Usa la clase Neo4jConnection para ejecutar la consulta
    with Neo4jConnection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD) as connection:
        connection.query(query)

    # query = f"""
    # LOAD CSV WITH HEADERS FROM '{edges}' AS row
    # MATCH (User1:User {{user_id: toInteger(row.user1_PID)}}),
    # (User2:User {{user_id: toInteger(row.user2_PID)}})
    # CREATE (User1)-[:CONNECT_TO {{importance: toFloat(row.shared)}}]->(User2)
    # """

    return "Nodes created"


# def get_connection_projection(
#    localhost: int, name_projection: str = "Users_relations"
# ) -> Neo4jConnection:
#    """


#    This function creates the connection and the projection of the graph in the database.
#    :param localhost: the port of the database
#    :param name_projection: the name of the projection
#    :return: the connection to the database
#    """
#    conn = Neo4jConnection(
#        uri="bolt://localhost:" + str(localhost), user="neo4j", pwd="admin"
#    )
#    try:
#        conn.query(
#            "CALL gds.graph.project('"
#            + name_projection
#            + "','User', {CONNECT_TO: {orientation: 'UNDIRECTED'}}, {relationshipProperties: 'importance'})"
#        )
#    except Exception as error_projection:
#        print("Projection failed:", error_projection)
#    return conn
#
#
# def create_nodes_and_correlations(localhost: int):
#    """
#    This function creates the nodes and the correlations in the graph database.
#    :param localhost: the port of the database
#    """
#    conn = Neo4jConnection(
#        uri="bolt://localhost:" + str(localhost), user="neo4j", pwd="admin"
#    )
#    conn.query(
#        """
#    LOAD CSV WITH HEADERS FROM 'file:///users_nodes_filtered.csv' AS row
#    FIELDTERMINATOR ',' CREATE (:User {user_id: toInteger(row.pid), username: row.username, first_name:row.first_name, last_name:row.last_name,alphabets:row.alphabets_detected, admin_score:row.admin_score,owner_score:row.owner_score, member_score:row.member_score,time_score:row.time_score, crowd_score:row.crowd_score,  total_score:row.total_score})    """
#    )
#    conn.query(
#        """
#        LOAD CSV WITH HEADERS FROM 'file:///users_corr_filtered.csv' AS row
#        FIELDTERMINATOR ','
#        MATCH (User1:User {user_id: toInteger(row.user1_PID)}),
#        (User2:User {user_id: toInteger(row.user2_PID)})
#        CREATE (User1)-[:CONNECT_TO {importance:toFloat(row.shared)}]->(User2)
#    """
#    )
#    return "connection created"
#
#
## Create a function that extract the features from the graph
# def get_features_from_graph(connection: Neo4jConnection):
#    """
#    This function extract the features from the graph and return them in a dictionary.
#    :param conn: the connection to the database
#    :param user_id: the id of the user
#    :return: a dictionary with the features
#    """
#    # get the features from the graph
#    # PageRank as a meassure of centrality
#    pageranks = connection.query(
#        "CALL gds.pageRank.write('Users_relations', {relationshipWeightProperty: 'importance', maxIterations: 20, dampingFactor: 0.85, writeProperty: 'pagerank'}) YIELD nodePropertiesWritten, ranIterations"
#    )
#    # degree centrality as a second meassure of centrality
#    centrality = connection.query(
#        "CALL gds.degree.write('Users_relations', {relationshipWeightProperty: 'importance', writeProperty: 'centrality'}) YIELD nodePropertiesWritten"
#    )
#    # louvain as a community detector
#    louvain = connection.query(
#        "CALL gds.louvain.write('Users_relations', {relationshipWeightProperty: 'importance', writeProperty: 'community'})YIELD communityCount, modularity, modularities"
#    )
#    # label propagation as a second community detector
#    labelpropagation = connection.query(
#        "CALL gds.labelPropagation.write('Users_relations', {relationshipWeightProperty: 'importance', writeProperty: 'labelPropagation'}) YIELD nodePropertiesWritten, ranIterations"
#    )
#    # triangle count
#    triangle = connection.query(
#        "CALL gds.triangleCount.write('Users_relations', {writeProperty: 'triangleCount'}) YIELD nodePropertiesWritten"
#    )
#    # local clustering coefficient as a third commyunity detector
#    local_clustering = connection.query(
#        "CALL gds.localClusteringCoefficient.write('Users_relations', {writeProperty: 'localClusteringCoefficient'}) YIELD nodePropertiesWritten"
#    )
#    features = {
#        "PageRank": pageranks,
#        "Degree Centrality": centrality,
#        "Louvain": louvain,
#        "Label Propagation": labelpropagation,
#        "Triangle Count": triangle,
#        "Local Clustering Coefficient": local_clustering,
#    }
#    return features
#
#
## Create a function that extract the new features and user_id from the graph
# def extract_new_features_from_graph(localhost: int):
#    """
#    This function extract the new features from the graph and return them in a dictionary.
#    :param conn: the connection to the database
#    :return: a dictionary with the features
#    """
#    # Create a driver instance to connect to the neo4j database
#    driver = GraphDatabase.driver(
#        "bolt://localhost:" + str(localhost), auth=("neo4j", "admin")
#    )
#
#    # Define a nodes to extract all the node information from the neo4j graph
#    query = "MATCH (n) RETURN n"
#
#    with driver.session() as session:
#        result = session.run(query)
#        nodes = []
#        for record in result:
#            node = record["n"]
#            node_dict = {
#                "id": node.id,
#                "labels": list(node.labels),
#                "properties": dict(node.items()),
#            }
#            nodes.append(node_dict)
#        return nodes
#
#
# if "__main__" == __name__:
#    localhost = 7687
#    create_nodes_and_correlations(localhost)
#    conn = get_connection_projection(localhost)
#    # get the features from the graph
#    features = get_features_from_graph(conn)
#    # get the new features from the graph
#    new_features = extract_new_features_from_graph(localhost)
#
## WIP ##
# """
# We can use link prediction to predict the links between a subset of interesting users.
# such us, between administrators, or between users we dont know a certain characteristic with other we do.
# We can create a function that use as an input a subset of users and return the link prediction between them.
# We can check documentation here: https://neo4j.com/docs/graph-data-science/current/alpha-algorithms/adamic-adar/
# """
## link prediction withouth subsets ---> WIP
## conn.query("CALL gds.linkPrediction.write('Users_relations', {relationshipWeightProperty: 'importance', writeProperty: 'linkPrediction'}) YIELD nodePropertiesWritten, ranIterations")
#
