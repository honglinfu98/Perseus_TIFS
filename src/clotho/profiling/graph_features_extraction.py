"""
In this module, we extract the features from the Neo4j graph and add it to the user information.
This module only work as a local conection to the Neo4j graph.
Neo4j must be open and the graph must be created.
"""
from neo4j import GraphDatabase
from py2neo import Node


class Neo4jConnection:
    """
    This class is used to create a connection to the graph database.
    """

    def __init__(self, uri: str, user: str, pwd: str) -> None:
        """
        This function initializes the connection to the graph database.
        :param uri: the uri of the database
        :param user: the user of the database
        :param pwd: the password of the database
        """

        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(
                self.__uri, auth=(self.__user, self.__pwd)
            )
        except Exception as error_driver:
            print("Failed to create the driver:", error_driver)

    def close(self: object) -> None:
        """
        This function closes the connection to the graph database.
        """
        if self.__driver is not None:  # type: ignore
            self.__driver.close()  # type: ignore

    def query(self: object, query: str, db=None) -> list:  # type: ignore
        """
        This function executes a query on the graph database.
        :param query: the query to execute
        :param db: the database to use
        :return: the result of the query
        """
        assert self.__driver is not None, "Driver not initialized!"  # type: ignore
        session = None
        response = None
        try:
            session = (
                self.__driver.session(database=db)  # type: ignore
                if db is not None
                else self.__driver.session()  # type: ignore
            )
            response = list(session.run(query))
        except Exception as error_query:
            print("Query failed:", error_query)
        finally:
            if session is not None:
                session.close()
        return response  # type: ignore


def get_connection_projection(
    localhost: int, name_projection: str = "Users_relations"
) -> Neo4jConnection:
    """
    This function creates the connection and the projection of the graph in the database.
    :param localhost: the port of the database
    :param name_projection: the name of the projection
    :return: the connection to the database
    """
    conn = Neo4jConnection(
        uri="bolt://localhost:" + str(localhost), user="neo4j", pwd="admin"
    )
    try:
        conn.query(
            "CALL gds.graph.project('"
            + name_projection
            + "','User', {CONNECT_TO: {orientation: 'UNDIRECTED'}}, {relationshipProperties: 'importance'})"
        )
    except Exception as error_projection:
        print("Projection failed:", error_projection)
    return conn


def create_nodes_and_correlations(localhost: int):
    """
    This function creates the nodes and the correlations in the graph database.
    :param localhost: the port of the database
    """
    conn = Neo4jConnection(
        uri="bolt://localhost:" + str(localhost), user="neo4j", pwd="admin"
    )
    conn.query(
        """
    LOAD CSV WITH HEADERS FROM 'file:///users_nodes_filtered.csv' AS row 
    FIELDTERMINATOR ',' CREATE (:User {user_id: toInteger(row.pid), username: row.username, first_name:row.first_name, last_name:row.last_name,alphabets:row.alphabets_detected, admin_score:row.admin_score,owner_score:row.owner_score, member_score:row.member_score,time_score:row.time_score, crowd_score:row.crowd_score,  total_score:row.total_score})    """
    )
    conn.query(
        """
        LOAD CSV WITH HEADERS FROM 'file:///users_corr_filtered.csv' AS row
        FIELDTERMINATOR ','
        MATCH (User1:User {user_id: toInteger(row.user1_PID)}),
        (User2:User {user_id: toInteger(row.user2_PID)})
        CREATE (User1)-[:CONNECT_TO {importance:toFloat(row.shared)}]->(User2)
    """
    )
    return "connection created"


# Create a function that extract the features from the graph
def get_features_from_graph(localhost: int):
    """
    This function extract the features from the graph and return them in a dictionary.
    :param conn: the connection to the database
    :param user_id: the id of the user
    :return: a dictionary with the features
    """
    # get the features from the graph
    # PageRank as a meassure of centrality
    pageranks = conn.query(
        "CALL gds.pageRank.write('Users_relations', {relationshipWeightProperty: 'importance', maxIterations: 20, dampingFactor: 0.85, writeProperty: 'pagerank'}) YIELD nodePropertiesWritten, ranIterations"
    )
    # degree centrality as a second meassure of centrality
    centrality = conn.query(
        "CALL gds.degree.write('Users_relations', {relationshipWeightProperty: 'importance', writeProperty: 'centrality'}) YIELD nodePropertiesWritten"
    )
    # louvain as a community detector
    louvain = conn.query(
        "CALL gds.louvain.write('Users_relations', {relationshipWeightProperty: 'importance', writeProperty: 'community'})YIELD communityCount, modularity, modularities"
    )
    # label propagation as a second community detector
    labelpropagation = conn.query(
        "CALL gds.labelPropagation.write('Users_relations', {relationshipWeightProperty: 'importance', writeProperty: 'labelPropagation'}) YIELD nodePropertiesWritten, ranIterations"
    )
    # triangle count
    triangle = conn.query(
        "CALL gds.triangleCount.write('Users_relations', {writeProperty: 'triangleCount'}) YIELD nodePropertiesWritten"
    )
    # local clustering coefficient as a third commyunity detector
    local_clustering = conn.query(
        "CALL gds.localClusteringCoefficient.write('Users_relations', {writeProperty: 'localClusteringCoefficient'}) YIELD nodePropertiesWritten"
    )
    features = {
        "PageRank": pageranks,
        "Degree Centrality": centrality,
        "Louvain": louvain,
        "Label Propagation": labelpropagation,
        "Triangle Count": triangle,
        "Local Clustering Coefficient": local_clustering,
    }
    return features


# Create a function that extract the new features and user_id from the graph
def extract_new_features_from_graph(localhost: int):
    """
    This function extract the new features from the graph and return them in a dictionary.
    :param conn: the connection to the database
    :return: a dictionary with the features
    """
    # Create a driver instance to connect to the neo4j database
    driver = GraphDatabase.driver(
        "bolt://localhost:" + str(localhost), auth=("neo4j", "admin")
    )

    # Define a nodes to extract all the node information from the neo4j graph
    query = "MATCH (n) RETURN n"

    with driver.session() as session:
        result = session.run(query)
        nodes = []
        for record in result:
            node = record["n"]
            node_dict = {
                "id": node.id,
                "labels": list(node.labels),
                "properties": dict(node.items()),
            }
            nodes.append(node_dict)
        return nodes


if "__main__" == __name__:
    localhost = 7687
    create_nodes_and_correlations(localhost)
    conn = get_connection_projection(localhost)
    # get the features from the graph
    features = get_features_from_graph(localhost)
    # get the new features from the graph
    new_features = extract_new_features_from_graph(localhost)

# WIP ##
"""
We can use link prediction to predict the links between a subset of interesting users.
such us, between administrators, or between users we dont know a certain characteristic with other we do.
We can create a function that use as an input a subset of users and return the link prediction between them.
We can check documentation here: https://neo4j.com/docs/graph-data-science/current/alpha-algorithms/adamic-adar/
"""
# link prediction withouth subsets ---> WIP
# conn.query("CALL gds.linkPrediction.write('Users_relations', {relationshipWeightProperty: 'importance', writeProperty: 'linkPrediction'}) YIELD nodePropertiesWritten, ranIterations")
