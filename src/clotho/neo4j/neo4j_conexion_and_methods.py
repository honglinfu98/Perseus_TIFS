"""
This module contains the class Neo4jConnection and the methods to create a connection to the graph database and to add nodes and edges to the graph.

"""
import logging

from neo4j import GraphDatabase

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()


class Neo4jConnection:
    """
    This class is used to create a connection to the graph database.
    """

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        # Don't forget to close the driver connection when you are finished with it
        self.driver.close()

    def query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return result.data()

    def add_nodes(self, dictionary_nodes: list, batch_size: int = 1000):
        for batch_start in range(0, len(dictionary_nodes), batch_size):
            batch = dictionary_nodes[batch_start : batch_start + batch_size]
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
                f"Nodes {batch_start + 1}-{min(batch_start + batch_size, len(dictionary_nodes))}/{len(dictionary_nodes)} added to the database"
            )

    def add_edges(self, dictionary_edges: list, batch_size: int = 1000):
        for batch_start in range(0, len(dictionary_edges), batch_size):
            batch = dictionary_edges[batch_start : batch_start + batch_size]
            query = """
            UNWIND $edges_batch AS edge
            MATCH (a:User {pid: edge.user1_PID}), (b:User {pid: edge.user2_PID})
            MERGE (a)-[r:SHARED {shared: edge.shared}]-(b)
            """
            self.query(query, parameters={"edges_batch": batch})
            logger.info(
                f"Edges {batch_start + 1}-{min(batch_start + batch_size, len(dictionary_edges))}/{len(dictionary_edges)} added to the database"
            )

    def create_graph_projection(
        self,
        name_projection: str,
        relationship_type: str = "SHARED",
        relationship_property: str = "shared",
    ):
        query = f"""
        CALL gds.graph.create(
            '{name_projection}',
            'User',
            '{relationship_type}',
            {{
                relationshipProperties: '{relationship_property}',
                orientation: 'UNDIRECTED'
            }}
        )
        """
        self.query(query)
        logger.info(f"Graph projection '{name_projection}' created.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
