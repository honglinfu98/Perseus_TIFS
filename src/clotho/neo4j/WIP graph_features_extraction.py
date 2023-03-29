"""
In this module, we extract the features from the Neo4j graph and add it to the user information.
This module only work as a local conection to the Neo4j graph.
Neo4j must be open and the graph must be created.


IMPORTANT: This module is not used in the project. It is only a WIP.
In order to make it functional we require neo4j paid version.
"""
# import logging
#
#
# from clotho.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME
# from clotho.neo4j.neo4j_conexion_and_methods import Neo4jConnection
#
# logging.basicConfig(
#    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger()
#
#
# with Neo4jConnection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD) as conn:
#    conn.create_graph_projection("my_graph_projection")
