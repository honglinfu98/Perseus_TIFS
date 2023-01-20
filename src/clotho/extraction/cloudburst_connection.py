"""
Script for connecting with cloudburst database
"""
import psycopg2
from sshtunnel import SSHTunnelForwarder

from clotho.config import (
    CLOUDBURST_HOST,
    CLOUDBURST_USERNAME,
    CLOUDBURST_PASS,
    SSH_PKEY,
    SSH_PRIVATE_KEY_PASSWORD,
)


class CloudburstDataBaseConnection:
    """
    Class for extracting tables from cloudburst through a SQL query
    """

    def __init__(self, sql_query: str, table_name: str) -> None:
        self.sql_query = sql_query
        self.table_name = table_name

    @classmethod
    def fetch_data(
        cls,
        sql_query: str,
        db_username: str = CLOUDBURST_USERNAME,
        db_password: str = CLOUDBURST_PASS,
        db_host: str = CLOUDBURST_HOST,
        db_port=25060,
        remote_host="161.35.13.185",
        remote_ssh_port=22,
        remote_username="root",
    ) -> list[tuple]:
        """
        Establish connection with cloudburst database
        """
        with SSHTunnelForwarder(
            (remote_host, remote_ssh_port),
            ssh_username=remote_username,
            remote_bind_address=(db_host, db_port),
            ssh_pkey=SSH_PKEY,
            ssh_private_key_password=SSH_PRIVATE_KEY_PASSWORD,
        ) as ssh_tunnel:
            try:
                conn = psycopg2.connect(
                    host="localhost",
                    port=ssh_tunnel.local_bind_port,
                    user=db_username,
                    password=db_password,
                    database="cloudburst",
                )

                cursor = conn.cursor()
                cursor.execute(sql_query)
                return cursor.fetchall()

            finally:
                cursor.close()
                conn.close()
