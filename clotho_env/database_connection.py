"""
Script for connecting with cloudburst database
"""

import os
import psycopg2
from sshtunnel import SSHTunnelForwarder


class CloudburstDataBaseConnection:
    """
    Class for extracting tables from cloudburst through a SQL query
    """

    def __init__(self, sql_query: str, table_name: str):
        self.sql_query = sql_query
        self.table_name = table_name

    @classmethod
    def fetch_data(
        cls,
        sql_query,
        ssh_private_key_password: str = "",
        ssh_pkey: str = os.environ.get("SSH_PKEY").rstrip("\r"),
        db_username: str = os.environ.get("DB_USERNAME").rstrip("\r"),
        db_password: str = os.environ.get("DB_PASS").rstrip("\r"),
        db_host="cloudburst-db-do-user-11945868-0.b.db.ondigitalocean.com",
        db_port=25060,
        remote_host="161.35.13.185",
        remote_ssh_port=22,
        remote_username="root",
    ):
        """
        Establish connection with cloudburst database
        """
        with SSHTunnelForwarder(
            (remote_host, remote_ssh_port),
            ssh_username=remote_username,
            ssh_pkey=ssh_pkey,
            ssh_private_key_password=ssh_private_key_password,
            remote_bind_address=(db_host, db_port),
        ) as ssh_tunnel:
            print("SSH tunnel connected")

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
