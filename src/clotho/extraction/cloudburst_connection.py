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


def cloudburst_db_connection(
    sql_query: str,
    db_username: str | None = CLOUDBURST_USERNAME,
    db_password: str | None = CLOUDBURST_PASS,
    db_host: str | None = CLOUDBURST_HOST,
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
                port=ssh_tunnel.local_bind_port,  # type: ignore #TODO: fix this
                user=db_username,
                password=db_password,
                database="cloudburst",
            )

            cursor = conn.cursor()
            cursor.execute(sql_query)
            return cursor.fetchall()

        finally:
            cursor.close()  # type: ignore #TODO: fix this
            conn.close()  # type: ignore #TODO: fix this
