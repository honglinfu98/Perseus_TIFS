"""
Cloudburst test database connection
"""
import psycopg2
from clotho.config import CLOUDBURST_USERNAME, CLOUDBURST_PASS, CLOUDBURST_HOST


def cloudburst_connection(
    sql_query: str,
    db_username=CLOUDBURST_USERNAME,
    db_password=CLOUDBURST_PASS,
    db_host=CLOUDBURST_HOST,
    db_port=25060,
):
    """
    Establish connection with cloudburst database
    :param sql_query: SQL query to execute
    :param db_username: Username for cloudburst database
    :param db_password: Password for cloudburst database
    :param db_host: Host for cloudburst database
    :param db_port: Port for cloudburst database
    :return: List of tuples containing the results of the query
    """
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        user=db_username,
        password=db_password,
        database="cloudburst",
    )
    cur = conn.cursor()
    cur.execute(sql_query)
    data = cur.fetchall()
    cur.close()
    conn.close()
    return data


if __name__ == "__main__":
    # Implementation example
    data = cloudburst_connection("select * from cloudburst_signals limit 1")
