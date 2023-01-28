import json
import logging
import datetime
from clotho.correlations.correlate_times import round_and_correlate

# from clotho.correlations.correlate_dicts import

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def convert_str_to_datetime(data: list[dict], key: str) -> list[dict]:
    """
    Convert a string to a datetime object
    :param data: List of dictionaries with the data to convert
    :param key: Key for the string to convert
    :return: List of dictionaries with the converted data
    """

    # format example  "2022-10-24 11:35:01"
    for item in data:
        try:
            item[key] = datetime.datetime.strptime(item[key], "%Y-%m-%d %H:%M:%S")
        except TypeError:
            logger.error("Error converting %s", item[key])
    return data


def filter_dict_list(data: list[dict], keys: list[str]) -> list[dict]:
    """
    Filter list of keys from the dicts in the list of dicts
    :param data: List of dictionaries with the data to filter
    :param keys: List of keys to filter
    :return: List of dictionaries with the filtered data
    """
    new_data = []
    for item in data:
        new_item = {}
        for key in keys:
            new_item[key] = item[key]
        new_data.append(new_item)
    return new_data


if __name__ == "__main__":

    # Import json file
    __path__ = "../data/users_data_featured.json"
    logger.info("Importing json file: %s", __path__)
    with open(__path__, "r") as f:
        users_featured = json.load(f)
    logger.info("Json file imported")

    # Run both functions for list ["created_at", "updated_at", "last_updated"]

    # SAMPLE
    users_featured = users_featured[:10000]

    # Filter list of keys from the dicts in the list of dicts
    logger.info("Filtering list of keys from the dicts in the list of dicts")
    data_for_correlate_dicts = filter_dict_list(
        users_featured, ["pid", "country", "script"]
    )
    logger.info("List of keys filtered")
    # Correlate dicts
    logger.info("Correlating dicts")
    # dicts_correlate =
    logger.info("Dicts correlated")

    # Correlate features time format

    list_keys = ["created_at", "updated_at", "last_online_at"]
    logger.info("Correlating features time format: %s", list_keys)
    result = {}
    for key in list_keys:
        # Convert string to datetime
        users_featured_time_corr = convert_str_to_datetime(users_featured, key)

        result[f"corr_{key}"] = round_and_correlate(
            users_featured_time_corr, "pid", key
        )
