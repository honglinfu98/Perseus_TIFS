"""
Save dict to json file.
"""
import json
import os
from datetime import datetime
from decimal import Decimal

import pandas as pd


def handle_non_serializable_objects(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def save_dict_to_json(dict_to_save: list[dict], file_name: str) -> None:
    """
    Save dict to json file.
    :param dict_to_save: dict to save
    :param file_name: name of the file
    """
    with open(file_name, "w") as outfile:
        json.dump(dict_to_save, outfile, default=handle_non_serializable_objects)


def detect_datetime_format(list_of_dicts):
    list_of_keys = []
    for key in list_of_dicts[0].keys():
        if isinstance(
            list_of_dicts[0][key], str
        ):  # Add this line to check if the value is a string
            try:
                pd.to_datetime(list_of_dicts[0][key])
                list_of_keys.append(key)
            except (AttributeError, TypeError):
                pass
    return list_of_keys


def convert_timestamps_to_strings(data_list: list[dict]) -> list[dict]:
    """
    Convert timestamps to strings.
    :param data_list: list of dicts
    :return: list of dicts
    """

    for data_dict in data_list:
        for key, value in data_dict.items():
            if isinstance(value, pd.Timestamp):
                data_dict[key] = value.strftime("%Y-%m-%dT%H:%M:%S")
            elif value is pd.NaT:  # Check for pd.NaT objects
                data_dict[key] = None
    return data_list


def save_dict(data_list: list[dict], file_name: str) -> None:
    """
    Save dict to json file.
    :param data_list: list of dicts
    :param file_name: name of the file
    """
    # Check if the folder exists
    if not os.path.exists("../data"):
        os.makedirs("../data")
    # Add the path to the file name
    file_name = "../data/" + file_name
    data_list_for_json = convert_timestamps_to_strings(data_list)
    save_dict_to_json(data_list_for_json, file_name)


if __name__ == "__main__":
    # dummy data
    dummy_dicts_list = [
        {
            "entity_id": "1",
            "source_datetime": datetime(2021, 1, 1, 1, 1, 1),
            "message_text": "Hello",
        },
        {
            "entity_id": "1",
            "source_datetime": datetime(2021, 1, 1, 1, 1, 1),
            "message_text": "Hello",
        },
        {
            "entity_id": "2",
            "source_datetime": datetime(2021, 1, 1, 1, 1, 1),
            "message_text": "Hello",
        },
    ]

    # save dict to json file
    save_dict(dummy_dicts_list, "dummy_dicts_list.json")
