"""
Save dict to json file.
"""
import json
from datetime import datetime
import pandas as pd


def save_dict_to_json(dict_to_save: list[dict], file_name: str) -> None:
    """
    Save dict to json file.
    :param dict_to_save: dict to save
    :param file_name: name of the file
    """
    with open(file_name, "w") as outfile:
        json.dump(dict_to_save, outfile)


def detect_datetime_format(list_of_dicts: list[dict]) -> list[str]:
    """ "
    Chek all the keys on the first dict from the list and return name
    of the key with the datetime format.
    :param list_of_dicts: list of dicts
    :return: name of the keys with the datetime format
    """
    list_of_keys = []
    for key in list_of_dicts[0].keys():
        try:
            pd.to_datetime(list_of_dicts[0][key])
            list_of_keys.append(key)
        except ValueError:
            continue
    return list_of_keys


def convert_datetime_to_string(data_list: list[dict], keys: list) -> list[dict]:
    """
    Convert datetime to string.
    :param data_list: list of dicts
    :param keys: list of keys with datetime format
    :return: list of dicts
    """
    for item in data_list:
        for key in keys:
            try:
                if key in item:
                    item[key] = item[key].strftime("%Y-%m-%d %H:%M:%S")
            except (AttributeError):
                continue
    return data_list


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
    # convert datetime to string
    dummy_dicts_list_for_json = convert_datetime_to_string(
        dummy_dicts_list, detect_datetime_format(dummy_dicts_list)
    )
    # save to json
    save_dict_to_json(dummy_dicts_list_for_json, "test.json")
