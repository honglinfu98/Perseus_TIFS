"""
This script will save generic dictionaries to csv files.
"""
import logging
import pandas as pd
from os import path
from clotho.constants import DATA_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def save_dict_to_csv(data: list, keys: list, path: str) -> None:
    """
    This function will save a list of dictionaries to a csv file.
    :param data: The list of dictionaries to save.
    :param path: The path to save the csv file to.
    :param keys: The keys to save to the csv file.
    """
    df = pd.DataFrame(data, columns=keys)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def keys_from_dict(data: list) -> list:
    """
    This function will create a list of keys from a dictionary.
    :param data: The dictionary to get the keys from.
    :return: A list of keys.
    """
    keys = []
    for item in data:
        for key in item.keys():
            if key not in keys:
                keys.append(key)
    return keys


def save_dict(data: list, file_name: str, save_path: str = DATA_PATH) -> None:
    """
    This function will save a list of dictionaries to a csv file.
    :param data: The list of dictionaries to save.
    :param file_name: The name of the file to save to.
    :param path: The path to save the csv file to.
    """
    keys = keys_from_dict(data)
    save_dict_to_csv(data, keys, path.join(save_path, file_name))


if __name__ == "__main__":
    # create a dictionary
    data = [
        {"start": 1, "end": 2, "mixed_weights": 0.5},
        {"start": 1, "end": 3, "mixed_weights": 0.5},
        {"start": 2, "end": 3, "mixed_weights": 0.5},
        {"start": 2, "end": 4, "mixed_weights": 0.5},
        {"start": 3, "end": 4, "mixed_weights": 0.5},
    ]
    # get the keys from the dictionary
    dict_keys = keys_from_dict(data)
    # save the dictionary to a csv file
    save_dict_to_csv(data, dict_keys, path.join(DATA_PATH, "test.csv"))
