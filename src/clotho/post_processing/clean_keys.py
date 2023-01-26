"""
This module will remove the keys from the list provided
"""


def clean_keys(
    data: list[dict], key_to_clean: str, remove_list: list[str]
) -> list[dict]:
    """
    This function will remove the provided data on the remove_list from the key_to_clean
    :param data: list of dict
    :param key_to_clean: key to clean
    :param remove_list: list of data to remove
    :return: list of dict with the cleaned key
    """
    for user in data:
        for item in remove_list:
            if item in user[key_to_clean]:
                user[key_to_clean].remove(item)
    return data


if __name__ == "__main__":
    dummy_data = [
        {"name": "John", "age": 20, "hobbies": ["football", "basketball", "tennis"]},
        {"name": "Mary", "age": 25, "hobbies": ["football", "tennis"]},
        {"name": "Peter", "age": 30, "hobbies": ["football", "basketball"]},
        {"name": "Jane", "age": 35, "hobbies": ["football", "tennis"]},
    ]

    from pprint import pprint

    pprint(clean_keys(dummy_data, "hobbies", ["football", "tennis"]))
