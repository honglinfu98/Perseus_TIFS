"""
This module will remove the keys from the list provided
"""


def clean_keys(
    data: list[dict], key_to_clean: str, remove_list: list[str]
) -> list[dict]:
    """
    This function will remove the provided the keys on the dict on the remove_list from the key_to_clean
    There is a dict inside the searched key in the list of dicts
    :param data: list of dict
    :param key_to_clean: key to clean
    :param remove_list: list of data to remove
    :return: list of dict with the cleaned key
    """
    for dict_ in data:
        for key in remove_list:
            if key in dict_[key_to_clean]:
                del dict_[key_to_clean][key]

    return data


if __name__ == "__main__":
    dummy_data = [
        {"entity_id": "-1001639961681", "language": {"en": 6}},
        {"entity_id": "-1001608999927", "language": {"Too short": 5, "en": 1}},
        {"entity_id": "-1001395566862", "language": {"en": 8, "ca": 3}},
        {"entity_id": "-1001652601224", "language": {"ca": 1}},
        {"entity_id": "-1001619333749", "language": {"en": 5}},
    ]

    from pprint import pprint

    pprint(clean_keys(dummy_data, "language", ["Too short", "en"]))
