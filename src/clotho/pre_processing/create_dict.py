"""
convert raw data fetched from tuples list to dictinary
"""


def tuple_to_dict(data: list[tuple], keys: list[str]) -> list[dict]:
    """
    This function converts the raw data fetched from tuples list to dictinary
    it checks if the length of the keys is equal to the length of the tuple
    :param data: List of tuples
    :param keys: List of keys
    :return: List of dictionaries
    """
    data_dict = []
    for row in data:
        if len(row) == len(keys):
            data_dict.append(dict(zip(keys, row)))
        else:
            raise ValueError(
                "The length of the keys is not equal to the length of the tuple"
            )
    return data_dict


if __name__ == "__main__":
    # Dummy data to test the function
    data = [
        (1, 2, 3),
        (11, 12, 13),
        (21, 22, 23),
    ]
    keys = ["entity_id", "username", "message_text"]

    # Test the function
    data_dict = tuple_to_dict(data, keys)
