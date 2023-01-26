def group_by_id_value_count(
    data_for_process: list[dict], group_by_key: str, key_to_value_count: str
) -> list[dict]:
    """
    Groups the data_for_process by the group_by_key and counts the number of times the key_to_value_count appears
    :param data_for_process: List of dictionaries that looks like this:
    [{"entity_id": channel1 , "key_to_value_count": key_to_value_count1},
    {"entity_id": channel1 , "key_to_value_count": key_to_value_count2},
    {"entity_id": channel1 , "key_to_value_count": key_to_value_count1},
    {"entity_id": channel2 , "key_to_value_count": key_to_value_count2},]
    :param group_by_key: The key to group by
    :param key_to_value_count: The key to count the number of times it appears
    :return: List of dictionaries that looks like this:
    [{"entity_id": channel1 , "key_to_value_count":{key_to_value_count1: count, key_to_value_count2: count, ...}},
    {"entity_id": channel2 , "key_to_value_count":{key_to_value_count2: count, key_to_value_count2: count, ...}},]
    """
    # Create a dictionary with the entity_id as key and the value is a dictionary with the languages as keys and the count as values
    entity_langs = {}
    for data in data_for_process:
        # If the entity_id is not in the dictionary, add it
        if data[group_by_key] not in entity_langs:
            entity_langs[data[group_by_key]] = {}
        # If the language is not in the dictionary, add it
        if data[key_to_value_count] not in entity_langs[data[group_by_key]]:
            entity_langs[data[group_by_key]][data[key_to_value_count]] = 0
        # Increment the count for the language
        entity_langs[data[group_by_key]][data[key_to_value_count]] += 1
    # Convert the dictionary to a list of dictionaries
    entity_langs = [
        {group_by_key: x, key_to_value_count: entity_langs[x]} for x in entity_langs
    ]
    return entity_langs


if __name__ == "__main__":
    # Test the function
    messages = [
        {"entity_id": "channel1", "language": "en"},
        {"entity_id": "channel1", "language": "fr"},
        {"entity_id": "channel1", "language": "es"},
        {"entity_id": "channel1", "language": "en"},
        {"entity_id": "channel2", "language": "es"},
    ]
    import pprint

    pprint.pprint(group_by_id_value_count(messages, "entity_id", "language"))
