def group_by_id_value_count(
    messages: list[dict], group_by_key: str, key_to_value_count: str
) -> list[dict]:
    """
    Groups the messages by entity_id and language
    in the output the key language will be a dictionary with the languages as keys and the count as values
    for each lenguage detected
    :param messages: List of dictionaries with the messages and the language detected
    :return: List of dictionaries that looks like this:
    [{"entity_id": channel1 , "key_to_value_count":{key_to_value_count1: count, key_to_value_count2: count, ...}},
    {"entity_id": channel2 , "key_to_value_count":{key_to_value_count2: count, key_to_value_count2: count, ...}},]
    """
    # Create a dictionary with the entity_id as key and the value is a dictionary with the languages as keys and the count as values
    entity_langs = {}
    for message in messages:
        # If the entity_id is not in the dictionary, add it
        if message[group_by_key] not in entity_langs:
            entity_langs[message[group_by_key]] = {}
        # If the language is not in the dictionary, add it
        if message[key_to_value_count] not in entity_langs[message[group_by_key]]:
            entity_langs[message[group_by_key]][message[key_to_value_count]] = 0
        # Increment the count for the language
        entity_langs[message[group_by_key]][message[key_to_value_count]] += 1
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
