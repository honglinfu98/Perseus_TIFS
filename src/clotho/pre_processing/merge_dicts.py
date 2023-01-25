"""
This module contains the function merge_dicts which merges two dictionaries
"""

# Function for merging two dictionaries
# The imput will be a list of list of dictionaries
# The merge will be done by the key provided
# The output will be a list of dictionaries merged by the key provided
# It will be possible to choose if keeping all the keys or only the ones not repeated
# The first dictionary will be the one that will be kept


def merge_dicts(
    list_of_dicts: list[dict], key: str, keep_all_keys: bool = True
) -> list[dict]:
    """
    This function merges two dictionaries
    LIMITATION:
    -The dictionaries must have ONE key repeated
    -Works only with two dictionaries at the same time

    TODO improve the function to work with more than two dictionaries at the same time

    :param list_of_dicts: List of dictionaries
    :param key: Key to merge the dictionaries
    :param keep_all_keys: Boolean to choose if keeping all the keys or only the ones not repeated
    :return: List of dictionaries merged by the key provided
    """
    merged_dicts = {}
    for current_dict in list_of_dicts:
        for current_key, current_value in current_dict.items():
            if current_key == key:
                if current_value in merged_dicts:
                    if keep_all_keys:
                        # Append "_a" to key for duplicate keys
                        for key in current_dict.keys():
                            if key in merged_dicts[current_value]:
                                current_dict[key + "_a"] = current_dict.pop(key)
                        merged_dicts[current_value].update(current_dict)
                    else:
                        merged_dicts[current_value].update(current_dict)
                else:
                    merged_dicts[current_value] = current_dict
    return list(merged_dicts.values())


if __name__ == "__main__":
    # Create the first list of dictionaries
    list_of_dicts = [
        {
            "user_PID": 1,
            "username": "user1",
            "first_name": "first_name1",
            "last_name": "last_name1",
        },
        {
            "user_PID": 2,
            "username": "user2",
            "first_name": "first_name2",
            "last_name": "last_name2",
        },
        {
            "user_PID": 3,
            "username": "user3",
            "first_name": "first_name3",
            "last_name": "last_name3",
        },
        {
            "user_PID": 4,
            "username": "user4",
            "first_name": "first_name4",
            "last_name": "last_name4",
        },
        {
            "user_PID": 5,
            "username": "user5",
            "first_name": "first_name5",
            "last_name": "last_name5",
        },
        {
            "user_PID": 6,
            "username": "user6",
            "first_name": "first_name6",
            "last_name": "last_name6",
        },
    ]

    # Create the second list of dictionaries and use different keys except the key "user_PID"
    # Don't reapeat any key used in the first list of dictionaries
    list_of_dicts_2 = [
        {
            "user_PID": 1,
            "message": "message1",
            "date": "date1",
            "time": "time1",
            "username": "adsfas",
        },
        {
            "user_PID": 2,
            "message": "message2",
            "date": "date2",
            "time": "time2",
            "username": "ussdafer2",
        },
        {
            "user_PID": 3,
            "message": "message3",
            "date": "date3",
            "time": "time3",
            "username": "asdf",
        },
        {
            "user_PID": 4,
            "message": "message4",
            "date": "date4",
            "time": "time4",
            "username": "asdfsadf",
        },
        {
            "user_PID": 5,
            "message": "message5",
            "date": "date5",
            "time": "time5",
            "username": "usdsafer5",
        },
        {
            "user_PID": 6,
            "message": "message6",
            "date": "date6",
            "time": "time6",
            "username": "dsafdsafsadf",
        },
    ]

    # Merge the two lists of dictionaries
    merged_dicts = merge_dicts(
        list_of_dicts_2 + list_of_dicts,
        key="user_PID",
        keep_all_keys=True,
    )
