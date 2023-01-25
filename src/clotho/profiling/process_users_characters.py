"""
This script processes the users characters
for instance, it detects the scripts of the usernames, first names and last names
"""
import unicodedata


def detect_script_on_characters(string: str) -> list[str]:
    """
    This function detects the scripts of a string
    :param string: String to detect the scripts
    :return: List of scripts detected
    """
    scripts_detected = []
    try:
        for character in string:
            name = unicodedata.name(character)
            # If the first word in the name is on known_scripts, replace name with it
            name = name.split(sep=" ")[0]
            scripts_detected.append(name)
        # Return the non repeated scripts
        return list(set(scripts_detected))
    except (ValueError, TypeError):
        return scripts_detected


def detect_scripts(data: list[dict], script_keys: list[str]) -> list[dict]:
    """
    This function filters the data by the keys specified and detects the scripts of the values of the keys specified
    :param data: List of dictionaries to filter and detect scripts on
    :param script_keys: List of keys to detect scripts on
    :return: List of dictionaries with the data and the added "scripts" key
    """
    # Create a new list to store the filtered data
    filtered_data = []
    # Check key by key if their are empty or has None value and detect the scripts on the all the keys with values
    for row in data:
        scripts = {}
        for key in script_keys:
            if row[key] is None or row[key] == "":
                continue
            else:
                scripts[key] = detect_script_on_characters(row[key])
        row["scripts"] = scripts
        # Transform the dictionary from scripts to a single list of scripts
        row["scripts"] = list(
            set([script for key in row["scripts"] for script in row["scripts"][key]])
        )
        filtered_data.append(row)

    return filtered_data


if __name__ == "__main__":
    # Example usage:
    data = [
        {
            "user_PID": "1",
            "username": "بالعالم",
            "first_name": "باعالم",
            "last_name": "last_name2",
        },
        {
            "user_PID": "2",
            "username": "你好",
            "first_name": "first_name2",
            "last_name": "last_name2",
        },
        {
            "user_PID": "3",
            "username": "username3",
            "first_name": "first_name3",
            "last_name": "",
        },
        {
            "user_PID": "4",
            "username": None,
            "first_name": None,
            "last_name": None,
        },
    ]
    filter_keys = ["username", "first_name", "last_name"]
    script_keys = ["username", "first_name", "last_name"]
    users_data_with_scripts = detect_scripts(data, script_keys)
