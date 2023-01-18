"""
This script processes the users characters
for instance, it detects the scripts of the usernames, first names and last names
"""

import pandas as pd
import unicodedata
from extraction.loader import load_cloudburst_user_members_data

def filter_usernames(users_members: pd.DataFrame) -> pd.DataFrame:
    """
    This function filters the usernames, first names and last names of the users
    :param users_members: Merged dataframe
    :return: Dataframe with the usernames, first names and last names
    """
    users_members_characters = users_members[[
        "user_PID", "username", "first_name", "last_name"]]
    return users_members_characters

# Scripts in unicode: "Arabic", "Cyrillic", "Greek", "Hebrew", "Japanese", "Korean", "Latin", "Thai"

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


def detect_scripts(users_members_characters: pd.DataFrame) -> pd.DataFrame:
    """
    This function detects the scripts of the usernames, first names and last names
    and append to a new column
    :param users_members_characters: Dataframe with the usernames, first names and last names
    :return: Dataframe with the usernames, first names, last names and the scripts detected
    """
    users_members_characters["username_scripts"] = users_members_characters["username"].apply(
        detect_script_on_characters)
    users_members_characters["first_name_scripts"] = users_members_characters["first_name"].apply(
        detect_script_on_characters)
    users_members_characters["last_name_scripts"] = users_members_characters["last_name"].apply(
        detect_script_on_characters)
    # Merge the scripts detected in one column
    users_members_characters["scripts"] = users_members_characters["username_scripts"] + \
        users_members_characters["first_name_scripts"] + \
        users_members_characters["last_name_scripts"]
    # Remove the repeated scripts
    users_members_characters["scripts"] = users_members_characters["scripts"].apply(
        lambda x: list(set(x)))
    # Delete the first three columns created
    users_members_characters = users_members_characters.drop(
        columns=["username_scripts", "first_name_scripts", "last_name_scripts"])
    return users_members_characters


def run_detect_scripts(members_users: pd.DataFrame) -> pd.DataFrame:
    """
    This function runs the detect_scripts function
    """
    users_members_characters = filter_usernames(members_users)
    users_members_detected_scripts = detect_scripts(users_members_characters)
    return users_members_detected_scripts


if __name__ == '__main__':
    cloudbusrt_user_members_data = load_cloudburst_user_members_data()
    users_members_characters_scripts = run_detect_scripts(
        cloudbusrt_user_members_data)
    print(users_members_characters_scripts)
