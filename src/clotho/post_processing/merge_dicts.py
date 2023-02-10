"""
This module contains the function merge_dicts which merges two dictionaries
"""

# Function for merging two dictionaries
# The imput will be a list of list of dictionaries
# The merge will be done by the key provided
# The output will be a list of dictionaries merged by the key provided
# It will be possible to choose if keeping all the keys or only the ones not repeated
# The first dictionary will be the one that will be kept

import pandas as pd


def merge_dicts(
    list_of_dicts_1: list[dict], list_of_dicts_2: list[dict], join_attr: str
) -> list[dict]:
    """
    This function merges two dictionaries
    :param list_of_dicts_1: list of dictionaries
    :param list_of_dicts_2: list of dictionaries
    :param join_attr: join_attr to merge the dictionaries
    :return: list of dictionaries.
    The dictionaries will be merged by the key provided on left
    """
    df1 = pd.DataFrame(list_of_dicts_1)
    df2 = pd.DataFrame(list_of_dicts_2)
    # Merge the 2 dataframes
    list_df1_df2 = pd.merge(df1, df2, on=join_attr, how="left")
    # Convert to dictionary
    merged_dict = list_df1_df2.to_dict("records")
    return merged_dict
