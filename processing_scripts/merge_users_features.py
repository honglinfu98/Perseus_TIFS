"""
Function to merge users and features dataframes by the user_PID
"""
import pandas as pd

# Creation of a function that has as input a group of dataframes and returns a single dataframe
# Merging all the dataframes by the user_PID, mantaing all the rows from all dataframes
def merge_dataframes(dataframes: list) -> list[tuple]:
    """
    This function merges all the dataframes by the user_PID, mantaing all the rows from all dataframes
    :param dataframes: List of dataframes
    :return: Dataframe
    """
    # Create a new dataframe
    merged_df = pd.DataFrame()
    # Iterate over the dataframes
    for df in dataframes:
        # If the dataframe is empty, skip it
        if df.empty:
            continue
        # If the merged dataframe is empty, assign the first dataframe to it
        if merged_df.empty:
            merged_df = df
        # If the merged dataframe is not empty, merge the current dataframe with the merged dataframe
        else:
            merged_df = pd.merge(merged_df, df, on='user_PID', how='outer')

    #Convert merged_df to a tuples list
    merged_tuple_list = [tuple(x) for x in merged_df.to_numpy()]

    return merged_tuple_list,merged_df
