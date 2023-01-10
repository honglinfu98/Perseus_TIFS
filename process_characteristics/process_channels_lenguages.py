"""
This script is used to process the channels to extract the characteristics for later use
on profiling the channels
"""
import pandas as pd
from langdetect import detect
from extraction.loader import load_cloudburst_signals

def call_cloudburst_signals()->pd.DataFrame:
    """
    Call the cloudburst signals
    output: pd.DataFrame
    """
    cloudbusrt_signals = load_cloudburst_signals()
    return cloudbusrt_signals

def detect_language(tuple_list:list[tuple]):
    """
    This function detects the language of the content of the channels
    :param tuple_list: List of tuples
    :return: List of tuples with the new column language
    """
    list_of_tuples = []
    for tup in tuple_list:
        lang = detect(str(tup[7]))
        print(lang)
        #add the new column to the tuple
        tup = tup + (lang,)
        #add tup to a new list of tuples
        list_of_tuples.append(tup)
    return list_of_tuples

def group_by_language(tuple_list:list[tuple]):
    """
    This function groups the channels by the language
    :param tuple_list: List of tuples
    :return: Dataframe grouped by the language
    """
    df = pd.DataFrame(tuple_list, columns=['id', 'channel_id', 'type', 'timestamp', 'commodity',"name" ,'nan', 'content', 'language'])
    grouped_df = df.groupby(['channel_id', 'name'])['language'].apply(list).reset_index()
    #Leave only unique values on the column language
    grouped_df['language'] = grouped_df['language'].apply(lambda x: list(set(x)))
    #Remove rows with empty list on the column language
    grouped_df = grouped_df[grouped_df['language'].map(len) > 0]
    tuple_list = [tuple(x) for x in grouped_df.values]
    return tuple_list


if __name__ == '__main__':
    cloudbusrt_signals_df = call_cloudburst_signals()
    #Convert a dataframe in a list of tuples
    tuple_list = [tuple(x) for x in cloudbusrt_signals_df.values]
    tuple_list = detect_language(tuple_list)
    langs_detected = group_by_language(tuple_list)
    #df to list of tuples
