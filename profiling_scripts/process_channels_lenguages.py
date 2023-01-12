"""
This script is used to process the channels to extract the characteristics for later use
on profiling the channels
"""
import logging
import pandas as pd
from langdetect import detect
from extraction_scripts.loader import load_cloudburst_signals

# Call logger
logger = logging.getLogger()

def detect_language(tuple_list: list[tuple],col_names_signals: list[str])->list[tuple]:
    """
    This function detects the language of the content of the channels
    :param tuple_list: List of tuples
    :param column_names: List of column names
    :return: List of tuples with the new column language
    """
    message_position = col_names_signals.index("message_text")
    entity_id_position = col_names_signals.index("entity_id")
    name_position = col_names_signals.index("username")
    # Extrac from the list of tuples the 3 columns that we need
    tuple_list = [(tup[entity_id_position], tup[name_position], tup[message_position]) for tup in tuple_list]
    # Langdetect library has a limit of 50 characters to detect the language
    tuple_list = [tup for tup in tuple_list if len(tup[2]) > 50]
    logger.info("Number of signals to detect language: %s", len(tuple_list))
    list_of_tuples = []
    detection_errors = []
    logger.info("Detecting language...")
    for idx, tup in enumerate(tuple_list):
        try:
            lang = detect(str(tup[2]))
            # logger.info("Detected language: %s - Position: %s", lang, idx)
            tup = tup + (lang,)
            list_of_tuples.append(tup)
        except Exception as lang_detect_e:  # TODO Add specific exceptio, too broad
            logger.error("Error: %s", lang_detect_e)
            detection_errors.append(lang_detect_e)
            continue
    logger.error("Number of errors:  %s", len(detection_errors))
    logger.error("Detection errors:  %s", detection_errors)
    return list_of_tuples

def group_by_language(detected_langs: list[tuple])->list[tuple]:
    """
    This function groups the channels by the language
    :param tuple_list: List of tuples
    :return: Dataframe grouped by the language
    """
    detected_langs_df = pd.DataFrame(detected_langs,
                                     columns=["entity_id", "username", "message_text", "language"])

    grouped_df = detected_langs_df.groupby(['entity_id', 'username'])[
        'language'].apply(list).reset_index()

    # Leave only unique values on the column language
    grouped_df['language'] = grouped_df['language'].apply(
        lambda x: list(set(x)))

    # Remove rows with empty list on the column language
    grouped_df = grouped_df[grouped_df['language'].map(len) > 0]
    detected_langs = [tuple(x) for x in grouped_df.values]
    return detected_langs

def detect_and_group_by_language(signal_df: pd.DataFrame)->list[tuple]:
    """
    This function detects the language of the content of the channels and groups by language
    :param signal_df: Dataframe with the signals
    :return: Dataframe grouped by the language
    """
    signals_tuple_list = [tuple(x) for x in signal_df.values]
    column_names = signal_df.columns.to_list()
    print(column_names)
    signals_tuple_list = detect_language(signals_tuple_list,column_names)
    langs_detected_grouped = group_by_language(signals_tuple_list)
    return langs_detected_grouped

if __name__ == '__main__':
    cloudbusrt_signals_df = load_cloudburst_signals()
    langs_detected = detect_and_group_by_language(cloudbusrt_signals_df)
