"""Run the scripts to detect the users' characteristics."""
import logging
import pandas as pd
from extraction.loader import load_cloudburst_user_members_data
from profiling.process_users_characters import run_detect_scripts
from profiling.process_users_phone_numbers_library import join_phone_number_data
from processing.merge_users_features import merge_dataframes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

if __name__ == "__main__":
    logger.info("Loading users-members")
    user_members  = load_cloudburst_user_members_data()
    logger.info("Users-members loaded")

    logger.info("Detecting users' scripts on username, first_name, last_name")
    scripts_by_character = run_detect_scripts(user_members)
    logger.info("Users' characteristics detected")

    logger.info("Processing phone number data")
    # Label the region of the phone number
    user_members_phones = join_phone_number_data(user_members)
    #Filter the columns we want to keep: user_PID, phone_number, region, country
    user_members_phones_df = pd.DataFrame(user_members_phones, columns=["user_PID", "phone_number", "region", "country"])
    logger.info("Phone number data processed")

    #Create a list with the dataframes to merge
    dataframes_to_merge = [scripts_by_character, user_members_phones_df]

    #Merge the dataframes and return a tuple list and a dataframe
    #on the dataframe we have the columns names
    (users_members_featured_tuple_list,
    user_members_featured_df) = merge_dataframes(dataframes_to_merge)
