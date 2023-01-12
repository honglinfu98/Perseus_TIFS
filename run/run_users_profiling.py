"""Run the scripts to detect the users' characteristics."""
import logging
import pandas as pd
from extraction_scripts.loader import load_cloudburst_user_members_data
from profiling_scripts.process_users_characters import run_detect_scripts
from profiling_scripts.process_users_phone_numbers import label_countries, create_dict
from processing_scripts.merge_users_features import merge_dataframes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

if __name__ == "__main__":
    logger.info("Loading users-members")
    user_members  = load_cloudburst_user_members_data()
    logger.info("Users-members loaded")

    logger.info("Detecting users' scripts on username, first_name, last_name")
    scripts_by_character = run_detect_scripts(user_members)
    logger.info("Users' characteristics detected")

    logger.info("Loading phone codes")
    phone_code_df = pd.read_csv("../support_data/cellphones_code.csv", sep=";")
    logger.info("Phone codes loaded")

    logger.info("Creating dictionary with phone codes")
    phone_code_dict = create_dict(phone_code_df, "country", "country_code")
    logger.info("Dictionary with phone codes created")

    logger.info("Detecting location by users' phone numbers")
    country_by_phone = label_countries(user_members,phone_code_dict)
    logger.info("Location by phone numbers detected")
    
    #Create a list with the dataframes to merge
    dataframes_to_merge = [scripts_by_character, country_by_phone]
    
    #Merge the dataframes and return a tuple list and a dataframe
    #on the dataframe we have the columns names
    (users_members_featured_tuple_list,
    user_members_featured_df) = merge_dataframes(dataframes_to_merge)