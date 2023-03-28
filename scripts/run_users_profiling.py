"""
This script runs the flowork of the users profiling.
"""
import json
import logging
from datetime import datetime
from decimal import Decimal

from clotho.extraction.loader import (
    COLUMNS_NAMES_USERS,
    USERS_SCORES_COLUMNS,
    load_cloudburst_users,
    load_cloudburst_users_score,
)
from clotho.post_processing.merge_dicts import merge_dicts
from clotho.pre_processing.create_dict import tuple_to_dict
from clotho.profiling.features_extraction import extract_features
from clotho.profiling.extract_phonenumbers_data import extract_data_phonenumber
from clotho.extraction.loader import load_cloudburst_users_score, USERS_SCORES_COLUMNS
from clotho.post_processing.merge_dicts import merge_dicts
from clotho.post_processing.save_dict_to_json import save_dict


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

if __name__ == "__main__":
    # Download the users data for users profiling ############################
    logger.info("Loading users")
    users_data = load_cloudburst_users()
    logger.info("Users loaded")
    logger.info("Number of users extracted: %s", len(users_data))

    # Convert to dictionary ##################################################
    logger.info("Converting to dictionary...")
    users_data_dict = tuple_to_dict(users_data, keys=COLUMNS_NAMES_USERS)
    logger.info("Users converted to dictionary")
    ###########################################################################

    # Extract features from: username, first_name, last_name, bio #############
    logger.info("Extracting features from: username, first_name, last_name, bio")
    script_keys_to_process = ["username", "first_name", "last_name", "bio"]
    # We will extract the features from the keys in the script_keys_to_process list
    # Features extracted: urls, accounts, hashtags, domains, script, phone_numbers, emails_adresses
    extract_features(users_data_dict, script_keys_to_process)
    logger.info("Features extracted")
    ###########################################################################

    # Extract features from: phone_numbers ####################################
    logger.info("Extracting features from: phone_number and phone_number_extracted")
    users_data_dict_country = extract_data_phonenumber(
        users_data_dict, "phone_number", "phone_number_extracted"
    )

    # Load users score ########################################################
    logger.info("Loading users score")
    users_score = load_cloudburst_users_score()
    logger.info("Users score loaded")
    ###########################################################################

    # Convert to dictionary ##################################################
    logger.info("Converting to dictionary...")
    users_score_dict = tuple_to_dict(users_score, keys=USERS_SCORES_COLUMNS)
    logger.info("Users score converted to dictionary")
    ###########################################################################

    # Merge the 2 list of dictionaries ########################################
    logger.info("Merging the 2 list of dictionaries...")
    users_featured = merge_dicts(
        users_data_dict_country, users_score_dict, join_attr="pid"
    )
    logger.info("Merged")

    ###########################################################################

    # Save dict to json file ##################################################
    logger.info("Saving to json file...")
    save_dict(users_featured, "users_featured.json")
    logger.info("Saved")
