"""
This script runs the flowork of the users profiling.
"""
import logging
from clotho.extraction.loader import load_cloudburst_users, COLUMNS_NAMES_USERS
from clotho.pre_processing.create_dict import tuple_to_dict

# from clotho.profiling.process_users_characters import detect_scripts
from clotho.profiling.process_users_bio import extract_features

# from clotho.profiling.process_users_bio import

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

if __name__ == "__main__":
    # Download the users data for users profiling ##############################
    logger.info("Loading users")
    users_data = load_cloudburst_users()
    logger.info("Users loaded")

    logger.info("Number of users extracted: %s", len(users_data))

    logger.info("Converting to dictionary...")
    users_data_dict = tuple_to_dict(users_data, keys=COLUMNS_NAMES_USERS)
    logger.info("Users converted to dictionary")

    # Detect users' scripts ##################################################
    logger.info("Detecting users' scripts on username, first_name and last_name")
    script_keys_to_process = ["username", "first_name", "last_name"]
    users_data_featured = extract_features(users_data_dict, script_keys_to_process)
    logger.info("Users' scripts detected")
    ###########################################################################

    # Bio data extraction ####################################################
    logger.info("Extracting features from bio")
    key_to_process = ["bio"]
    users_data_featured = extract_features(users_data_featured, key_to_process)
    logger.info("Features extracted from bio")
    ###########################################################################
