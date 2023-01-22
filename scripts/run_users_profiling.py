"""
This script runs the flowork of the users profiling.
"""
import logging
from clotho.extraction.loader import load_cloudburst_users, COLUMNS_NAMES_USERS
from clotho.pre_processing.create_dict import tuple_to_dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

if __name__ == "__main__":
    # Download the users data for users profiling ##############################
    logger.info("Loading users-members")
    user_members = load_cloudburst_users()
    logger.info("Users-members loaded")

    logger.info("Number of users-members extracted: %s", len(user_members))

    logger.info("Converting to dictionary...")
    user_members_df = tuple_to_dict(user_members, keys=COLUMNS_NAMES_USERS)
    logger.info("Users-members converted to dictionary")
    ###########################################################################

    logger.info("Detecting users' scripts on username, first_name, last_name")

    logger.info("Users' characteristics detected")

    logger.info("Processing phone number data")

    logger.info("Phone number data processed")
