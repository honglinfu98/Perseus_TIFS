"""
This script downloads the users data from the database and converts it to a dictionary.
"""
import logging
from clotho.extraction.loader import load_cloudburst_users, COLUMNS_NAMES_USERS
from clotho.pre_processing.create_dict import tuple_to_dict
from clotho.post_processing.save_dict_to_csv import save_dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


if __name__ == "__main__":

    # Download the channels data for network analysis
    logger.info("Downloading users data for network analysis...")
    cloudburst_channels = load_cloudburst_users()
    logger.info("Users data downloaded")

    logger.info("Number of users extracted: %s", len(cloudburst_channels))

    logger.info("Converting to dictionary...")
    cloudburst_channels = tuple_to_dict(cloudburst_channels, keys=COLUMNS_NAMES_USERS)
    logger.info("Users data converted to dictionary")

    logger.info("Saving users data to csv file...")
    save_dict(cloudburst_channels, "users_data.csv")
    logger.info("Users data saved to csv file")
