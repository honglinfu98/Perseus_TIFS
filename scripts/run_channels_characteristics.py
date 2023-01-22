"""
This script downloads the channels data from the database and converts it to a dictionary.
"""
import logging
from clotho.extraction.loader import load_cloudburst_channels, COLUMN_NAMES_CHANNELS
from clotho.pre_processing.create_dict import tuple_to_dict
from clotho.post_processing.save_dict_to_csv import save_dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


if __name__ == "__main__":

    # Download the channels data for network analysis
    logger.info("Downloading channels data for network analysis...")
    cloudburst_channels = load_cloudburst_channels()
    logger.info("Channels data downloaded")

    logger.info("Number of channels extracted: %s", len(cloudburst_channels))

    logger.info("Converting to dictionary...")
    cloudburst_channels = tuple_to_dict(cloudburst_channels, keys=COLUMN_NAMES_CHANNELS)
    logger.info("Channels data converted to dictionary")

    logger.info("Saving channels data to csv file...")
    save_dict(cloudburst_channels, "channels.csv")
    logger.info("Channels data saved to csv file")
