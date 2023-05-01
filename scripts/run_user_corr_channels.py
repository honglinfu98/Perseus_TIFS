"""
This script runs the correlation between users in the same channel
"""
import logging
from clotho.extraction.loader import (
    load_cloudburst_channel_members,
    COLUMNS_NAMES_CHANNEL_MEMBERS,
)
from clotho.pre_processing.create_dict import tuple_to_dict
from clotho.correlations.correlate_shared_chats import run_user_corr_channels
from clotho.post_processing.save_dict_to_json import save_dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


if __name__ == "__main__":
    channel_members = load_cloudburst_channel_members()

    logger.info("Channel members loaded")

    channel_members_dict = tuple_to_dict(channel_members, COLUMNS_NAMES_CHANNEL_MEMBERS)

    logger.info("Channel members converted to dict")

    logger.info("Counting pair features")

    channel_members_dict = run_user_corr_channels(channel_members_dict)
    # Save dict to json file ##################################################
    logger.info("Saving to json file...")
    save_dict(channel_members_dict, "channel_members_dict.json")
    logger.info("Saved")
