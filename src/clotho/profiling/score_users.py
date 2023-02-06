"""
This script is used to score users based on their activity on scored pump channels.
"""
import logging
from clotho.extraction.loader import (
    load_cloudburst_signals,
    COLUMNS_NAMES_SIGNALS,
    load_cloudburst_channel_members,
    COLUMNS_NAMES_CHANNEL_MEMBERS,
    load_cloudburst_channels,
    COLUMN_NAMES_CHANNELS,
)
from clotho.pre_processing.create_dict import tuple_to_dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def users_chats_dict(data: list[dict]) -> list[dict]:
    """
    Creates a dict of users and the chats they are in
    :param data: list of dicts (user_PID, chat_PID)
    example: [{"user_PID": "1", "chat_PID": "1"}, {"user_PID": "2", "chat_PID": "2"}]
    :return: dict of users and the chats they are in
    example: {"1": {"1"}, "2": {"2"}}
    """
    users = {}
    logger.info("Step 1: Creating users dict")
    for idx, d in enumerate(data):
        user_PID = d["user_PID"]
        chat_PID = d["chat_PID"]
        if user_PID in users:
            users[user_PID].add(chat_PID)
        else:
            users[user_PID] = set([chat_PID])
        if idx % 1000 == 0:
            logger.info("Processed %s rows - Step 1", idx)
    # create a new dict with the following structure:
    # {"user_PID": user.keys(), "chat_PID": user.values()}
    dict_users = []
    for user in users:
        dict_users.append({"user_PID": user, "chat_PID": list(users[user])})
    return dict_users


def filter_channels_with_score(channels_data_dict: list[dict]) -> list:
    """
    Filter channels with score.
    :param channels_data_dict: List of dicts with channels data.
    :return: List of dicts with unique "entity_id" and "score" the keys: ["entity_id", "channel_time_score", "channel_crowd_score"]
    """
    channels_with_score = []
    for channel in channels_data_dict:
        if (
            channel["channel_time_score"] != 0
            and channel["channel_time_score"] is not None
        ):
            channels_with_score.append(
                {
                    "entity_id": channel["entity_id"],
                    "channel_time_score": channel["channel_time_score"],
                }
            )
        if (
            channel["channel_crowd_score"] != 0
            and channel["channel_crowd_score"] is not None
        ):
            channels_with_score.append(
                {
                    "entity_id": channel["entity_id"],
                    "channel_crowd_score": channel["channel_crowd_score"],
                }
            )

        # Remove duplicates
        channels_with_score = [
            dict(t) for t in {tuple(d.items()) for d in channels_with_score}
        ]
    return channels_with_score


def filter_entity_id_and_pid(chats_data_dict: list[dict]) -> list[dict]:
    """
    Filter from chats keys pid and entity_id.
    :param chats_data_dict: List of dicts with chats data.
    :return: List of dicts
    """
    chats_data_dict_filtered = []
    for chat in chats_data_dict:
        chats_data_dict_filtered.append(
            {
                "entity_id": chat["entity_id"],
                "chat_PID": chat["pid"],
            }
        )
    return chats_data_dict_filtered


def map_chat_entities(users, entities):
    """
    Map chat entities to users.
    :param users: List of dicts with users data.
    :param entities: List of dicts with entities data.
    :return: List of dicts with users data and their chat entities.
    """
    entity_map = {entity["chat_PID"]: entity["entity_id"] for entity in entities}
    result = []
    for user in users:
        chat_entities = [
            entity_map[pid] for pid in user["chat_PID"] if pid in entity_map
        ]
        result.append({"user_PID": user["user_PID"], "chat_PID": chat_entities})
    return result


def score_users(channels_with_score: list[dict], users_list: list[dict]) -> list[dict]:
    """
    Score users based on the channels they are in.
    :param channels_with_score: List of dicts with channels data.
    :param users_list: List of dicts with users data.
    :return: List of dicts with users data and their scores.
    """
    for user in users_list:
        user["user_time_score"] = 0
        user["user_crowd_score"] = 0
        for channel in channels_with_score:
            if channel["entity_id"] in user["chat_PID"]:
                if "channel_time_score" in channel:
                    user["user_time_score"] += channel["channel_time_score"]
                if "channel_crowd_score" in channel:
                    user["user_crowd_score"] += channel["channel_crowd_score"]
    return users_list


if __name__ == "__main__":
    # Download the signals data for users scoring ############################
    signals_data = load_cloudburst_signals()
    signals_data_dict = tuple_to_dict(signals_data, COLUMNS_NAMES_SIGNALS)
    # ###########################################################################

    # Filter channels and their scores ########################################
    channels_with_scores = filter_channels_with_score(signals_data_dict)
    # ###########################################################################

    # Download the users data for users scoring ##############################
    users_data = load_cloudburst_channel_members()
    users_data_dict = tuple_to_dict(users_data, COLUMNS_NAMES_CHANNEL_MEMBERS)
    ###########################################################################

    # Create a dict with users and their chats ################################
    users_chats = users_chats_dict(users_data_dict)
    ###########################################################################

    # Download the channels data for users scoring ###########################
    channels_data = load_cloudburst_channels()
    channels_data_dict = tuple_to_dict(channels_data, COLUMN_NAMES_CHANNELS)
    # ###########################################################################

    # Filter from chats keys pid and entity_id ###############################
    chats_data_dict_filtered = filter_entity_id_and_pid(channels_data_dict)
    # ###########################################################################

    # Score users based on the channels they are in ###########################
    users_entity_id = map_chat_entities(users_chats, chats_data_dict_filtered)
    ###########################################################################

    # # Test data for users scoring
    # channels_with_scores = [
    #     {
    #         "entity_id": "1234",
    #         "channel_time_score": 0.0,
    #         "channel_crowd_score": 0.5,
    #     },
    #     {
    #         "entity_id": "12345",
    #         "channel_time_score": 0.6,
    #         "channel_crowd_score": 0.0,
    #     },
    #     {
    #         "entity_id": "123456",
    #         "channel_time_score": 0.0,
    #         "channel_crowd_score": 0.9,
    #     },
    # ]

    # users_entity_id = [
    #     {"user_PID": "1", "chat_PID": ["1234", "12345", "123456"]},
    #     {"user_PID": "2", "chat_PID": ["1234"]},
    #     {"user_PID": "3", "chat_PID": ["1234", "12345"]},
    # ]

    # Score users based on the channels they are in ###########################
    users_with_score = score_users(channels_with_scores, users_entity_id)
    ###########################################################################

    # Filter users with scores != 0 ###########################################
    users_with_score_filtered = [
        user for user in users_with_score if user["user_time_score"] != 0
    ]
    users_with_score_filtered_2 = [
        user for user in users_with_score_filtered if user["user_crowd_score"] != 0
    ]

    users_with_score_filtered = [
        user for user in users_with_score if user["user_time_score"] > 1
    ]
    users_with_score_filtered_2 = [
        user for user in users_with_score_filtered if user["user_crowd_score"] > 1
    ]
    ###########################################################################
