"""
This module contains the functions to correlate the number of shared chats between users
"""
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def users_chats_dict(data: list[dict]) -> dict:
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

    return users


def delete_users(users: dict, min_values: int) -> dict:
    """
    Remove users with less than min_values chats
    :param users: dict of users and the chats they are in
    example: {"1": {"1", "2", "3"}, "2": {"1", "2"}}
    :param min_values: minimum number of chats
    example: 3
    :return: dict of users and the chats they are in
    example: {"1": {"1", "2", "3"}}
    """
    logger.info("Deleting users with less than %s chats", min_values)
    result = {}
    for idx, (user, chats) in enumerate(users.items()):
        if len(chats) >= min_values:
            result[user] = chats
        if idx % 1000 == 0:
            logger.info("Processed %s users - Step 2", idx)
    return result


def counting_shared_chats(users: dict) -> list[dict]:
    """
    Counts the number of shared chats between users
    :param users: dict of users and the chats they are in
    example: {"1": {"1", "2", "3"}, "2": {"1", "2", "3"}}
    :return: dict of users and the number of shared chats
    example: {"1,2": 3, "1,3": 1}
    """
    result = []
    logger.info("Step 2: Counting shared chats")
    for idx, (u1, chats1) in enumerate(users.items()):
        for u2, chats2 in users.items():
            if u1 < u2:
                shared = chats1 & chats2
                if len(shared) >= 3:
                    result.append(
                        {"user1_PID": u1, "user2_PID": u2, "shared": len(shared)}
                    )
        if idx % 1000 == 0:
            logger.info("Processed %s users - Step 2", idx)
    return result


def run_user_corr_channels(channel_members_dict):
    users_chats = users_chats_dict(channel_members_dict)
    logger.info("Users dict created, length: %s", len(users_chats))

    users_chat_filtered = delete_users(users_chats, 2)
    logger.info(
        "Users with less than 3 chats deleted, new length: %s",
        len(users_chat_filtered),
    )

    result = counting_shared_chats(users_chat_filtered)

    return result


if __name__ == "__main__":

    from pprint import pprint

    channel_members_dict = [
        {"user_PID": "1", "chat_PID": "1"},
        {"user_PID": "2", "chat_PID": "1"},
        {"user_PID": "4", "chat_PID": "2"},
        {"user_PID": "4", "chat_PID": "1"},
        {"user_PID": "1", "chat_PID": "2"},
        {"user_PID": "1", "chat_PID": "3"},
        {"user_PID": "2", "chat_PID": "3"},
        {"user_PID": "2", "chat_PID": "2"},
        {"user_PID": "22", "chat_PID": "9"},
    ]

    logger.info("Counting pair features")

    channel_members_dict = run_user_corr_channels(channel_members_dict)

    pprint(channel_members_dict)
