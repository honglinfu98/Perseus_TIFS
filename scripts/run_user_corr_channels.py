import logging
import pandas as pd
from clotho.extraction.loader import (
    load_cloudburst_channel_members,
    COLUMNS_NAMES_CHANNEL_MEMBERS,
)
from clotho.pre_processing.create_dict import tuple_to_dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def first_group(df_user_to_group: pd.DataFrame) -> pd.DataFrame:
    # Change data type to string and create a new column with the correlated entity_ids for each chat_PID
    df_user_to_group["chat_PID"] = df_user_to_group["chat_PID"].astype(str)
    df_user_to_group["user_PID"] = df_user_to_group["user_PID"].astype(str)

    df_user_to_group["correlated_chats_ids"] = df_user_to_group.groupby(["chat_PID"])[
        "user_PID"
    ].transform(lambda x: ",".join(x))

    # set the new column as an array of strings to manage it better
    df_user_to_group["correlated_chats_ids"] = df_user_to_group[
        "correlated_chats_ids"
    ].str.split(",")

    # Remove the chats with their own from the correlated_chats_ids list
    for index, row in df_user_to_group.iterrows():
        correlated_chats_ids = row["correlated_chats_ids"]
        user_PID = row["user_PID"]
        correlated_chats_ids = [e for e in correlated_chats_ids if e != user_PID]
        df_user_to_group.at[index, "correlated_chats_ids"] = correlated_chats_ids

    # contar el número de correlated_chats_ids por chat_PID
    df_user_to_group["correlated_chats_ids_count"] = df_user_to_group[
        "correlated_chats_ids"
    ].str.len()
    # order by syze of correlated_chats_ids_count
    df_user_to_group = df_user_to_group.sort_values(
        by=["correlated_chats_ids_count"], ascending=False
    )

    return df_user_to_group


def group_data(df: pd.DataFrame, appearence: int = 0) -> pd.DataFrame:
    """
    This function groups the data by user_PID and counts the amount of times it appears the same correlated_chats_ids,
    then it converts the columns correlated_chats_ids from list of list to dictionary and leave only the ones who has
    over a certain amount of times it appears
    :param df: pd.DataFrame
    :param appearence: int by default the value is 0
    :return: pd.DataFrame
    """
    # group by user_PID and count the amount of times it appears the same correlated_chats_ids
    grouped_df = (
        df.groupby(["user_PID"])["correlated_chats_ids"].apply(list).reset_index()
    )
    # convert the columns correlated_chats_ids from list of list to dictionary
    grouped_df["correlated_chats_ids"] = grouped_df["correlated_chats_ids"].apply(
        lambda x: [item for sublist in x for item in sublist]
    )
    grouped_df["correlated_chats_ids"] = grouped_df["correlated_chats_ids"].apply(
        lambda x: dict((i, x.count(i)) for i in x)
    )
    # leave only the ones who has over a certain amount of times it appears
    grouped_df["correlated_chats_ids"] = grouped_df["correlated_chats_ids"].apply(
        lambda x: {k: v for k, v in x.items() if v >= appearence}
    )
    # eliminate the rows that have an empty dictionary
    grouped_df = grouped_df[grouped_df["correlated_chats_ids"].map(len) > 0]
    return grouped_df


def change_data_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function changes the data sctructure to be able to use it in the network graph
    :param df: pd.DataFrame
    :return: pd.DataFrame
    """
    # Initialize the dataframe
    output_df = pd.DataFrame(columns=["user1", "user2", "weight"])

    # Iterate through each row of the original dataframe
    for i, row in df.iterrows():
        user_PID = row["user_PID"]
        correlated_entity_ids = row["correlated_chats_ids"]
        # Only add users that are diferent user1 from user2

        for correlated_entity_id, value in correlated_entity_ids.items():
            # Append a new row to the output dataframe
            output_df = output_df.append(
                {"user1": user_PID, "user2": correlated_entity_id, "weight": value},
                ignore_index=True,
            )
    return output_df


def drop_duplicated_rows(df_to_clean: pd.DataFrame) -> pd.DataFrame:
    """
    This function sorts the user1 and user2 columns
    """
    df_to_clean.iloc[:, 0:2] = list(
        df_to_clean[["user1", "user2"]].apply(sorted, axis=1)
    )  # type: ignore
    # drop duplicates
    df_to_clean = df_to_clean.drop_duplicates()
    return df_to_clean


if __name__ == "__main__":

    # channel_members_dict = [
    #     {"user_PID": "1", "chat_PID": "1"},
    #     {"user_PID": "2", "chat_PID": "1"},
    #     {"user_PID": "4", "chat_PID": "2"},
    #     {"user_PID": "4", "chat_PID": "1"},
    #     {"user_PID": "1", "chat_PID": "2"},
    #     {"user_PID": "1", "chat_PID": "3"},
    #     {"user_PID": "2", "chat_PID": "3"},
    #     {"user_PID": "2", "chat_PID": "2"},
    #     {"user_PID": "22", "chat_PID": "9"},
    # ]

    channel_members = load_cloudburst_channel_members()

    logger.info("Channel members loaded")
    logger.info("Number of channel members: %s", len(channel_members))
    channel_members_dict = tuple_to_dict(channel_members, COLUMNS_NAMES_CHANNEL_MEMBERS)
    logger.info("Channel members converted to dict")

    logger.info("Number of channel members: %s", len(channel_members_dict))

    logger.info("Counting pair features")

    channel_members_dict = pd.DataFrame(channel_members_dict)
    logger.info("Channel members converted to dataframe")
    output = first_group(channel_members_dict)
    logger.info("First group done")
    output = group_data(channel_members_dict, 3)
    logger.info("Group data done")
    output = change_data_structure(output)
    logger.info("Change data structure done")
    output = drop_duplicated_rows(output)
    logger.info("Drop duplicated rows done")
