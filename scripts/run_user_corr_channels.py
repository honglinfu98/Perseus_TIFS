import logging
import pandas as pd
from clotho.extraction.loader import (
    load_cloudburst_channel_members,
    COLUMNS_NAMES_CHANNEL_MEMBERS,
)
from clotho.pre_processing.create_dict import tuple_to_dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


if __name__ == "__main__":
    # channel_members = load_cloudburst_channel_members()
    # logger.info("Channel members loaded")
    # logger.info("Number of channel members: %s", len(channel_members))
    # channel_members_dict = tuple_to_dict(channel_members, COLUMNS_NAMES_CHANNEL_MEMBERS)
    # logger.info("Channel members converted to dict")
    # Dummy data

    # logger.info("Number of channel members: %s", len(channel_members_dict))
    # # Run the function
    logger.info("Counting pair features")

    def get_common_keys(channel_members_dict: pd.DataFrame) -> pd.DataFrame:
        """
        This function takes a list of dictionaries and returns a dataframe with a new column
        where it stores a list of all the users that share the same chat_ID
        """
        df = pd.DataFrame(channel_members_dict)
        # Create a new column where we will store in a list all the users that share the same chat_ID
        df["common_users"] = ""
        # Fill with all the user_ID that share the same chat_ID that the user_ID in that row
        for index, row in df.iterrows():
            df.at[index, "common_users"] = list(
                df.loc[df["chat_ID"] == row["chat_ID"], "user_ID"]
            )
        # Delete the user_ID from the row from that list in the column common_users
        for index, row in df.iterrows():
            df.at[index, "common_users"] = [
                x for x in row["common_users"] if x != row["user_ID"]
            ]

        return df

    # Iterate to create a new dataframe:
    # Each combination will be a row (user_id, commor_user, value) the value will be always 1 at the moment
    # The new dataframe will have the following columns: user_id, common_use, value
    # At the end will leave only one combination per row (user_id, common_user) no matter the order it's the same
    def create_new_df(df: pd.DataFrame) -> pd.DataFrame:
        new_df = pd.DataFrame(columns=["user_id", "common_user", "value"])
        for index, row in df.iterrows():
            for user in row["common_users"]:
                new_df = new_df.append(
                    {"user_id": row["user_ID"], "common_user": user, "value": 0},
                    ignore_index=True,
                )
        new_df = new_df.groupby(["user_id", "common_user"]).sum().reset_index()
        new_df["value"] = 1
        return new_df

    channel_members_dict = [
        {"user_ID": "1", "chat_ID": "1"},
        {"user_ID": "2", "chat_ID": "1"},
        {"user_ID": "4", "chat_ID": "2"},
        {"user_ID": "4", "chat_ID": "1"},
        {"user_ID": "1", "chat_ID": "2"},
        {"user_ID": "2", "chat_ID": "2"},
        {"user_ID": "22", "chat_ID": "9"},
    ]

    output = get_common_keys(channel_members_dict)
    output = create_new_df(output)
    from pprint import pprint

    pprint(output)
