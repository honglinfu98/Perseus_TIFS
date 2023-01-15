"""
This script process the users phone numbers
and labels the country of the phone number
"""

import pandas as pd
from extraction.loader import load_cloudburst_users_data

# Function to create a dict from a 2 dataframe column


def create_dict(code_df: pd.DataFrame, col1: str, col2: str) -> dict:
    """
    This function create a dict from a 2 dataframe column
    :param df: Dataframe
    :param col1: Column name
    :param col2: Column name
    :return: Dict
    """
    return dict(zip(code_df[col2], code_df[col1]))


def label_countries(users_df: pd.DataFrame, codes_dict: dict):
    """
    This function labels the country of the phone number
    :param df: Dataframe with the column phone_number
    :param country_code_dict: Dictionary with the country code as key and the country as value
    :return: Dataframe with the column country
    """
    # Create a dict with the country code as key and the country as value
    # Create a new column country
    users_df['country'] = ''
    # Order the country code dict by length of the code, first the longest
    # This is to avoid matching the wrong country code
    # For example United States has the code 1,
    # but if we match the code 1 first, we will get the wrong country
    # TODO this parcially solves the problem, but it is not perfect
    # TODO we need to find a better way to match the country code
    # TODO maybe we can use the phone number length too
    # TODO or maybe we can use the phone number format
    # TODO we have another library that can help us with this
    # 
    codes_dict = {k: v for k, v in sorted(
        codes_dict.items(), key=lambda item: len(item[0]), reverse=True)}

    #Extract from the df only the columns we will use
    users_df = users_df[['user_PID','phone_number','country']]
    for index, row in users_df.iterrows():
        phone_number = row['phone_number']
        if pd.isnull(phone_number):
            continue
        for code, country in codes_dict.items():
            if phone_number[:len(code)] == code:
                # print(
                #     f"Matched: {phone_number[:len(code)]}, {code}, {country}")
                users_df.loc[index, 'country'] = country

    # If the country is still empty, and there is a phone number give it a value of 'Unknown'
    users_df.loc[(users_df['country'] == '') & (
        users_df['phone_number'].notnull()), 'country'] = 'Unknown'

    # If the country is still empty, and there is no phone number give it a value of 'No available phone number'
    users_df.loc[(users_df['country'] == '') & (users_df['phone_number'].isnull()),
           'country'] = 'No available phone number'

    return users_df


if __name__ == '__main__':

    # Calling the users data from cloudburst
    users_data = load_cloudburst_users_data()

    # Call the phone codes database
    phone_code_df = pd.read_csv("../support_data/cellphones_code.csv", sep=";")
    # Create a dict with the country code as key and the country as value
    phone_code_dict = create_dict(phone_code_df, "country", "country_code")
    # Label the country of the phone number
    users_data = label_countries(users_data, phone_code_dict)