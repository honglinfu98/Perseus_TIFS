"""
This script processes the phone numbers of the users
with the help of the phonenumbers library
"""
import phonenumbers as phonenumbers
import pandas as pd
from extraction.loader import load_cloudburst_users_data

def label_countries(users_df: pd.DataFrame):
    """
    This function labels the country of the phone number
    :param df: Dataframe with the column phone_number
    :return: Dataframe with the column country
    """
    # Create a new column country
    users_df['country'] = ''
    for index, row in users_df.iterrows():
        phone_number = row['phone_number']
        if pd.isnull(phone_number):
            continue
        try:
            parsed_number = phonenumbers.parse(phone_number)
            region_code = phonenumbers.region_code_for_number(parsed_number)
            users_df.loc[index, 'country'] = region_code
        except phonenumbers.phonenumberutil.NumberParseException:
            users_df.loc[index, 'country'] = 'Invalid phone number'

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
    # Label the country of the phone number
    users_data = label_countries(users_data)
