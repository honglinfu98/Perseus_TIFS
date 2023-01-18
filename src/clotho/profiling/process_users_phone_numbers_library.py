"""
This script processes the phone numbers of the users
with the help of the phonenumbers library
"""
import pandas as pd
import pycountry
import phonenumbers
from phonenumbers import geocoder
from extraction.loader import load_cloudburst_users_data

def get_region(phone: str) -> str:
    """
    This function labels the country of the phone number
    """
    try:
        phone_number = phonenumbers.parse("+" + str(phone))
        return geocoder.description_for_number(phone_number, "en")
    except phonenumbers.phonenumberutil.NumberParseException:
        return "Unknown"
    
def get_country(phone: str) -> str:
    """
    This function labels the country of the phone number
    """
    try:
        phone_number = phonenumbers.parse("+" + str(phone))
        country_code = phonenumbers.region_code_for_number(phone_number)
        country = pycountry.countries.get(alpha_2=country_code)
        if country is None:
            return "Unknown"
        else:
            return country.name
    except phonenumbers.phonenumberutil.NumberParseException:
        return "Unknown"

def join_phone_number_data(users_members_data: pd.DataFrame) -> list[tuple]:
    """
    This function joins the phone number data to the user data
    keeping the user_PID column for each user
    """
    # Applying the function to label the country of the phone number
    users_members_data = users_members_data[users_members_data["phone_number"].apply(lambda x: x is not None)]
    # Label the region of the phone number
    users_members_data["region"] = users_members_data["phone_number"].apply(get_region)
    # Label the country of the phone number
    users_members_data["country"] = users_members_data["phone_number"].apply(get_country)

    return users_members_data
    
if __name__ == '__main__':

    # Calling the users data from cloudburst
    user_data = load_cloudburst_users_data()
    # Applying the function to run all the scripts
    join_phone_number_data(user_data)
