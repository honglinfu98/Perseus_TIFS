"""
This script processes the phone numbers of the users
with the help of the phonenumbers library
"""
import pycountry
import phonenumbers
from phonenumbers import geocoder
from phonenumbers import carrier


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
    except (phonenumbers.phonenumberutil.NumberParseException, LookupError):
        return "Unknown"


def get_phone_number_type(phone: str) -> str:
    """
    This function labels the type of the phone number
    """
    try:
        phone_number = phonenumbers.parse("+" + str(phone))
        return str(phonenumbers.number_type(phone_number))
    except phonenumbers.phonenumberutil.NumberParseException:
        return "Unknown"


def get_phone_number_carrier(phone: str) -> str:
    """
    This function labels the carrier of the phone number
    """
    try:
        phone_number = phonenumbers.parse("+" + str(phone))
        return str(carrier.name_for_number(phone_number, "en"))
    except phonenumbers.phonenumberutil.NumberParseException:
        return "Unknown"


def extract_data_phonenumber(
    phone: list[dict], phonenumbers_key: str, phonenumbers_extracted_key: str
) -> list[dict]:
    """
    This function will go over each dict from the provided list
    and process the keys provided it will create a new key by feature extraction.
    The new keys will be:
    - region
    - country
    - phone_number_type
    - carrier
    :param phone: list of dicts
    :param phonenumbers_key: key of the phone number
    :param phonenumbers_extracted_key: key of the extracted phone numbers
    :return: list of dicts with the new keys
    """
    for user in phone:
        user["region"] = [get_region(user[phonenumbers_key])]
        user["country"] = [get_country(user[phonenumbers_key])]
        user["phone_number_type"] = [get_phone_number_type(user[phonenumbers_key])]
        user["carrier"] = [get_phone_number_carrier(user[phonenumbers_key])]
        # Now we will process the phone numbers extracted
        # and append to the the keys created above the values of the extracted phone numbers
        for number in user[phonenumbers_extracted_key]:
            user["region"] += [get_region(number)]
            user["country"] += [get_country(number)]
            user["phone_number_type"] += [get_phone_number_type(number)]
            user["carrier"] += [get_phone_number_carrier(number)]
    return phone


if __name__ == "__main__":
    # Dummy data for testing
    data = [
        {
            "user_id": 1,
            "phone_number": 13238741060,  # Los Angeles
            "phone_number_extracted": [34613794258, 987654321],  # Spain, Unknown
        },
        {
            "user_id": 2,
            "phone_number": 13238741060,  # Los Angeles
            "phone_number_extracted": [543512194921, 987654321],  # Argentina, Unknown
        },
        {
            "user_id": 3,
            "phone_number": 543512194921,  # Argentina
            "phone_number_extracted": [13238741060, 987654321],  # Los Angeles, Unknown
        },
    ]

    # Extract the data
    data_phone_numbers = extract_data_phonenumber(
        data, "phone_number", "phone_number_extracted"
    )
