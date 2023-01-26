"""
This script runs the flowork of the users profiling.
"""
import logging
from clotho.extraction.loader import load_cloudburst_users, COLUMNS_NAMES_USERS
from clotho.pre_processing.create_dict import tuple_to_dict
from clotho.profiling.features_extraction import extract_features
from clotho.profiling.extract_phonenumbers_data import (
    extract_data_phonenumber,
)

# from clotho.profiling.process_users_bio import

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

if __name__ == "__main__":
    # Download the users data for users profiling ############################
    logger.info("Loading users")
    users_data = load_cloudburst_users()
    logger.info("Users loaded")
    logger.info("Number of users extracted: %s", len(users_data))

    # Convert to dictionary ##################################################
    logger.info("Converting to dictionary...")
    users_data_dict = tuple_to_dict(users_data, keys=COLUMNS_NAMES_USERS)
    logger.info("Users converted to dictionary")
    ###########################################################################

    # Extract features from: username, first_name, last_name, bio #############
    logger.info("Extracting features from: username, first_name, last_name, bio")
    script_keys_to_process = ["username", "first_name", "last_name", "bio"]
    # We will extract the features from the keys in the script_keys_to_process list
    # Features extracted: urls, accounts, hashtags, domains, script, phone_numbers, emails_adresses
    users_data_featured = extract_features(users_data_dict, script_keys_to_process)
    logger.info("Features extracted")
    ###########################################################################

    # Extract phone numbers data###############################################
    logger.info("Extracting phone numbers data")
    users_data_featured = extract_data_phonenumber(
        users_data_featured, "phone_number", "phone_number_extracted"
    )
    logger.info("Phone numbers data extracted")
    ###########################################################################

    # ###########################################################################
    # ###########################################################################
    # # Checking of the results #################################################
    # # Filters #################################################################
    # filt_users_phones = []
    # for user in users_data_featured:
    #     if user["phone_number_extracted"] != []:
    #         filt_users_phones.append(user)

    # filt_users_emails = []
    # for user in users_data_featured:
    #     if user["emails_adresses"] != []:
    #         filt_users_emails.append(user)

    # filt_users_urls = []
    # for user in users_data_featured:
    #     if user["urls"] != []:
    #         filt_users_urls.append(user)

    # filt_users_accounts = []
    # for user in users_data_featured:
    #     if user["accounts"] != []:
    #         filt_users_accounts.append(user)

    # filt_users_hashtags = []
    # for user in users_data_featured:
    #     if user["hashtags"] != []:
    #         filt_users_hashtags.append(user)

    # filt_users_domains = []
    # for user in users_data_featured:
    #     if user["domains"] != []:
    #         filt_users_domains.append(user)

    # filt_users_scripts = []
    # for user in users_data_featured:
    #     if user["script"] != []:
    #         filt_users_scripts.append(user)

    # ###########################################################################

    # # Lists with the filters applied ##########################################
    # emails_list = [user["emails_adresses"] for user in filt_users_emails]
    # urls_list = [user["urls"] for user in filt_users_urls]
    # accounts_list = [user["accounts"] for user in filt_users_accounts]
    # hashtags_list = [user["hashtags"] for user in filt_users_hashtags]
    # domains_list = [user["domains"] for user in filt_users_domains]
    # scripts_list = [user["script"] for user in filt_users_scripts]
    # phone_numbers_list = [user["phone_number_extracted"] for user in filt_users_phones]

    # ###########################################################################

    # # Show the results ########################################################
    # logger.info(
    #     "-------------------->>>> Number of users with email: %s",
    #     len(filt_users_emails),
    # )
    # logger.info("Emails: %s", emails_list[:50])
    # logger.info("-------------------->>>> Number of Urls: %s", len(urls_list))
    # # show first 50 urls
    # logger.info("Urls: %s", urls_list[:50])
    # logger.info("-------------------->>>> Number of Accounts: %s", len(accounts_list))
    # logger.info("Accounts: %s", accounts_list[:50])
    # logger.info("-------------------->>>> Number of Hashtags: %s", len(hashtags_list))
    # logger.info("Hashtags: %s", hashtags_list[:50])
    # logger.info("-------------------->>>> Number of Domains: %s", len(domains_list))
    # logger.info("Domains: %s", domains_list[:50])
    # logger.info("-------------------->>>> Number of Scripts: %s", len(scripts_list))
    # # List of list to list
    # scripts_list = [item for sublist in scripts_list for item in sublist]
    # # List of unique scriptS
    # scripts_list = list(set(scripts_list))
    # logger.info("Scripts: %s", scripts_list[:50])
    # logger.info(
    #     "-------------------->>>> Number of Phone Numbers: %s", len(phone_numbers_list)
    # )
    # logger.info("Phone Numbers: %s", phone_numbers_list[:50])

    # ###########################################################################

    # # Show first dict with pprint from each filter ############################
    # from pprint import pprint

    # print("First 2 dicts with pprint from each filter:")
    # print("Phone number detected: ")
    # print("Email detected: ")
    # pprint(filt_users_emails[0:1])
    # print("Url detected: ")
    # pprint(filt_users_urls[0:1])
    # print("Account detected: ")
    # pprint(filt_users_accounts[0:1])
    # print("Hashtag detected: ")
    # pprint(filt_users_hashtags[0:1])
    # print("Domain detected: ")
    # pprint(filt_users_domains[0:1])
    # print("Script detected: ")
    # pprint(filt_users_scripts[0:1])
    # print("Phone number detected: ")
    # pprint(filt_users_phones[0:1])
    # ###########################################################################

    # # Filter russian users   ###########################################
    # filt_users_russian = []
    # for user in users_data_featured:
    #     # need to be one string on the list from the key script that match "cyrillic"
    #     if any("CYRILLIC" in s for s in user["script"]):
    #         filt_users_russian.append(user)
    # pprint(filt_users_russian[0:1])

    # # From filt_phone_numbers_list to filter a dict with only the keys that we want

    # keys_list = [
    #     "username",
    #     "first_name",
    #     "last_name",
    #     "bio",
    #     "phone_number",
    #     "phone_number_extracted",
    #     "country",
    #     "region",
    #     "carrier",
    #     "phone_number_type",
    # ]
    # filt_users_phones_clean = [
    #     {k: v for k, v in user.items() if k in keys_list} for user in filt_users_phones
    # ]
