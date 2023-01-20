import re
import pandas as pd
from urllib.parse import urlparse
from clotho.profiling.process_users_characters import detect_script_on_characters


def extract_urls(string: str) -> list[str]:
    """
    Extracts the urls of a string
    :param string: String to extract the urls
    :return: List of urls extracted
    """
    try:
        return re.findall(
            r"(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+(?<!\.)(?<!\.)\.[\w/\-?=%.]+(?<!\.)",
            string,
        )
    except TypeError:
        return []


def get_accounts(string: str) -> list[str]:
    """
    Extracts the accounts of a string
    :param string: String to extract the accounts
    :return: List of accounts extracted
    """
    try:
        return [word for word in string.split() if re.match("^@", word)]
    except AttributeError:
        return []


def get_hashtags(string: str) -> list[str]:
    """
    Extracts the hashtags of a string
    :param string: String to extract the hashtags
    :return: List of hashtags extracted
    """
    try:
        return [word for word in string.split() if re.match("^#", word)]
    except AttributeError:
        return []


def process_users_bio(
    bio_tuples_list: list[tuple], columns_names: list[str]
) -> pd.DataFrame:
    """
    Processes the bio of the users
    :param users_members: Merged dataframe
    :return: Dataframe with the bio, urls, accounts, hashtags and domains
    """
    bios = pd.DataFrame(bio_tuples_list, columns=columns_names)
    bios["urls"] = bios["bio"].apply(extract_urls)
    bios["accounts"] = bios["bio"].apply(get_accounts)
    bios["hashtags"] = bios["bio"].apply(get_hashtags)
    bios["domains"] = bios["urls"].apply(lambda x: [urlparse(url).netloc for url in x])
    bios["characters"] = bios["bio"].apply(detect_script_on_characters)
    return bios
