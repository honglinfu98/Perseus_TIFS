"""
This module contains the functions to extract features from the users data
it request for the keys to process and the users data.
It returns the users data with the features extracted:
    - urls
    - accounts
    - hashtags
    - domains
    - script
    - alphabets_detected
    - phone_numbers
    - emails_adresses
"""
import re
import unicodedata
from urllib.parse import urlparse
from clotho.constants import LIST_OF_ALPHABETS


def get_script(string: str) -> list[str]:
    """
    This function detects the scripts of a string
    :param string: String to detect the scripts
    :return: List of scripts detected
    """
    scripts_detected = []
    try:
        for character in string:
            name = unicodedata.name(character)
            # If the first word in the name is on known_scripts, replace name with it
            name = name.split(sep=" ")[0]
            scripts_detected.append(name)
        # Return the non repeated scripts
        return list(set(scripts_detected))
    except (ValueError, TypeError):
        return scripts_detected


def extract_urls(string: str) -> list:
    """
    Extracts the urls of a string
    And polish the urls extraction with regex
    :param string: String to extract the urls
    :return: List of urls extracted
    """
    try:
        urls = re.findall(
            r"(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+(?<!\.)(?<!\.)\.[\w/\-?=%.]+(?<!\.)",
            string,
        )

        # Another step regex to avoid extraction when there is a only on character before the dot
        urls = [url for url in urls if not re.match(r"\.[a-z]", url)]

        # Another step regex to avoid extraction when there is an @ before the first dot
        urls = [url for url in urls if not re.match(r".*@.*\..*", url)]

        # Another step regex to avoid extraction when there is cases like ".a..b..c.""
        urls = [url for url in urls if not re.match(r".*\..*\..*\.*", url)]

        return urls
    except TypeError:
        return []


def get_accounts(string: str) -> list:
    """
    Extracts the accounts of a string
    :param string: String to extract the accounts
    :return: List of accounts extracted
    """
    try:
        return [word for word in string.split() if re.match("^@", word)]
    except AttributeError:
        return []


def get_email(string: str) -> list:
    """
    Extracts the email of a string
    :param string: String to extract the email
    :return: List of all the emails extracted
    """
    try:
        return re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", string)
    except TypeError:
        return []


def get_hashtags(string: str) -> list:
    """
    Extracts the hashtags of a string
    :param string: String to extract the hashtags
    :return: List of hashtags extracted
    """
    try:
        return [word for word in string.split() if re.match("^#", word)]
    except AttributeError:
        return []


def extract_domain(url: str) -> str:
    """
    Extracts the domain from a URL
    :param url: URL to extract the domain from
    :return: Domain extracted
    """
    if not url.startswith("http"):
        url = "http://" + url
    parsed_url = urlparse(url)
    return parsed_url.netloc


def get_possible_phone_numbers(string: str) -> list:
    """
    Extracts the phone numbers of a string
    :param string: String to extract the phone numbers
    :return: List of phone numbers extracted
    """
    # TODO: Improve the regex to extract the phone numbers
    try:
        return re.findall(r"\+?[0-9]{7,}", string)
    except TypeError:
        return []


def filter_alphabets_from_script(
    script: list[dict], list_alphabets: list[str]
) -> list[dict]:
    """
    Filters the alphabets from the script
    :param script: List of dictionaries with the script
    :param list_alphabets: List of alphabets to filter
    :return: List of dictionaries with the script filtered
    """
    for x in script:
        for script in x["script"]:
            if script in list_alphabets:
                if "alphabets_detected" in x:
                    # Change the value to a list if it's not already
                    if not isinstance(x["alphabets_detected"], list):
                        x["alphabets_detected"] = [x["alphabets_detected"]]
                    x["alphabets_detected"].append(script)
                else:
                    x["alphabets_detected"] = [script]

    return script


def extract_features(
    data: list[dict],
    key_list: list[str],
    list_of_alphabets: list[str] = LIST_OF_ALPHABETS,
) -> list[dict]:
    """
    Extracts the urls, accounts, hashtags, script, phone numbers, and email addresses from the keys in the key_list
    :return: List dictionaries from the users with the urls, accounts, hashtags,
    script, phone numbers, and email addresses appended
    """
    check_list = [
        "urls",
        "accounts",
        "hashtags",
        "script",
        "phone_numbers",
        "emails_adresses",
    ]

    for k in key_list:
        print(f"Extracting features from the key:'{k}'")
        for idx, user in enumerate(data):
            # if idx % 10000 == 0:
            #     print(f"Extracting features from {idx} users")
            # check if the key already exists in the user dictionary
            if any(key in user for key in check_list):
                user["urls"] += extract_urls(user[k])
                user["domains"] += [extract_domain(url) for url in user["urls"]]
                user["accounts"] += get_accounts(user[k])
                user["hashtags"] += get_hashtags(user[k])
                user["script"] += get_script(user[k])
                user["phone_number_extracted"] += get_possible_phone_numbers(user[k])
                user["emails_adresses"] += get_email(user[k])
            else:
                user["urls"] = extract_urls(user[k])
                user["domains"] = [extract_domain(url) for url in user["urls"]]
                user["accounts"] = get_accounts(user[k])
                user["hashtags"] = get_hashtags(user[k])
                user["script"] = get_script(user[k])
                user["phone_number_extracted"] = get_possible_phone_numbers(user[k])
                user["emails_adresses"] = get_email(user[k])

    # Clean the list from the key script from duplicates
    for user in data:
        user["script"] = list(set(user["script"]))

    data = filter_alphabets_from_script(data, list_of_alphabets)

    return data


if __name__ == "__main__":

    from pprint import pprint

    test_data = [
        {"pid": 1, "bio": "Blockchain is the Future"},
        {"pid": 2, "bio": "@nakhlexchange, أول منصة عراقية للعملات الرقمية"},
        {"pid": 3, "bio": "Nothing."},
        {"pid": 4, "bio": "،", "phone_number": "431532453245"},
        {
            "pid": 8,
            "bio": "elegant girl✨. 34613794258",
            "phone_number": "431532453245",
        },
        {"pid": 5, "bio": "..."},
        {"pid": 6, "bio": "تسجيل كباتن كريم واوبر وجيني", "phone_number": []},
        {"pid": 7, "bio": "تبو قحطان"},
        {"pid": 8, "bio": "https://t.me/li5x12", "phone_number": None},
        {"pid": 9, "bio": "Twitter : @angieo4_ Tg: @angiedaguro"},
        {"pid": 11, "bio": "@mmmmmm"},
        {"pid": 12, "bio": ".للتواصل معي اࢪسل ڪلمة (ﺣّ͠ـلْـم) او(𝙳𝚁𝙴𝙰𝙼) 🙂"},
        {"pid": 13, "bio": "6 year experience in trading ❤️"},
        {"pid": 14, "bio": "Anh Tài"},
        {"pid": 15, "bio": "ــ ㅤㅤ(:♡0:00 ●━━━━━━─────── ♾ ⇆ㅤㅤ◁ㅤㅤ❚❚ㅤㅤ▷ㅤㅤㅤㅤ↻"},
        {"pid": 16, "bio": "fghj"},
        {"pid": 17, "bio": "fb.com/ahmed.ahmed.507"},
        {"pid": 18, "bio": "تعاملاتي : @repsReal_orez اكثر من 550 تقيم"},
        {
            "pid": 19,
            "bio": "www.instagram.com/ahmed_ahmed_507 sandiasndiands somtheing@some.as",
        },
        {"pid": 20, "bio": "تعاملاتي : @repsReal_orez اكثر من 550 تقيم"},
        {"pid": 21, "bio": None},
        {"pid": 22, "bio": ""},
    ]

    # the same test data but adding a random username key
    test_data2 = [
        {"pid": 1, "bio": "Blockchain is the Future", "username": "blockchain"},
        {
            "pid": 2,
            "bio": "أول منصة عراقية للعملات الرقمية | @nakhlexchange",
            "username": "nakhlexchange",
        },
        {"pid": 3, "bio": "Nothing.", "username": "nothing"},
        {"pid": 4, "bio": "،", "phone_number": "431532453245", "username": "nothing"},
        {
            "pid": 8,
            "bio": "elegant girl✨. 34613794258",
            "phone_number": "431532453245",
            "username": "nothing",
        },
        {"pid": 5, "bio": "...", "username": "nothing"},
        {
            "pid": 6,
            "bio": "تسجيل كباتن كريم واوبر وجيني",
            "phone_number": [],
            "username": "my_email@something.com",
        },
        {"pid": 7, "bio": "تبو قحطان", "username": "nothing"},
    ]

    extract_features(test_data, ["bio"])

    extract_features(test_data2, ["bio", "username"])

    pprint(test_data)

    # pprint(test_data2)
