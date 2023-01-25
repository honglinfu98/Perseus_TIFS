"""
This module contains the functions to extract the urls, accounts and hashtags from the bio of each user
"""
import re
import unicodedata
from urllib.parse import urlparse


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


def extract_urls(string: str) -> list[str]:
    """
    Extracts the urls of a string
    :param string: String to extract the urls
    :return: List of urls extracted
    """
    try:
        urls = re.findall(
            r"(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+(?<!\.)(?<!\.)\.[\w/\-?=%.]+(?<!\.)",
            string,
        )

        pattern = re.compile(
            r"(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+[a-zA-Z]{2,}\.[\w/\-?=%.]+(?<!\.)"
        )
        return [url for url in urls if pattern.match(url)]
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


def get_email(string: str) -> list[str]:
    """
    Extracts the email of a string
    :param string: String to extract the email
    :return: List of email extracted
    """
    try:
        return [
            word
            for word in string.split()
            if re.match("^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", word)
        ]
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


def get_possible_phone_numbers(string: str) -> list[str]:
    """
    Extracts the phone numbers of a string
    :param string: String to extract the phone numbers
    :return: List of phone numbers extracted
    """
    try:
        return re.findall(r"\+?[0-9]{10,}", string)
    except TypeError:
        return []


def extract_features(data: list[dict], key_list: list[str]) -> list[dict]:
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
        for user in data:
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
    return data


if __name__ == "__main__":

    dict = [
        {"pid": 1, "bio": "Blockchain is the Future"},
        {"pid": 2, "bio": "أول منصة عراقية للعملات الرقمية | @nakhlexchange"},
        {"pid": 3, "bio": "Nothing."},
        {"pid": 7, "bio": "،", "phone_number": "431532453245"},
        {
            "pid": 8,
            "bio": "elegant girl✨. 34613794258",
            "phone_number": "431532453245",
        },
        {"pid": 9, "bio": "..."},
        {"pid": 14, "bio": "تسجيل كباتن كريم واوبر وجيني", "phone_number": []},
        {"pid": 15, "bio": "تبو قحطان"},
        {"pid": 16, "bio": "https://t.me/li5x12", "phone_number": None},
        {"pid": 17, "bio": "Twitter : @angieo4_ Tg: @angiedaguro"},
        {"pid": 25, "bio": "@mmmmmm"},
        {"pid": 26, "bio": ".للتواصل معي اࢪسل ڪلمة (ﺣّ͠ـلْـم) او(𝙳𝚁𝙴𝙰𝙼) 🙂"},
        {"pid": 27, "bio": "6 year experience in trading ❤️"},
        {"pid": 28, "bio": "Anh Tài"},
        {"pid": 29, "bio": "ــ ㅤㅤ(:♡0:00 ●━━━━━━─────── ♾ ⇆ㅤㅤ◁ㅤㅤ❚❚ㅤㅤ▷ㅤㅤㅤㅤ↻"},
        {"pid": 31, "bio": "fghj"},
        {"pid": 32, "bio": "fb.com/ahmed.ahmed.507"},
        {"pid": 33, "bio": "تعاملاتي : @repsReal_orez اكثر من 550 تقيم"},
        {
            "pid": 34,
            "bio": "www.instagram.com/ahmed_ahmed_507 sandiasndiands somtheing@some.as",
        },
        {"pid": 35, "bio": "تعاملاتي : @repsReal_orez اكثر من 550 تقيم"},
        {"pid": 36, "bio": None},
        {"pid": 37, "bio": ""},
    ]

    from pprint import pprint

    pprint(extract_features(dict, ["bio"]))
