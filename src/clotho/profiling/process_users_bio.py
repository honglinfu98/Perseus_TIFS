"""
This module contains the functions to extract the urls, accounts and hashtags from the bio of each user
"""
import re
import pycountry
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


def extract_phone_numbers(string: str) -> list[str]:
    """
    Extracts the phone numbers of a string
    :param string: String to extract the phone numbers
    :return: List of phone numbers extracted
    """
    try:
        return re.findall(r"\+?[0-9]{10,}", string)
    except TypeError:
        return []


def append_phone_numbers(data: list[dict], key: str) -> list[dict]:
    """
    Appends the phone numbers to the data to the key = phone_number, don't overwrite
    if the key is not present, we will create it
    :param data: List of users
    :return: List of users with the phone number appended
    """
    for user in data:
        if "phone_number_extracted" in user:
            if (
                user["phone_number_extracted"] is None
                or user["phone_number_extracted"] == []
            ):
                user["phone_number_extracted"] = []
            else:
                if not isinstance(user["phone_number_extracted"], list):
                    user["phone_number_extracted"] = [user["phone_number_extracted"]]
            extracted_phone_numbers = extract_phone_numbers(user[key])
            user["phone_number_extracted"] += extracted_phone_numbers
        else:
            user["phone_number_extracted"] = extract_phone_numbers(user[key])
    return data


def extract_features(data: list[dict], key_list: list[str]) -> list[dict]:
    """
    Extracts the urls, accounts, hashtags, domains, script, and phone numbers from the bio of each user
    :param data: List of users
    :return: List of users with the urls, accounts, hashtags, domains, script, and phone numbers extracted
    """
    check_list = [
        "urls",
        "accounts",
        "hashtags",
        "domains",
        "script",
        "phone_numbers",
        "emails_adresses",
    ]

    for k in key_list:
        for user in data:
            # check if the key already exists in the user dictionary
            if any(key in user for key in check_list):
                user["urls"] += extract_urls(user[k])
                user["accounts"] += get_accounts(user[k])
                user["hashtags"] += get_hashtags(user[k])
                user["domains"] += [extract_domain(url) for url in user["urls"]]
                user["script"] += detect_script_on_characters(user[k])
                user["emails_adresses"] += get_email(user[k])
            else:
                user["urls"] = extract_urls(user[k])
                user["accounts"] = get_accounts(user[k])
                user["hashtags"] = get_hashtags(user[k])
                user["domains"] = [extract_domain(url) for url in user["urls"]]
                user["script"] = detect_script_on_characters(user[k])
                user["emails_adresses"] = get_email(user[k])

        data = append_phone_numbers(data, k)

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
        {"pid": 34, "bio": "www.instagram.com/ahmed_ahmed_507 sandiasndiands"},
        {"pid": 35, "bio": "تعاملاتي : @repsReal_orez اكثر من 550 تقيم"},
        {"pid": 36, "bio": None},
        {"pid": 37, "bio": ""},
    ]

    from pprint import pprint

    pprint(extract_features(dict, ["bio"]))
