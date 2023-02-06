"""
This script is used to process the channels to extract the characteristics for later use
on profiling the channels
"""
import pprint
import logging
from langdetect import detect

# Call logger
logger = logging.getLogger()


def detect_language(message: str, min_length: int = 50) -> str:
    """
    This function detects the language of the content of the channels
    :param message: Message to detect the language
    :return: Language detected
    """
    if message is None:

        return "None"

    try:

        if len(message) > min_length:

            lang = detect(str(message))

        else:

            lang = "Too short"

        return lang

    except (Exception) as lang_detect_e:  # TODO Add specific exceptions, too broad
        logger.error("Error: %s", lang_detect_e)

        return f"Error: {lang_detect_e}"


def detect_lang_list_dict(
    messages: list[dict], keys_to_process: list[str]
) -> list[dict]:
    """
    This function detects the language of the content of the channels
    check if the key exists in the dictionary
    if the key language exists, append the new language detected
    if it's not a list, make it a list
    if it's a list append the new language detected
    if the key doesn't exist, create it and add the language detected in form of a list
    :param messages: List of dictionaries with the messages to detect the language
    :param key_to_process: List of Keys to process
    :return: List of dictionaries with the messages and the language detected
    """

    for idx, message in enumerate(messages):
        if idx % 1000 == 0:
            logger.info("Processed %s messages", idx)
        for key in keys_to_process:

            if "language" in message.keys():

                if not isinstance(message["language"], list):

                    message["language"] = [message["language"]]

                if message[key] is not None:

                    message["language"].append(detect_language(message["language"][0]))

            else:

                message["language"] = [detect_language(message[key])]

    return messages


if __name__ == "__main__":
    messages = [
        {
            "entity_id": 1,
            "message_text": "Hello, I hope you're doing well. How was your day today?",
            "bio": "Hola, espero que estés bien. ¿Cómo ha sido tu día hoy?",
        },
        {
            "entity_id": 2,
            "message_text": "Hola, espero que estés bien. ¿Cómo ha sido tu día hoy?",
            "bio": "Hola, espero que estés bien. ¿Cómo ha sido tu día hoy?",
        },
        {
            "entity_id": 2,
            "message_text": "Bonjour, j'espère que vous allez bien. Comment s'est passée votre journée aujourd'hui?",
            "bio": "Hola, espero que estés bien. ¿Cómo ha sido tu día hoy?",
        },
        {
            "entity_id": 1,
            "message_text": None,
            "bio": None,
        },
    ]

    messages = detect_lang_list_dict(messages, ["message_text", "bio"])

    pprint.pprint(messages)
