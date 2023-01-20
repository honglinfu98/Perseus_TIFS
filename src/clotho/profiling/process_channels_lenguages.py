"""
This script is used to process the channels to extract the characteristics for later use
on profiling the channels
"""
import pprint
import logging
from langdetect import detect

# Call logger
logger = logging.getLogger()


def detect_language(message: str) -> str:
    """
    This function detects the language of the content of the channels
    :param message: Message to detect the language
    :return: Language detected
    """
    try:
        lang = detect(str(message))
    except Exception as lang_detect_e:  # TODO Add specific exceptions, too broad
        logger.error("Error: %s", lang_detect_e)
        return None
    return lang


def detect_lang_list_dict(messages: list[dict], min_length: int = 50) -> list[dict]:
    """
    This function detects the language of the content of the channels
    :param messages: List of dictionaries with the messages to detect the language
    :return: List of dictionaries with the messages and the language detected
    """
    for idx, message in enumerate(messages):
        if len(message["message_text"]) > min_length:
            message["language"] = detect_language(message["message_text"])

        if idx % 1000 == 0:
            logger.info("Messages processed: %s", idx)

    return messages


if __name__ == "__main__":
    messages = [
        {
            "entity_id": 1,
            "message_text": "Hello, I hope you're doing well. How was your day today?",
        },
        {
            "entity_id": 2,
            "message_text": "Hola, espero que estés bien. ¿Cómo ha sido tu día hoy?",
        },
        {
            "entity_id": 2,
            "message_text": "Bonjour, j'espère que vous allez bien. Comment s'est passée votre journée aujourd'hui?",
        },
        {
            "entity_id": 1,
            "message_text": "Hello, I hope you're doing well. How was your day today?",
        },
    ]

    messages = detect_lang_list_dict(messages)

    pprint.pprint(messages)
