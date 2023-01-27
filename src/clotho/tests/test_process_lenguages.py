"""
Test the process_lenguages.py module
"""

from clotho.profiling.process_lenguages import detect_language, detect_lang_list_dict

import sys

sys.path.append("..")


def test_detect_language():
    """
    Test the function detect_language
    """
    # Test with a string in English
    string = "Hello, my name is 1234"
    expected = "Too short"  # Is empty because the string is too short
    assert detect_language(string) == expected

    # Test with a string in English
    string = "Hello, my name is 1234 and I am from the United States of America, I like to eat pizza and drink beer."
    expected = "en"
    assert detect_language(string) == expected

    # Test with a string in Spanish
    string = "Hola, mi nombre es 1234 y soy de los Estados Unidos de América, me gusta comer pizza y beber cerveza."
    expected = "es"
    assert detect_language(string) == expected

    # Test with a string in French
    string = "Bonjour,  je m'appelle 1234 et je suis des États-Unis d'Amérique, j'aime manger de la pizza et boire de la bière."
    expected = "fr"
    assert detect_language(string) == expected

    # Test with a None string
    string = None
    expected = "None"  # The try-except catch and returns the error
    assert detect_language(string) == expected  # type: ignore - is for testing purposes


def test_detect_lang_list_dict():
    """
    Test the function detect_lang_list_dict
    """
    # Test with a list of dictionaries
    dict_with_messages = [
        {
            "entity_id": 1,
            "message_text": "Hello, I hope you're doing well. How was your day today?, I like to eat pizza and drink beer and I am from the United States of America.",
        },
        {
            "entity_id": 2,
            "message_text": "Hola, espero que estés bien. ¿Cómo ha sido tu día hoy?, me gusta comer pizza y beber cerveza y soy the Argentina.",
        },
        {
            "entity_id": 2,
            "message_text": "Bonjour, j'espère que vous allez bien. Comment s'est passée votre journée aujourd'hui?",
        },
        {
            "entity_id": 1,
            "message_text": None,
        },
    ]
    expected = [
        {
            "entity_id": 1,
            "message_text": "Hello, I hope you're doing well. How was your day today?, I like to eat pizza and drink beer and I am from the United States of America.",
            "language": ["en"],
        },
        {
            "entity_id": 2,
            "message_text": "Hola, espero que estés bien. ¿Cómo ha sido tu día hoy?, me gusta comer pizza y beber cerveza y soy the Argentina.",
            "language": ["es"],
        },
        {
            "entity_id": 2,
            "message_text": "Bonjour, j'espère que vous allez bien. Comment s'est passée votre journée aujourd'hui?",
            "language": ["fr"],
        },
        {
            "entity_id": 1,
            "message_text": None,
            "language": ["None"],
        },
    ]

    output = detect_lang_list_dict(dict_with_messages, keys_to_process=["message_text"])
    print(output)
    print(expected)
    assert output == expected
