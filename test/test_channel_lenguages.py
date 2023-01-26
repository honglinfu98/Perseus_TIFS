"""
Unit tests for process_users_phone_numbers.py
"""
import pytest
import pandas as pd
from clotho.profiling.process_lenguages import detect_and_group_by_language

# df -> Columns on input: channel_id, name, message_text
# list[tuples] -> Columns on output: channel_id, name, languages


@pytest.mark.parametrize(
    "channel_messages,output",
    [
        (
            pd.DataFrame(
                [
                    [
                        1,
                        "channel1",
                        "Hello, I hope you're doing well. How was your day today?",
                    ],
                    [
                        1,
                        "channel1",
                        "Hola, espero que estés bien. ¿Cómo ha sido tu día hoy?",
                    ],
                    [
                        1,
                        "channel1",
                        "Bonjour, j'espère que vous allez bien. Comment s'est passée votre journée aujourd'hui?",
                    ],
                    [1, "channel1", "Hello, It's a nice day today. How about you?"],
                    # This message won't be detected on the count because it's too short (less than 50 characters)
                    [
                        2,
                        "channel2",
                        "Hola, espero que ayer te haya ido bien. ¿Cómo ha sido tu día hoy?",
                    ],
                ],
                columns=["entity_id", "username", "message_text"],
            ),
            [
                (1, "channel1", {"en": 1, "fr": 1, "es": 1}),
                (2, "channel2", {"es": 1}),
            ],
        ),
    ],
)
def test_detect_and_group_by_language(
    channel_messages: pd.DataFrame, output: list[tuple]
) -> None:
    """
    Test the function detect_and_group_by_language
    :param channel_members: Input dataframe
    :param output: Expected output
    """
    detected_languages = detect_and_group_by_language(channel_messages)
    # Convert the languages dictionary to a list of tuples, sort it by key and convert it back to a dictionary
    detected_languages = [
        (x[0], x[1], dict(sorted(x[2].items()))) for x in detected_languages
    ]
    output = [(x[0], x[1], dict(sorted(x[2].items()))) for x in output]
    print(detected_languages)
    print(output)
    assert detected_languages == output
