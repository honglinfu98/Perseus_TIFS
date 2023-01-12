"""
Unit tests for process_users_phone_numbers.py
"""
import pytest
import pandas as pd
from profiling_scripts.process_channels_lenguages import detect_and_group_by_language

#df -> Columns on input: channel_id, name, message_text
#list[tuples] -> Columns on output: channel_id, name, languages

@pytest.mark.parametrize(
    "channel_members,output",
    [
        (
            pd.DataFrame(
                [
                    [1, "channel1", "Hello, I hope you're doing well. How was your day today?"],
                    [1, "channel1", "Hola, espero que estés bien. ¿Cómo ha sido tu día hoy?"],
                    [1, "channel1", "Bonjour, j'espère que vous allez bien. Comment s'est passée votre journée aujourd'hui?"],
                    [1, "channel1", "Hello, It's a nice day today. How about you?"],
                    [1, "channel2", "Hola, espero que ayer te haya ido bien. ¿Cómo ha sido tu día hoy?"],
                ],
                columns=["entity_id", "username", "message_text"],
            ),
            [
                (1, "channel1", ["en", "fr", "es"]),
                (1, "channel2", ["es"]),
            ],
        ),
    ],
)

def test_detect_and_group_by_language(channel_members: pd.DataFrame, output: list[tuple])->None:
    """
    Test the function detect_and_group_by_language
    :param channel_members: Input dataframe
    :param output: Expected output
    """
    detected_lenguages = detect_and_group_by_language(channel_members)
    #Sort the output by the first element of the tuple
    detected_lenguages = [(x[0], x[1], sorted(x[2])) for x in detected_lenguages]
    #Create a new tuple and sort alphabetically the last element of the tuple
    output = [(x[0], x[1], sorted(x[2])) for x in output]
    print(output)
    assert detected_lenguages == output
