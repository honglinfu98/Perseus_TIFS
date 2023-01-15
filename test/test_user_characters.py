"""
This file contains the tests for the functions in the file process_characteristics/process_users_characters.py
"""
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from profiling.process_users_characters import run_detect_scripts

@pytest.mark.parametrize(
    "user_members,output",
    [
        (
            pd.DataFrame(
                {
                    "user_PID": [1, 2, 3, 4, 5,6],
                    "username": ["user1", "user2", "user3", "user4", "user5",None],
                    "first_name": ["John", "Ελένη", "张", "أحمد", "Иван",None],
                    "last_name": ["Smith", "Johnson", "Wang", "Mohammed", "Ivanov",None]
                }
            ),
            pd.DataFrame(
                {
                    "user_PID": [1, 2, 3, 4, 5,6],
                    "username": ["user1", "user2", "user3", "user4", "user5",None],
                    "first_name": ["John", "Ελένη", "张", "أحمد", "Иван",None],
                    "last_name": ["Smith", "Johnson", "Wang", "Mohammed", "Ivanov",None],
                    "scripts": [['DIGIT', 'LATIN'],['DIGIT', 'LATIN', 'GREEK'],['DIGIT', 'LATIN', 'CJK'],['DIGIT', 'LATIN', 'ARABIC'],['DIGIT', 'LATIN', 'CYRILLIC'],[]],
                }
            ),

        ),
    ],
)


#Function to test the function run_detect_scripts. Specially for working with df.
def test_run_detect_scripts(user_members: pd.DataFrame, output: pd.DataFrame)-> None:
    """
    Test the function run_detect_scripts
    """
    detected_scripts = run_detect_scripts(user_members)
    detected_scripts["scripts"] = detected_scripts["scripts"].apply(lambda x: sorted(x))
    output["scripts"] = output["scripts"].apply(lambda x: sorted(x))
    print(detected_scripts)
    print(output)
    assert_frame_equal(detected_scripts, output, check_like=True, check_dtype=False, check_column_type=False)

# def test_run_detect_scripts(user_members, output):
#     detected_scripts = run_detect_scripts(user_members)
#     print(detected_scripts)
#     print(output)
#     assert detected_scripts == output
