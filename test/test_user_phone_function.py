"""
Unit tests for process_users_phone_numbers.py
"""
import pytest
import pandas as pd
from profiling_scripts.process_users_phone_numbers import label_countries, create_dict

@pytest.mark.parametrize(
    "dataframe,code_dict,output",
    [
        (
            pd.DataFrame(
                {
                    "user_PID":[1, 2, 3, 4, 5, 6],
                    "phone_number": [
                        "1-2025550176",
                        "34-202-555-0176",
                        "54-202-555-0176",
                        "312025550176",
                        "12-202-555-0176",
                        None,
                    ],
                }
            ),
            {
                "1": "United States",
                "54": "Argentina",
                "34": "Spain",
                },
            pd.DataFrame(
                {   
                    "user_PID":[1, 2, 3, 4, 5, 6],
                    "phone_number": [
                        "1-2025550176",
                        "34-202-555-0176",
                        "54-202-555-0176",
                        "312025550176",
                        "12-202-555-0176",
                        None,
                    ],
                    "country": [
                        "United States",
                        "Spain",
                        "Argentina",
                        "Unknown",
                        "United States",
                        "No available phone number",
                    ],
                }
            ),
        )
    ],
)

def test_label_countries(dataframe:pd.DataFrame, code_dict:dict, output:pd.DataFrame)->None:
    """
    This function test tdhe function label_countries
    """
    print(label_countries(dataframe, code_dict))
    print(output)
    assert label_countries(dataframe, code_dict).equals(output)


#Create a unit test with pytest for function create_dict in process_users_phone_numbers.py

@pytest.mark.parametrize(
    "code_df,col1,col2",
    [
        (
            pd.DataFrame(
                {
                    "country_code": ["1", "13", "14", "55"],
                    "country": [
                        "United States",
                        "Unknown",
                        "Unknown",
                        "Unknown",
                    ],
                }
            ),
            "country",
            "country_code",
        )
    ],
)
def test_create_dict(code_df:pd.DataFrame, col1:str, col2:str)->dict:
    """"
    This function test the function create_dict
    :param code_df: Dataframe
    :param col1: Column name
    :param col2: Column name
    :return: Dict
    """
    assert create_dict(code_df, col1, col2) == {
        "1": "United States",
        "13": "Unknown",
        "14": "Unknown",
        "55": "Unknown",
    }
