"""
Unit tests for process_users_phonenumbers_library.py
"""
from clotho.profiling.extract_phonenumbers_data import get_region, get_country
import sys

sys.path.append("..")


def test_get_region():
    """
    This function tests the function get_region
    """
    print(get_region("13238741060"))
    assert get_region("13238741060") == "Los Angeles, CA"
    print(get_region("34613794258"))
    assert get_region("34613794258") == "Spain"
    print(get_region("543512194921"))
    assert get_region("543512194921") == "Córdoba, Córdoba"
    print(get_region("999025550176"))
    assert get_region("999025550176") == "Unknown"
    print(get_country("None"))
    assert get_country("None") == "Unknown"


def test_get_country():
    """
    This function tests the function get_country
    """
    print(get_country("13238741060"))
    assert get_country("13238741060") == "United States"
    print(get_country("34613794258"))
    assert get_country("34613794258") == "Spain"
    print(get_country("543512194921"))
    assert get_country("543512194921") == "Argentina"
    print(get_country("999025550176"))
    assert get_country("999025550176") == "Unknown"
    print(get_country("None"))
    assert get_country("None") == "Unknown"
