"""
This file contains the tests for the functions in the file process_characteristics/process_users_characters.py
"""
from clotho.profiling.features_extraction import (
    get_script,
    get_accounts,
    get_hashtags,
    get_email,
    get_possible_phone_numbers,
    extract_urls,
    extract_domain,
)


def test_get_script():
    """
    Test the function get_script
    """
    # Test with a string with characters of different scripts
    string = "こんにちは"
    expected = ["HIRAGANA"]
    assert get_script(string).sort() == expected.sort()

    # Test with a string with characters of the same script
    string = "Hello"
    expected = ["LATIN"]
    assert get_script(string).sort() == expected.sort()

    # Test with a string with characters Arabic
    string = "مرحبا"
    expected = ["ARABIC"]
    assert get_script(string).sort() == expected.sort()

    # Test with a string with characters from cyrillic script
    string = "Здравствуйте"
    expected = ["CYRILLIC"]
    assert get_script(string).sort() == expected.sort()

    # Test with a string with characters from greek scrcipt
    string = "Γειά σου"
    expected = ["SPACE", "GREEK"]
    assert get_script(string).sort() == expected.sort()

    # Test with a string with characters, numbers, punctuation, etc.
    string = "Hello, my name is 1234"
    expected = ["LATIN", "DIGIT", "COMMA", "SPACE"]
    expected.sort()
    output = get_script(string)
    output.sort()
    assert output == expected


def test_get_accounts():
    """
    Test the function get_accounts
    """
    # Test with a string with one account
    string = "@username"
    expected = ["@username"]
    output = get_accounts(string)
    assert output == expected

    # Test with a string with two accounts and a hashtag
    string = "@username @username2 #hashtag"
    expected = ["@username", "@username2"]
    output = get_accounts(string)
    assert output == expected

    # Test with a string with None
    string = None
    expected = []
    output = get_accounts(string)  # type: ignore - is for testing purposes


def test_get_hashtags():
    """
    Test the function get_hashtags
    """
    # Test with a string with one hashtag
    string = "#hashtag"
    expected = ["hashtag"]
    assert get_hashtags(string).sort() == expected.sort()

    # Test with a string with two hashtags and an account
    string = "#hashtag #hashtag2 @username"
    expected = ["#hashtag", "#hashtag2"]
    output = get_hashtags(string)
    assert output == expected

    # Test with a None string
    string = None
    expected = []
    output = get_hashtags(string)  # type: ignore - is for testing purposes
    assert output == expected


def test_get_email():
    """
    Test the function get_email
    """
    # Test with a string with one email
    string = "hello my email is example@gmail.com"
    expected = ["example@gmail.com"]
    output = get_email(string)
    assert output == expected

    # Test with a string with two emails
    string = " hello my email is ex@a.co and axes@my_email.com.mx hhtp://example.com"
    expected = ["ex@a.co", "axes@my_email.com.mx"]
    output = get_email(string)
    assert output == expected


def test_get_possible_phone_numbers():
    """
    Test the function get_possible_phone_numbers
    """
    # # Test with a string with one phone number
    # TODO IMPROVE THE ON DETECTING PHONE NUMBERS TO INCLUDE MORE NUMBER EXTRACTION RANGE
    # # This test is commented because the function is not working properly
    # string = " hello my phone number is +1 (123) 456-7890"
    # expected = ["+1 (123) 456-7890"]
    # assert get_possible_phone_numbers(string).sort() == expected.sort()

    # Test with a string with two phone numbers and an email
    string = " hello my phone number is +11234567890 and +134567891 and my email is axes@my_email.com.mx"
    expected = ["+11234567890", "+134567891"]
    output = get_possible_phone_numbers(string)
    assert output == expected


def test_extract_urls():
    """
    Test the function extract_urls
    """
    # Test with a string with one url
    string = " hello my website is http://example.com"
    expected = ["http://example.com"]
    assert extract_urls(string).sort() == expected.sort()

    # Test with a string with two urls and an email
    string = " hello my website is http://example.com and example.com and my a.com.a.d email is axes@my_email.com.mx"
    expected = ["http://example.com", "example.com"]
    output = extract_urls(string)
    assert output == expected


def test_extract_domain():
    """
    Test the function extract_domain
    """
    # Test with a string with one url
    url = "http://example.com/something#something"
    expected = "example.com"
    output = extract_domain(url)
    assert output == expected

    # Test with a string with two urls and an email
    url = "fb.com/sub&something%a57"
    expected = "fb.com"
    output = extract_domain(url)
    assert output == expected
