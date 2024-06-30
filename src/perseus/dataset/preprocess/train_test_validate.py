"""
We serperate the data into train, test and validate sets.
"""
from os import path
import pickle
from perseus.settings import PROJECT_ROOT


def seperate_train_test_validate(start_date = '2024-01-10', end_date = '2024-02-05'):
    """
    Seperate the data into train, test and validate sets.
    """

    with open(path.join(PROJECT_ROOT, "data","all_scored.pkl"), "rb") as file:
        df = pickle.load(file)        
    
    df = df[~df["telegram_chat_id"].isna()]


    train_df = df[(df['source_posted_at'] < start_date)]

    test_df = df[(df['source_posted_at'] >= start_date) & (df['source_posted_at'] <= end_date)]

    validate_df = df[(df['source_posted_at'] > end_date)]


    with open(path.join(PROJECT_ROOT, "data","train_signals.pkl"), "wb") as file:
        pickle.dump(train_df,file)        

    with open(path.join(PROJECT_ROOT, "data","test_signals.pkl"), "wb") as file:
        pickle.dump(test_df,file)        

    with open(path.join(PROJECT_ROOT, "data","validate_signals.pkl"), "wb") as file:
        pickle.dump(validate_df,file)        

    return 


def get_train_scored_signals():
    """
    Get signals from the database for pre-pump scoring.
    """


    with open(path.join(PROJECT_ROOT, "data","train_signals.pkl"), "rb") as file:
        signals = pickle.load(file)      
    result = signals[~signals["telegram_chat_id"].isna()]

    return result


def get_test_scored_signals():
    """
    Get signals from the database for pre-pump scoring.
    """


    with open(path.join(PROJECT_ROOT, "data","test_signals.pkl"), "rb") as file:
        signals = pickle.load(file)
    result = signals[~signals["telegram_chat_id"].isna()]

    return result


def get_valid_scored_signals():
    """
    Get signals from the database for pre-pump scoring.
    """

    with open(path.join(PROJECT_ROOT, "data","validate_signals.pkl"), "rb") as file:
        signals = pickle.load(file)       
    result = signals[~signals["telegram_chat_id"].isna()]


    return result



if __name__ == "__main__":
    # seperate_train_test_validate()
    test = get_test_scored_signals()
    train = get_train_scored_signals()
    validate = get_valid_scored_signals()