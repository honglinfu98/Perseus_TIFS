from os import path
import pandas as pd
import pickle
from os import path
from perseus.dataset.preprocess.train_test_validate import (
    get_test_scored_signals,
    get_train_scored_signals,
    get_valid_scored_signals,
)

from perseus.settings import PROJECT_ROOT
from concurrent.futures import ProcessPoolExecutor, as_completed
from perseus.scripts.data_checking.row_processer import process_row


if __name__ == "__main__":
    # open signals_btc_retuen.csv from PROJECT_ROOT/data
    signals_btc_return = pd.read_csv(
        path.join(PROJECT_ROOT, "data", "signals_btc_return.csv")
    )

    # open btc return from PROJECT_ROOT/data
    btc_return = pd.read_csv(path.join(PROJECT_ROOT, "data", "returns.csv"))

    # rename btc_return increase_percentage to btc_increase_percentage
    btc_return.rename(
        columns={"increase_percentage": "btc_increase_percentage"}, inplace=True
    )

    # merge the signals_btc_return with get_test_scored_signals, get_train_scored_signals, get_valid_scored_signals on id
    train_signals = get_train_scored_signals()
    train_signals = train_signals.merge(
        btc_return, on="id", how="left", suffixes=("", "_btc")
    )

    # filter test_signals' message_text and get the ones with btc or BTC in the text
    btc_train_signals = train_signals[
        train_signals["message_text"].str.contains("btc", case=False, na=False)
    ]
    # filter btc_test_signals and get the ones with na btc_increase_percentage
    btc_train_signals_na = btc_train_signals[
        btc_train_signals["btc_increase_percentage"].isna()
    ]

    # filter out btc_train_signals_na with commodity == "BTC"
    btc_train_signals_na = btc_train_signals_na[
        btc_train_signals_na["commodity"] != "BTC"
    ]

    valid_signals = get_valid_scored_signals()
    valid_signals = valid_signals.merge(
        btc_return, on="id", how="left", suffixes=("", "_btc")
    )

    # filter test_signals' message_text and get the ones with btc or BTC in the text
    btc_valid_signals = valid_signals[
        valid_signals["message_text"].str.contains("btc", case=False, na=False)
    ]
    # filter btc_test_signals and get the ones with na btc_increase_percentage
    btc_valid_signals_na = btc_valid_signals[
        btc_valid_signals["btc_increase_percentage"].isna()
    ]

    btc_valid_signals_na = btc_valid_signals_na[
        btc_valid_signals_na["commodity"] != "BTC"
    ]

    test_signals = get_test_scored_signals()
    test_signals = test_signals.merge(
        btc_return, on="id", how="left", suffixes=("", "_btc")
    )

    # filter test_signals' message_text and get the ones with btc or BTC in the text
    btc_test_signals = test_signals[
        test_signals["message_text"].str.contains("btc", case=False, na=False)
    ]
    # filter btc_test_signals and get the ones with na btc_increase_percentage
    btc_test_signals_na = btc_test_signals[
        btc_test_signals["btc_increase_percentage"].isna()
    ]

    btc_test_signals_na = btc_test_signals_na[btc_test_signals_na["commodity"] != "BTC"]

    results_train = []
    row_dicts = btc_train_signals_na.to_dict(orient="records")

    with ProcessPoolExecutor(max_workers=12) as executor:
        future_to_row = {
            executor.submit(process_row, row): row["id"] for row in row_dicts
        }

        for future in as_completed(future_to_row):
            row_id = future_to_row[future]
            try:
                result = future.result()
                if result:
                    results_train.append(result)
                    print(f"Completed processing row id {row_id}")
            except Exception as e:
                print(f"Row {row_id} generated an exception: {e}")

    results_valid = []
    row_dicts = btc_valid_signals_na.to_dict(orient="records")

    with ProcessPoolExecutor(max_workers=12) as executor:
        future_to_row = {
            executor.submit(process_row, row): row["id"] for row in row_dicts
        }

        for future in as_completed(future_to_row):
            row_id = future_to_row[future]
            try:
                result = future.result()
                if result:
                    results_valid.append(result)
                    print(f"Completed processing row id {row_id}")
            except Exception as e:
                print(f"Row {row_id} generated an exception: {e}")

    results_test = []
    row_dicts = btc_test_signals_na.to_dict(orient="records")

    with ProcessPoolExecutor(max_workers=12) as executor:
        future_to_row = {
            executor.submit(process_row, row): row["id"] for row in row_dicts
        }

        for future in as_completed(future_to_row):
            row_id = future_to_row[future]
            try:
                result = future.result()
                if result:
                    results_test.append(result)
                    print(f"Completed processing row id {row_id}")
            except Exception as e:
                print(f"Row {row_id} generated an exception: {e}")

    train_missing_df = pd.DataFrame(results_train)
    train_missing_df.rename(
        columns={"increase_percentage": "btc_increase_percentage"}, inplace=True
    )
    valid_missing_df = pd.DataFrame(results_valid)
    valid_missing_df.rename(
        columns={"increase_percentage": "btc_increase_percentage"}, inplace=True
    )
    test_missing_df = pd.DataFrame(results_test)
    test_missing_df.rename(
        columns={"increase_percentage": "btc_increase_percentage"}, inplace=True
    )

    # merge the train_missing_df with train_signals on id
    train_signals = train_signals.merge(
        train_missing_df, on="id", how="left", suffixes=("", "_missing")
    )
    train_signals["btc_base_return"] = (
        train_signals["btc_increase_percentage"]
        + train_signals["btc_increase_percentage_missing"]
    )
    train_signals.drop(
        columns=["btc_increase_percentage", "btc_increase_percentage_missing"],
        inplace=True,
    )
    # save the train_signals to PROJECT_ROOT/data/train_signals_btc.pkl
    with open(path.join(PROJECT_ROOT, "data", "train_signals_btc.pkl"), "wb") as file:
        pickle.dump(train_signals, file)

    # merge the valid_missing_df with valid_signals on id
    valid_signals = valid_signals.merge(
        valid_missing_df, on="id", how="left", suffixes=("", "_missing")
    )
    valid_signals["btc_base_return"] = (
        valid_signals["btc_increase_percentage"]
        + valid_signals["btc_increase_percentage_missing"]
    )
    valid_signals.drop(
        columns=["btc_increase_percentage", "btc_increase_percentage_missing"],
        inplace=True,
    )
    # save the valid_signals to PROJECT_ROOT/data/valid_signals_btc.pkl
    with open(path.join(PROJECT_ROOT, "data", "valid_signals_btc.pkl"), "wb") as file:
        pickle.dump(valid_signals, file)

    # merge the test_missing_df with test_signals on id
    test_signals = test_signals.merge(
        test_missing_df, on="id", how="left", suffixes=("", "_missing")
    )
    test_signals["btc_base_return"] = (
        test_signals["btc_increase_percentage"]
        + test_signals["btc_increase_percentage_missing"]
    )
    test_signals.drop(
        columns=["btc_increase_percentage", "btc_increase_percentage_missing"],
        inplace=True,
    )
    # save the test_signals to PROJECT_ROOT/data/test_signals_btc.pkl
    with open(path.join(PROJECT_ROOT, "data", "test_signals_btc.pkl"), "wb") as file:
        pickle.dump(test_signals, file)
