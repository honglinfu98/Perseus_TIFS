"""
This script is used to out put the label mapping csv file for labeling
"""

from os import path
import pickle
import pandas as pd
from perseus.settings import PROJECT_ROOT


def create_label_mapping(
    n: int,
    train_or_test: str,
):
    """
    Function to create label mapping based on top n frequency, which makes the labeling process easier
    """
    if train_or_test == "test":
        with open(path.join(PROJECT_ROOT, "data", "test_signals.pkl"), "rb") as file:
            data = pickle.load(file)
        data = data[~data["telegram_chat_id"].isna()]

    elif train_or_test == "train":
        with open(path.join(PROJECT_ROOT, "data", "train_signals.pkl"), "rb") as file:
            data = pickle.load(file)
        data = data[~data["telegram_chat_id"].isna()]
    elif train_or_test == "valid":
        with open(
            path.join(PROJECT_ROOT, "data", "validate_signals.pkl"), "rb"
        ) as file:
            data = pickle.load(file)
        data = data[~data["telegram_chat_id"].isna()]

    commodity_frequency = (
        data.groupby(["commodity", "telegram_chat_id"])
        .size()
        .reset_index(name="frequency")
    )

    commodity_frequency_sorted = commodity_frequency.sort_values(
        ["commodity", "frequency"], ascending=[True, False]
    )

    commodity_frequency_lists = commodity_frequency_sorted.groupby("commodity").apply(
        lambda x: x[["telegram_chat_id", "frequency"]].values.tolist()
    )

    label_mapping = {}
    for commodity, freq_list in commodity_frequency_lists.items():
        # Sort and take top n
        top_n_chat_ids = set(
            [
                chat_id
                for chat_id, _ in sorted(freq_list, key=lambda x: x[1], reverse=True)[
                    :n
                ]
            ]
        )
        label_mapping[commodity] = {
            chat_id: 1 if chat_id in top_n_chat_ids else 0 for chat_id, _ in freq_list
        }

    return label_mapping


def export_csv_for_labeling():
    """
    Export the label mapping to CSV for labeling
    """
    # filter commodity less than 3 and with respect to the gs keys

    train_label_mapping = create_label_mapping(1, "train")
    test_label_mapping = create_label_mapping(1, "test")
    valid_label_mapping = create_label_mapping(1, "valid")

    data = []

    for outer_key, inner_dict in train_label_mapping.items():
        for inner_key, value in inner_dict.items():
            data.append((outer_key, inner_key, value))

    # Convert the list into a DataFrame
    df = pd.DataFrame(data, columns=["Symbol", "Code", "Value"])

    # Export to CSV
    csv_file_path = path.join(PROJECT_ROOT, "data", "train_labeling.csv")
    df.to_csv(csv_file_path, index=False)

    data = []

    for outer_key, inner_dict in test_label_mapping.items():
        for inner_key, value in inner_dict.items():
            data.append((outer_key, inner_key, value))

    # Convert the list into a DataFrame
    df = pd.DataFrame(data, columns=["Symbol", "Code", "Value"])

    # Export to CSV
    csv_file_path = path.join(PROJECT_ROOT, "data", "test_labeling.csv")
    df.to_csv(csv_file_path, index=False)

    data = []

    for outer_key, inner_dict in valid_label_mapping.items():
        for inner_key, value in inner_dict.items():
            data.append((outer_key, inner_key, value))

    # Convert the list into a DataFrame
    df = pd.DataFrame(data, columns=["Symbol", "Code", "Value"])

    # Export to CSV
    csv_file_path = path.join(PROJECT_ROOT, "data", "valid_labeling.csv")
    df.to_csv(csv_file_path, index=False)

    return


def read_labeling_csv_back_to_dict(train_test_valid: str):
    """
    Read the edited CSV file
    """
    if train_test_valid == "train":
        train_edited_df = pd.read_csv(
            path.join(PROJECT_ROOT, "data", "train_labeling_finished.csv")
        )

        edited_train_label_mapping = {}
        for _, row in train_edited_df.iterrows():
            if row["Symbol"] not in edited_train_label_mapping:
                edited_train_label_mapping[row["Symbol"]] = {}
            edited_train_label_mapping[row["Symbol"]][row["Code"]] = row["Value"]

        return edited_train_label_mapping

    elif train_test_valid == "test":
        test_edited_df = pd.read_csv(
            path.join(PROJECT_ROOT, "data", "test_labeling_0308.csv")
        )

        edited_test_label_mapping = {}
        for _, row in test_edited_df.iterrows():
            if row["Symbol"] not in edited_test_label_mapping:
                edited_test_label_mapping[row["Symbol"]] = {}
            edited_test_label_mapping[row["Symbol"]][row["Code"]] = row["Value"]

        return edited_test_label_mapping

    elif train_test_valid == "valid":
        valid_edited_df = pd.read_csv(
            path.join(PROJECT_ROOT, "data", "valid_labeling_0318.csv")
        )

        edited_valie_label_mapping = {}
        for _, row in valid_edited_df.iterrows():
            if row["Symbol"] not in edited_valie_label_mapping:
                edited_valie_label_mapping[row["Symbol"]] = {}
            edited_valie_label_mapping[row["Symbol"]][row["Code"]] = row["Value"]

        return edited_valie_label_mapping


if __name__ == "__main__":

    a = read_labeling_csv_back_to_dict("train")
    b = read_labeling_csv_back_to_dict("test")
    c = read_labeling_csv_back_to_dict("valid")
