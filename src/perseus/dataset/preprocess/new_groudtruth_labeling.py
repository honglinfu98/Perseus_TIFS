from os import path
import pickle
import pandas as pd
from perseus.dataset.preprocess.train_test_validate import (

    get_test_scored_signals,
    get_train_scored_signals,
    get_valid_scored_signals,
)
from perseus.settings import PROJECT_ROOT


# Function to create label mapping based on top n frequency
def create_label_mapping(n: int, train_or_test: str):
    # signals = get_scored_signals()
    if train_or_test == "test":
        # with open(path.join(PROJECT_ROOT, "data", "signals.pkl"), "rb") as file:
        #     data = pickle.load(file)
        data = get_test_scored_signals()
    elif train_or_test == "train":
        data = get_train_scored_signals()
    elif train_or_test == "valid":
        data = get_valid_scored_signals()

    # Grouping by commodity and telegram_chat_id and counting the occurrences
    commodity_frequency = (
        data.groupby(["commodity", "telegram_chat_id"])
        .size()
        .reset_index(name="frequency")
    )

    # Sorting within each commodity group by frequency in descending order
    commodity_frequency_sorted = commodity_frequency.sort_values(
        ["commodity", "frequency"], ascending=[True, False]
    )

    # Extracting the sorted list for each commodity
    commodity_frequency_lists = commodity_frequency_sorted.groupby("commodity").apply(
        lambda x: x[["telegram_chat_id", "frequency"]].values.tolist()
    )

    # Example: Create label mapping for top 3 telegram_chat_ids for each commodity
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

    # label_mapping_2 = {}

    # for key in label_mapping:
    #     inner_dict = label_mapping[key]
    #     first_key = next(iter(inner_dict))  # Get the first key of the inner dictionary
    #     inner_dict[first_key] = [
    #         inner_dict[first_key],
    #         2,
    #     ]  # Change the first key's value to a list [1, 2]
    #     label_mapping_2[key] = inner_dict

    return label_mapping





def export_csv_for_labeling():
    # filter commodity less than 3 and with respect to the gs keys 


    train_label_mapping = create_label_mapping(1, "train")
    test_label_mapping = create_label_mapping(1, "test")
    valid_label_mapping = create_label_mapping(1, "valid")

    data = []

    for outer_key, inner_dict in train_label_mapping.items():
        for inner_key, value in inner_dict.items():
            data.append((outer_key, inner_key, value))

    # Convert the list into a DataFrame
    df = pd.DataFrame(data, columns=['Symbol', 'Code', 'Value'])

    # Export to CSV
    csv_file_path = path.join(PROJECT_ROOT, "data", "train_labeling.csv")
    df.to_csv(csv_file_path, index=False)



    data = []
    
    for outer_key, inner_dict in test_label_mapping.items():
        for inner_key, value in inner_dict.items():
            data.append((outer_key, inner_key, value))

    # Convert the list into a DataFrame
    df = pd.DataFrame(data, columns=['Symbol', 'Code', 'Value'])

    # Export to CSV
    csv_file_path = path.join(PROJECT_ROOT, "data", "test_labeling.csv")
    df.to_csv(csv_file_path, index=False)

    
    data = []
    
    for outer_key, inner_dict in valid_label_mapping.items():
        for inner_key, value in inner_dict.items():
            data.append((outer_key, inner_key, value))

    # Convert the list into a DataFrame
    df = pd.DataFrame(data, columns=['Symbol', 'Code', 'Value'])

    # Export to CSV
    csv_file_path = path.join(PROJECT_ROOT, "data", "valid_labeling.csv")
    df.to_csv(csv_file_path, index=False)


    return




def read_labeling_csv_back_to_dict():
    # Read the edited CSV file
    train_edited_df = pd.read_csv(path.join(PROJECT_ROOT, "data", "train_labeling.csv"))

    edited_train_label_mapping = {}
    for _, row in train_edited_df.iterrows():
        if row['Symbol'] not in edited_train_label_mapping:
            edited_train_label_mapping[row['Symbol']] = {}
        edited_train_label_mapping[row['Symbol']][row['Code']] = row['Value']


    test_edited_df = pd.read_csv(path.join(PROJECT_ROOT, "data", "test_labeling.csv"))

    edited_test_label_mapping = {}
    for _, row in test_edited_df.iterrows():
        if row['Symbol'] not in edited_test_label_mapping:
            edited_test_label_mapping[row['Symbol']] = {}
        edited_test_label_mapping[row['Symbol']][row['Code']] = row['Value']

    

    valid_edited_df = pd.read_csv(path.join(PROJECT_ROOT, "data", "valid_labeling.csv"))

    edited_valie_label_mapping = {}
    for _, row in valid_edited_df.iterrows():
        if row['Symbol'] not in edited_valie_label_mapping:
            edited_valie_label_mapping[row['Symbol']] = {}
        edited_valie_label_mapping[row['Symbol']][row['Code']] = row['Value']


    return edited_train_label_mapping,edited_test_label_mapping, edited_valie_label_mapping



    


if __name__ == "__main__":
    train_label_mapping = create_label_mapping(1, "train")
    test_label_mapping = create_label_mapping(1, "test")
    valid_label_mapping = create_label_mapping(1, "valid")