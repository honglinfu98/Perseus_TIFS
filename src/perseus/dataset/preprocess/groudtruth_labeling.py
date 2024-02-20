from os import path
import pickle
from perseus.dataset.extract.cloudburst_connection import (
    get_masterminds,
    get_train_masterminds,
    get_scored_signals,
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
        # with open(path.join(PROJECT_ROOT, "data", "signals2.pkl"), "rb") as file:
        #     data = pickle.load(file)        
        with open(path.join(PROJECT_ROOT, "src","perseus","dataset","extract","download","test_signals.pkl"), "rb") as file:
            data = pickle.load(file)       
        data = data[~data["telegram_chat_id"].isna()]

        # data = get_scored_signals()
    elif train_or_test == "train":
        # with open(path.join(PROJECT_ROOT, "data", "signals1.pkl"), "rb") as file:
        #     data = pickle.load(file)
        with open(path.join(PROJECT_ROOT, "src","perseus","dataset","extract","download","train_signals.pkl"), "rb") as file:
            data = pickle.load(file)       
        data = data[~data["telegram_chat_id"].isna()]
    elif train_or_test == "validate":
        # with open(path.join(PROJECT_ROOT, "data", "signals3.pkl"), "rb") as file:
        #     data = pickle.load(file)

        with open(path.join(PROJECT_ROOT, "src","perseus","dataset","extract","download","validate_signals.pkl"), "rb") as file:
            data = pickle.load(file)   
        data = data[~data["telegram_chat_id"].isna()]
 
        # data = get_valid_scored_signals()

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

    label_mapping_2 = {}

    for key in label_mapping:
        inner_dict = label_mapping[key]
        first_key = next(iter(inner_dict))  # Get the first key of the inner dictionary
        inner_dict[first_key] = [
            inner_dict[first_key],
            2,
        ]  # Change the first key's value to a list [1, 2]
        label_mapping_2[key] = inner_dict

    return label_mapping_2


if __name__ == "__main__":
    label_mapping = create_label_mapping(3, "train")
