from os import path
import pickle
from perseus.dataset.preprocess.train_test_validate import (
    get_test_scored_signals,
    get_train_scored_signals,
    get_valid_scored_signals,
)
from perseus.settings import PROJECT_ROOT



# # from os import path
# import pickle 
# from torch_geometric.loader import DataLoader
# from perseus.dataset.compare_paper_graph import combine_features, graph_features
# from perseus.dataset.preprocess.train_test_validate import get_test_scored_signals, get_train_scored_signals, get_valid_scored_signals
# # from perseus.dataset.preprocess.groudtruth_labeling import create_label_mapping
# from perseus.dataset.preprocess.process import features_engineer, get_graphs, process_dataframe, aggregate_data, assign_event_ids
# from perseus.dataset.gnn_dataset_preparation import (
#     prepare_data,
# )

# from perseus.settings import PROJECT_ROOT
# from itertools import combinations
# from sklearn.metrics.pairwise import cosine_similarity
# # from itertools import combinations
# # from sklearn.metrics.pairwise import cosine_similarity

# from torch_geometric.utils import to_undirected, is_undirected
# import torch
# from torch_geometric.data import Data


# Function to create label mapping based on top n frequency
def create_label_mapping(n: int, train_or_test: str, c_features = dict):
    # signals = get_scored_signals()
    if train_or_test == "test":
        # with open(path.join(PROJECT_ROOT, "data", "signals.pkl"), "rb") as file:
        #     data = pickle.load(file)
        # with open(path.join(PROJECT_ROOT, "data", "signals2.pkl"), "rb") as file:
        #     data = pickle.load(file)        
        with open(path.join(PROJECT_ROOT, "data","test_signals.pkl"), "rb") as file:
            data = pickle.load(file)       
        data = data[~data["telegram_chat_id"].isna()]

        # data = get_scored_signals()
    elif train_or_test == "train":
        # with open(path.join(PROJECT_ROOT, "data", "signals1.pkl"), "rb") as file:
        #     data = pickle.load(file)
        with open(path.join(PROJECT_ROOT, "data","train_signals.pkl"), "rb") as file:
            data = pickle.load(file)       
        data = data[~data["telegram_chat_id"].isna()]
    elif train_or_test == "validate":
        # with open(path.join(PROJECT_ROOT, "data", "signals3.pkl"), "rb") as file:
        #     data = pickle.load(file)

        with open(path.join(PROJECT_ROOT, "data","validate_signals.pkl"), "rb") as file:
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

    # for key in label_mapping:
    #     inner_dict = label_mapping[key]
    #     first_key = next(iter(inner_dict))  # Get the first key of the inner dictionary
    #     inner_dict[first_key] = [
    #         inner_dict[first_key],
    #         2,
    #     ]  # Change the first key's value to a list [1, 2]
    #     label_mapping_2[key] = inner_dict

    for key in label_mapping:
        try:
            inner_dict = label_mapping[key]
            inner_df = c_features[key]
            sorted_df = inner_df.sort_values(by=['rating', 'average_speed'], ascending=[False, False])
            top_row = sorted_df.iloc[0]
            top_key = top_row['telegram_chat_id']
            inner_dict[top_key] = [
                inner_dict[top_key],
                2
            ]
            label_mapping_2[key] = inner_dict
        except:
            pass


    return label_mapping_2


if __name__ == "__main__":
    train_signals = get_train_scored_signals()
    test_signals = get_test_scored_signals()
    validate_signals = get_valid_scored_signals()

    gs_ls = []
    features_ls = []
    market_features_ls = []
    P_dicts_ls = []

    for i in [train_signals,test_signals,validate_signals]:
        processed_signals = process_dataframe(i)
        ided_signals = assign_event_ids(processed_signals)
        cascade, no_nodes, id_mapping = aggregate_data(ided_signals)
        gs, results, As, P_dict = get_graphs(cascade, no_nodes, id_mapping)
        graph_feature = graph_features(gs)
        market_feature = features_engineer(processed_signals)
        combine_feature = combine_features(market_feature, graph_feature)
        gs_ls.append(gs)
        features_ls.append(combine_feature)
        market_features_ls.append(market_feature)
        P_dicts_ls.append(P_dict)


    label_mapping_ls = []
    label_mapping_ls.append(create_label_mapping(3, "train", features_ls[0])) 
    label_mapping_ls.append(create_label_mapping(3, "test", features_ls[1]))
    label_mapping_ls.append(create_label_mapping(3, "validate", features_ls[2]))

