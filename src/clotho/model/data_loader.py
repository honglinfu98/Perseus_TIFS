from os import path
import pickle
import random
from torch_geometric.loader import DataLoader
from clotho.settings import PROJECT_ROOT
from clotho.dataset.extract.cloudburst_connection import (
    get_masterminds,
    get_train_masterminds,
    get_scored_signals,
    get_train_scored_signals,
)
from clotho.dataset.preprocess.process import (
    aggregate_data,
    assign_event_ids,
    process_dataframe,
    features_engineer,
    get_graphs,
)
from clotho.dataset.preprocess.groudtruth_labeling import create_label_mapping
from clotho.dataset.gnn_dataset_preparation import (
    graph_features,
    combine_features,
    prepare_data,
)

# with open(path.join(PROJECT_ROOT, "data", "signals.pkl"), "rb") as file:
#     signals = pickle.load(file)


# def get_data_loader():
#     processed_signals = process_dataframe(signals)
#     ided_signals = assign_event_ids(processed_signals)

#     cascade, no_nodes, id_mapping = aggregate_data(ided_signals)
#     gs, results, As, P_dict = get_graphs(cascade, no_nodes, id_mapping)
#     f = graph_features(gs)
#     market_features = features_engineer(processed_signals)
#     c = combine_features(market_features, f)
#     label_mapping = create_label_mapping(3)

#     # Split gs into training and testing sets
#     all_keys = list(gs.keys())
#     random.shuffle(all_keys)
#     split_index = int(len(all_keys) * 0.8)  # 80% for training

#     train_keys = set(all_keys[:split_index])
#     test_keys = set(all_keys[split_index:])

#     train_graphs = {key: gs[key] for key in train_keys}
#     test_graphs = {key: gs[key] for key in test_keys}

#     # Prepare data for training and testing sets
#     train_data = prepare_data(train_graphs, c, label_mapping)
#     test_data = prepare_data(test_graphs, c, label_mapping)

#     train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
#     test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

#     return train_loader, test_loader


def get_data_loader():
    train_signals = get_train_scored_signals()
    processed_train_signals = process_dataframe(train_signals)
    ided_train_signals = assign_event_ids(processed_train_signals)
    train_cascade, train_no_nodes, train_id_mapping = aggregate_data(ided_train_signals)
    train_gs, results, As, P_dict = get_graphs(
        train_cascade, train_no_nodes, train_id_mapping
    )
    train_f = graph_features(train_gs)
    train_market_features = features_engineer(processed_train_signals)
    train_c = combine_features(train_market_features, train_f)
    train_label_mapping = create_label_mapping(3, "train")
    print("train_data_prepared")

    signals = get_scored_signals()
    processed_signals = process_dataframe(signals)
    ided_signals = assign_event_ids(processed_signals)
    cascade, no_nodes, id_mapping = aggregate_data(ided_signals)
    gs, results, As, P_dict = get_graphs(cascade, no_nodes, id_mapping)
    f = graph_features(gs)
    market_features = features_engineer(processed_signals)
    c = combine_features(market_features, f)
    label_mapping = create_label_mapping(3, "test")
    print("test_data_prepared")





    # First, find the common keys as before
    keys_train_gs = set(train_gs.keys())
    keys_train_c = set(train_c.keys())
    keys_train_label_mapping = set(train_label_mapping.keys())

    # Find the intersection of keys across all dictionaries
    common_keys = keys_train_gs & keys_train_c & keys_train_label_mapping

    # Now create new dictionaries with only the common keys
    new_train_gs = {k: train_gs[k] for k in common_keys}
    new_train_c = {k: train_c[k] for k in common_keys}
    new_train_label_mapping = {k: train_label_mapping[k] for k in common_keys}

    # These new dictionaries now have only the entries with keys present in all three original dictionaries





    # First, find the common keys as before
    tkeys_train_gs = set(gs.keys())
    tkeys_train_c = set(c.keys())
    tkeys_train_label_mapping = set(label_mapping.keys())

    # Find the intersection of keys across all dictionaries
    tcommon_keys = tkeys_train_gs & tkeys_train_c & tkeys_train_label_mapping

    # Now create new dictionaries with only the common keys
    tnew_train_gs = {k: gs[k] for k in tcommon_keys}
    tnew_train_c = {k: c[k] for k in tcommon_keys}
    tnew_train_label_mapping = {k: label_mapping[k] for k in tcommon_keys}

    # These new dictionaries now have only the entries with keys present in all three original dictionaries








    # Prepare data for training and testing sets
    train_data = prepare_data(new_train_gs, new_train_c, new_train_label_mapping)
    test_data = prepare_data(tnew_train_gs, tnew_train_c, tnew_train_label_mapping)

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    a, b = get_data_loader()
