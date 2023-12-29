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

with open(path.join(PROJECT_ROOT, "data", "signals.pkl"), "rb") as file:
    signals = pickle.load(file)


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

    # signals = get_scored_signals()
    processed_signals = process_dataframe(signals)
    ided_signals = assign_event_ids(processed_signals)
    cascade, no_nodes, id_mapping = aggregate_data(ided_signals)
    gs, results, As, P_dict = get_graphs(cascade, no_nodes, id_mapping)
    f = graph_features(gs)
    market_features = features_engineer(processed_signals)
    c = combine_features(market_features, f)
    label_mapping = create_label_mapping(3, "test")

    # Prepare data for training and testing sets
    train_data = prepare_data(train_gs, train_c, train_label_mapping)
    test_data = prepare_data(gs, c, label_mapping)

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    a, b = get_data_loader()
