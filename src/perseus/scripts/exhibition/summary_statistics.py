import numpy as np
from perseus.dataset.preprocess.train_test_validate import (
    get_test_scored_signals,
    get_train_scored_signals,
    get_valid_scored_signals,
)
from perseus.dataset.preprocess.process import (
    compute_weighted_graph_features,
    features_engineer,
    get_graphs,
    process_dataframe,
    aggregate_data,
    assign_event_ids,
    graph_features,
    combine_features,
)
from perseus.dataset.preprocess.groudtruth_labeling import (
    read_labeling_csv_back_to_dict,
)
from perseus.dataset.dataset_preparation import get_split_data_pickle_wn


train_signals = get_train_scored_signals()
validate_signals = get_valid_scored_signals()
test_signals = get_test_scored_signals()

datasets = [train_signals, validate_signals, test_signals]
gs_ls = []
features_ls = []
market_features_ls = []
P_dicts_ls = []
label_mapping_ls = []
P_com_ls = []
cascade_ls = []

for dataset in datasets:
    processed_signals = process_dataframe(dataset)
    ided_signals = assign_event_ids(processed_signals)
    cascade, no_nodes, id_mapping, cascade_labeling = aggregate_data(ided_signals)
    gs, results, As, P_dict, P_com = get_graphs(cascade, no_nodes, id_mapping)
    graph_feature = graph_features(gs)
    market_feature = features_engineer(processed_signals)
    weighted_feature = compute_weighted_graph_features(P_com)

    combine_feature = combine_features(market_feature, graph_feature, weighted_feature)

    gs_ls.append(gs)
    features_ls.append(combine_feature)
    market_features_ls.append(market_feature)
    P_dicts_ls.append(P_dict)
    P_com_ls.append(P_com)
    cascade_ls.append(cascade)

label_mapping_ls = [
    read_labeling_csv_back_to_dict("train"),
    read_labeling_csv_back_to_dict("valid"),
    read_labeling_csv_back_to_dict("test"),
]

# Create a common key set for gs_ls and label_mapping_ls, and filter out the keys that are not common
for i in range(3):
    common_keys = set(gs_ls[i].keys()).intersection(set(label_mapping_ls[i].keys()))
    gs_ls[i] = {k: v for k, v in gs_ls[i].items() if k in common_keys}
    label_mapping_ls[i] = {
        k: v for k, v in label_mapping_ls[i].items() if k in common_keys
    }


direct = get_split_data_pickle_wn("DDINA")
weighted = get_split_data_pickle_wn("DDM")

number_of_cryptocurrency = [len(cascade_ls[0]), len(cascade_ls[1]), len(cascade_ls[2])]
crowd_pumps = [len(datasets[0]), len(datasets[1]), len(datasets[2])]
crowd_pumps_event = [
    sum([len(v) for k, v in cascade_ls[0].items()]),
    sum([len(v) for k, v in cascade_ls[1].items()]),
    sum([len(v) for k, v in cascade_ls[2].items()]),
]
mastermind = [
    sum([sum(np.array(i[0].y[:, 0])) for i in direct[0]]),
    sum([sum(np.array(i[0].y[:, 0])) for i in direct[1]]),
    sum([sum(np.array(i[0].y[:, 0])) for i in direct[2]]),
]

accomplices = [
    sum([len(np.array(i[0].y[:, 0])) - sum(np.array(i[0].y[:, 0])) for i in direct[0]]),
    sum([len(np.array(i[0].y[:, 0])) - sum(np.array(i[0].y[:, 0])) for i in direct[1]]),
    sum([len(np.array(i[0].y[:, 0])) - sum(np.array(i[0].y[:, 0])) for i in direct[2]]),
]

number_of_graphs = [len(gs_ls[0]), len(gs_ls[1]), len(gs_ls[2])]

number_of_nodes = [
    sum([len(v.nodes()) for k, v in gs_ls[0].items()]),
    sum([len(v.nodes()) for k, v in gs_ls[1].items()]),
    sum([len(v.nodes()) for k, v in gs_ls[2].items()]),
]
number_of_edges_directed = [
    sum([len(np.array(i[0].edge_index[0, :])) for i in direct[0]]),
    sum([len(np.array(i[0].edge_index[0, :])) for i in direct[1]]),
    sum([len(np.array(i[0].edge_index[0, :])) for i in direct[2]]),
]
number_of_edges_weighted = [
    sum([len(np.array(i[0].edge_index[0, :])) for i in weighted[0]]),
    sum([len(np.array(i[0].edge_index[0, :])) for i in weighted[1]]),
    sum([len(np.array(i[0].edge_index[0, :])) for i in weighted[2]]),
]
print(f"number_of_cryptocurrency" + str(number_of_cryptocurrency))
print(f"crowd_pumps" + str(crowd_pumps))
print(f"crowd_pumps_event" + str(crowd_pumps_event))
print(f"mastermind" + str(mastermind))
print(f"accomplices" + str(accomplices))
print(f"number_of_graphs" + str(number_of_graphs))
print(f"number_of_nodes" + str(number_of_nodes))
print(f"number_of_edges_directed" + str(number_of_edges_directed))
print(f"number_of_edges_weighted" + str(number_of_edges_weighted))
