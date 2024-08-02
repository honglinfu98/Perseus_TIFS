"""
This script is used to compare the nodes and edges yearly 
"""

from os import path
import pickle
from perseus.dataset.dataset_preparation import combine_features, graph_features
from perseus.dataset.preprocess.process import (
    aggregate_data,
    assign_event_ids,
    features_engineer,
    process_dataframe,
    get_graphs,
)
from perseus.settings import PROJECT_ROOT


datasets = []
for year in range(2018, 2025):
    with open(path.join(PROJECT_ROOT, "data", f"{year}_signals.pkl"), "rb") as file:
        datasets.append(pickle.load(file))


gs_ls = []
features_ls = []
market_features_ls = []
P_dicts_ls = []

for dataset in datasets:
    processed_signals = process_dataframe(dataset)
    ided_signals = assign_event_ids(processed_signals)
    cascade, no_nodes, id_mapping, cascade_labeling = aggregate_data(ided_signals)
    gs, results, As, P_dict = get_graphs(cascade, no_nodes, id_mapping)
    graph_feature = graph_features(gs)
    market_feature = features_engineer(processed_signals)
    combine_feature = combine_features(market_feature, graph_feature)
    gs_ls.append(gs)
    features_ls.append(combine_feature)
    market_features_ls.append(market_feature)
    P_dicts_ls.append(P_dict)
