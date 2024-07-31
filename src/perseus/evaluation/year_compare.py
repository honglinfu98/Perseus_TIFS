from os import path
import pickle
from perseus.model.magamaga import combine_features, graph_features
from perseus.dataset.preprocess.process import (
    aggregate_data,
    assign_event_ids,
    features_engineer,
    process_dataframe,
    get_graphs,
)
from perseus.settings import PROJECT_ROOT


# load all the data from 2018 to 2024 from PROJECT "data" whose fileanme is 2018_signals.pkl, 2019_signals.pkl, 2020_signals.pkl, 2021_signals.pkl, 2022_signals.pkl, 2023_signals.pkl, 2024_signals.pkl into datasets
# just like the code below
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
