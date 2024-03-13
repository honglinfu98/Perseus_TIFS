import networkx as nx
import numpy as np
from scipy import stats
from os import path
import pickle 
from torch_geometric.loader import DataLoader
from perseus.dataset.compare_paper_graph import combine_features, graph_features
from perseus.dataset.preprocess.train_test_validate import get_test_scored_signals, get_train_scored_signals, get_valid_scored_signals
# from perseus.dataset.preprocess.groudtruth_labeling import create_label_mapping
from perseus.dataset.preprocess.process import features_engineer, get_graphs, process_dataframe, aggregate_data, assign_event_ids
from perseus.dataset.gnn_dataset_preparation import (
    prepare_data,
)
from sklearn.model_selection import train_test_split
from perseus.dataset.preprocess.groudtruth_labeling import create_label_mapping, read_labeling_csv_back_to_dict
from perseus.settings import PROJECT_ROOT
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
# from itertools import combinations
# from sklearn.metrics.pairwise import cosine_similarity

from torch_geometric.utils import to_undirected, is_undirected
import torch
from torch_geometric.data import Data
# from clotho.model.magamaga import get_data_loader


# Function to aggregate and compare centrality measures
def aggregate_and_compare(gs_ls, label_mapping_ls):
    group_0_degrees = []
    group_1_degrees = []
    group_0_betweenness = []
    group_1_betweenness = []

    # Iterate over each network and its corresponding labels
    for key, graph in gs_ls.items():
        degree_centrality = nx.degree_centrality(graph)
        betweenness_centrality = nx.betweenness_centrality(graph)
        labels = label_mapping_ls.get(key, {})

        # Aggregate centrality measures by group
        for node, group in labels.items():
            if group == 0:
                group_0_degrees.append(degree_centrality.get(node, 0))
                group_0_betweenness.append(betweenness_centrality.get(node, 0))
            elif group == 1:
                group_1_degrees.append(degree_centrality.get(node, 0))
                group_1_betweenness.append(betweenness_centrality.get(node, 0))

    # Perform T-tests on the aggregated data
    degree_ttest_result = stats.ttest_ind(group_0_degrees, group_1_degrees, equal_var=False)
    betweenness_ttest_result = stats.ttest_ind(group_0_betweenness, group_1_betweenness, equal_var=False)

    return degree_ttest_result, betweenness_ttest_result


if __name__ == "__main__":
    # Assuming placeholders for signal processing functions
    signals = get_test_scored_signals()  # Placeholder function
    processed_signals = process_dataframe(signals)  # Placeholder function
    ided_signals = assign_event_ids(processed_signals)  # Placeholder function
    cascade, no_nodes, id_mapping, cascade_labeling = aggregate_data(ided_signals)  # Placeholder function
    gs, results, As, P_dict = get_graphs(cascade, no_nodes, id_mapping)  # Placeholder function
    graph_feature = graph_features(gs)  # Placeholder function
    market_feature = features_engineer(processed_signals)  # Placeholder function
    combine_feature = combine_features(market_feature, graph_feature)  # Placeholder function
    label_mapping = read_labeling_csv_back_to_dict("test")  # Placeholder function


    # Running the pooled analysis
    degree_ttest_result, betweenness_ttest_result = aggregate_and_compare(gs, label_mapping)

    # Output results
    print(f"Degree Centrality T-test result: {degree_ttest_result}")
    print(f"Betweenness Centrality T-test result: {betweenness_ttest_result}")



