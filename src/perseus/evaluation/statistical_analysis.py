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

from os import path
import pickle
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from perseus.dataset.preprocess.train_test_validate import (
    get_test_scored_signals,
    get_train_scored_signals,
)
from perseus.settings import PROJECT_ROOT
from perseus.dataset.preprocess.process import (
    aggregate_data,
    assign_event_ids,
    process_dataframe,
    features_engineer,
    get_graphs,
)
from perseus.dataset.preprocess.groudtruth_labeling import create_label_mapping
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity


def calculate_effsize_efficiency(G, ego):
    # Get the ego network
    ego_net = nx.ego_graph(G, ego, undirected=False)

    # Get alters in the ego network (excluding ego)
    alters = set(ego_net.nodes()) - {ego}
    num_alters = len(alters)
    # Inside your calculate_effsize_efficiency function:
    avg_degree = 0  # Default to 0
    if num_alters > 0:
        avg_degree = (
            sum(
                ego_net.degree(n) - (1 if ego_net.has_edge(n, ego) else 0)
                for n in alters
            )
            / num_alters
        )

    # Calculate effective size
    eff_size = num_alters - avg_degree

    # Calculate efficiency
    efficiency = eff_size / num_alters if num_alters > 0 else 0

    return eff_size, efficiency


def out_ego_graph(G, node, radius=1):
    """
    Extract the out-ego network of a specified node in a directed graph.
    """
    # Extract the out-ego network
    out_ego = nx.ego_graph(G, node, radius=radius)
    return out_ego


def in_ego_graph(G, node, radius=1):
    """
    Extract the in-ego network of a specified node in a directed graph.
    """
    # Reverse the graph
    G_reverse = G.reverse(copy=True)
    # Extract the in-ego network
    in_ego = nx.ego_graph(G_reverse, node, radius=radius)
    return in_ego


# def aggregate_and_compare(gs_ls, label_mapping_ls):
#     # Initialize containers for each centrality measure by group
#     group_0_degrees, group_1_degrees = [], []
#     group_0_betweenness, group_1_betweenness = [], []
#     group_0_closeness, group_1_closeness = [], []
#     group_0_eigenvector, group_1_eigenvector = [], []
#     group_0_pagerank, group_1_pagerank = [], []

#     # Iterate over each network and its corresponding labels
#     for key, graph in gs_ls.items():
#         # Calculate centrality measures
#         degree_centrality = nx.degree_centrality(graph)
#         betweenness_centrality = nx.betweenness_centrality(graph)
#         closeness_centrality = nx.closeness_centrality(graph)
#         # eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=1000)
#         pagerank = nx.pagerank(graph)

#         labels = label_mapping_ls.get(key, {})

#         # Aggregate centrality measures by group
#         for node, group in labels.items():
#             if group == 0:
#                 group_0_degrees.append(degree_centrality.get(node, 0))
#                 group_0_betweenness.append(betweenness_centrality.get(node, 0))
#                 group_0_closeness.append(closeness_centrality.get(node, 0))
#                 # group_0_eigenvector.append(eigenvector_centrality.get(node, 0))
#                 group_0_pagerank.append(pagerank.get(node, 0))
#             elif group == 1:
#                 group_1_degrees.append(degree_centrality.get(node, 0))
#                 group_1_betweenness.append(betweenness_centrality.get(node, 0))
#                 group_1_closeness.append(closeness_centrality.get(node, 0))
#                 # group_1_eigenvector.append(eigenvector_centrality.get(node, 0))
#                 group_1_pagerank.append(pagerank.get(node, 0))

#     # Perform T-tests on the aggregated data for each centrality measure
#     results = {}
#     results['degree'] = stats.ttest_ind(group_0_degrees, group_1_degrees, equal_var=False)
#     results['betweenness'] = stats.ttest_ind(group_0_betweenness, group_1_betweenness, equal_var=False)
#     results['closeness'] = stats.ttest_ind(group_0_closeness, group_1_closeness, equal_var=False)
#     # results['eigenvector'] = stats.ttest_ind(group_0_eigenvector, group_1_eigenvector, equal_var=False)
#     results['pagerank'] = stats.ttest_ind(group_0_pagerank, group_1_pagerank, equal_var=False)

#     return results
# from scipy import stats

def aggregate_and_compare_combined(gs_ls, label_mapping_ls):
    # Initialize containers for centrality measures and additional features by group
    metrics_by_group = {
        "degree_centrality": [[], []],
        "betweenness_centrality": [[], []],
        "closeness_centrality": [[], []],
        "pagerank": [[], []],
        "in_ratio": [[], []],
        "out_ratio": [[], []],
        "out_nodes": [[], []],
        "eff_size": [[], []],
        "efficiency": [[], []],
        "density": [[], []],
        "clustering_coeff": [[], []],
    }

    # Iterate over each network and its corresponding labels
    for key, graph in gs_ls.items():
        labels = label_mapping_ls.get(key, {})

        # Calculate centrality measures for each node
        degree_centrality = nx.degree_centrality(graph)
        betweenness_centrality = nx.betweenness_centrality(graph)
        closeness_centrality = nx.closeness_centrality(graph)
        pagerank = nx.pagerank(graph)

        for node in graph.nodes():
            group = labels.get(node)
            if group is not None:
                # Aggregate centrality measures
                metrics_by_group["degree_centrality"][group].append(degree_centrality.get(node, 0))
                metrics_by_group["betweenness_centrality"][group].append(betweenness_centrality.get(node, 0))
                metrics_by_group["closeness_centrality"][group].append(closeness_centrality.get(node, 0))
                metrics_by_group["pagerank"][group].append(pagerank.get(node, 0))

                # Calculate and aggregate additional graph features
                in_ego = in_ego_graph(graph, node)
                out_ego = out_ego_graph(graph, node)
                in_ratio = len(in_ego.nodes) / len(graph.nodes)
                out_ratio = len(out_ego.nodes) / len(graph.nodes)
                eff_size, efficiency = calculate_effsize_efficiency(graph, node)
                density = nx.density(out_ego)
                clustering_coeff = nx.clustering(graph, node)

                metrics_by_group["in_ratio"][group].append(in_ratio)
                metrics_by_group["out_ratio"][group].append(out_ratio)
                metrics_by_group["out_nodes"][group].append(len(out_ego.nodes))
                metrics_by_group["eff_size"][group].append(eff_size)
                metrics_by_group["efficiency"][group].append(efficiency)
                metrics_by_group["density"][group].append(density)
                metrics_by_group["clustering_coeff"][group].append(clustering_coeff)

    # Perform T-tests on the aggregated data for each metric
    ttest_results = {}
    for metric, groups in metrics_by_group.items():
        ttest_results[metric] = stats.ttest_ind(groups[0], groups[1], equal_var=False, nan_policy='omit')

    return ttest_results


# Function to process signals and return necessary components for analysis
def process_signals_and_get_components(signal_function):
    signals = signal_function()  # Get signals using the provided function
    processed_signals = process_dataframe(signals)  # Process signals
    ided_signals = assign_event_ids(processed_signals)  # Assign event IDs
    cascade, no_nodes, id_mapping, cascade_labeling = aggregate_data(ided_signals)  # Aggregate data
    gs, results, As, P_dict = get_graphs(cascade, no_nodes, id_mapping)  # Get graphs
    graph_feature = graph_features(gs)  # Get graph features
    market_feature = features_engineer(processed_signals)  # Engineer market features
    combine_feature = combine_features(market_feature, graph_feature)  # Combine features
    label_mapping = read_labeling_csv_back_to_dict("test")  # Read label mapping (adjust as needed)
    return gs, label_mapping



# Define a function to assign significance levels
def assign_significance(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''

# Main execution block
if __name__ == "__main__":
    # Process each dataset
    gs_train, label_mapping_train = process_signals_and_get_components(get_train_scored_signals)
    gs_valid, label_mapping_valid = process_signals_and_get_components(get_valid_scored_signals)
    gs_test, label_mapping_test = process_signals_and_get_components(get_test_scored_signals)

    # Combine all graphs and label mappings
    all_gs = {**gs_train, **gs_valid, **gs_test}
    all_label_mappings = {**label_mapping_train, **label_mapping_valid, **label_mapping_test}

    # Running the pooled analysis
    results = aggregate_and_compare_combined(all_gs, all_label_mappings)


    # Transform the t-test results into a format suitable for DataFrame construction
    data = {
        'Metric': [],
        'Statistic': [],
        'P-value': [],
        'Degrees of Freedom': [],
        'Significance': []  # New column for significance marks
    }

    for metric, result in results.items():
        data['Metric'].append(metric)
        data['Statistic'].append(result.statistic)
        data['P-value'].append(f"{result.pvalue:.3g}")  # Format p-value for readability
        # Include degrees of freedom if available
        data['Degrees of Freedom'].append(getattr(result, 'df', None))  # Use getattr for compatibility
        # Assign significance level
        data['Significance'].append(assign_significance(result.pvalue))

    # Create a DataFrame
    df_results = pd.DataFrame(data)

    # Optionally, set the Metric column as the index for better presentation
    df_results.set_index('Metric', inplace=True)

    # Show or return the DataFrame
    print(df_results)