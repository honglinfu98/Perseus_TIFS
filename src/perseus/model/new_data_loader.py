# from os import path
# import networkx as nx
# from itertools import combinations
# from sklearn.metrics.pairwise import cosine_similarity
# import pickle
# import random
# from torch_geometric.loader import DataLoader
# from clotho.settings import PROJECT_ROOT
# from clotho.dataset.extract.cloudburst_connection import (
#     get_masterminds,
#     get_train_masterminds,
#     get_scored_signals,
#     get_train_scored_signals,
# )

# from clotho.dataset.preprocess.process import (
#     aggregate_data,
#     assign_event_ids,
#     process_dataframe,
#     features_engineer,
#     get_graphs,
# )

# from clotho.dataset.preprocess.groudtruth_labeling import create_label_mapping
# # from clotho.dataset.gnn_dataset_preparation import (
# #     graph_features,
# #     combine_features,
# #     prepare_data,
# # )

# from os import path
# import pickle
# import pandas as pd
# import networkx as nx
# import torch
# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader
# from clotho.dataset.extract.cloudburst_connection import (
#     get_scored_signals,
#     get_train_scored_signals,
# )

# from clotho.settings import PROJECT_ROOT
# from clotho.dataset.preprocess.process import (
#     aggregate_data,
#     assign_event_ids,
#     process_dataframe,
#     features_engineer,
#     get_graphs,
# )

# from clotho.dataset.preprocess.groudtruth_labeling import create_label_mapping
# from itertools import combinations
# from sklearn.metrics.pairwise import cosine_similarity
# from clotho.dataset.extract.cloudburst_connection import get_direct_link
# from itertools import combinations
# from sklearn.metrics.pairwise import cosine_similarity
# def calculate_effsize_efficiency(G, ego):
#     # Get the ego network
#     ego_net = nx.ego_graph(G, ego, undirected=False)
#     # Get alters in the ego network (excluding ego)
#     alters = set(ego_net.nodes()) - {ego}
#     num_alters = len(alters)
#     # Inside your calculate_effsize_efficiency function:
#     avg_degree = 0  # Default to 0
#     if num_alters > 0:
#         avg_degree = (
#             sum(
#                 ego_net.degree(n) - (1 if ego_net.has_edge(n, ego) else 0)
#                 for n in alters
#             )
#             / num_alters
#         )
#     # Calculate effective size
#     eff_size = num_alters - avg_degree
#     # Calculate efficiency
#     efficiency = eff_size / num_alters if num_alters > 0 else 0
#     return eff_size, efficiency

# def out_ego_graph(G, node, radius=1):
#     """
#     Extract the out-ego network of a specified node in a directed graph.
#     """
#     # Extract the out-ego network
#     out_ego = nx.ego_graph(G, node, radius=radius)
#     return out_ego

# def in_ego_graph(G, node, radius=1):
#     """
#     Extract the in-ego network of a specified node in a directed graph.
#     """
#     # Reverse the graph
#     G_reverse = G.reverse(copy=True)
#     # Extract the in-ego network
#     in_ego = nx.ego_graph(G_reverse, node, radius=radius)
#     return in_ego

# def graph_features(gs):
#     dfs = {}
#     # Looping through each key in the scores dictionary
#     for key in gs.keys():
#         # Creating lists to store node_ids, in-ego ratios, and out-ego ratios
#         node_ids = []
#         in_ratios = []
#         out_ratios = []
#         out_nodes = []
#         eff_sizes = []
#         efficiencies = []
#         density = []
#         clustering_coeffs = []
#         for i in gs[key].nodes():
#             node_id = i
#             node_ids.append(node_id)
#             # Calculating the in-ego ratio
#             inn = len(in_ego_graph(gs[key], node_id).nodes) / len(gs[key])
#             in_ratios.append(inn)
#             # Calculating the out-ego ratio
#             outt = len(out_ego_graph(gs[key], node_id).nodes) / len(gs[key])
#             out_ratios.append(outt)
#             out_nodes.append(len(out_ego_graph(gs[key], node_id).nodes))
#             # Calculate the density of the ego network
#             den = nx.density(out_ego_graph(gs[key], node_id))
#             density.append(den)
#             eff_size, efficiency = calculate_effsize_efficiency(gs[key], node_id)
#             eff_sizes.append(eff_size)
#             efficiencies.append(efficiency)
#             # Calculate clustering coefficient for the node
#             clustering_coeff = nx.clustering(gs[key], node_id)
#             clustering_coeffs.append(clustering_coeff)
#         # Creating a DataFrame for each key and storing it in the dfs dictionary
#         dfs[key] = pd.DataFrame(
#             {
#                 "telegram_chat_id": node_ids,
#                 "in_ratio": in_ratios,
#                 "out_ratio": out_ratios,
#                 "out_nodes": out_nodes,
#                 "eff_size": eff_sizes,
#                 "efficiency": efficiencies,
#                 "density": density,
#                 "clustering_coeff": clustering_coeffs,
#             }
#         )
#     return dfs

# def combine_features(market_features, graph_features):
#     combined_data = {}
#     for key in market_features.keys():
#         # Check if the key exists in dfs and 'telegram_chat_id' exists in both DataFrames
#         if (
#             key in graph_features
#             and "telegram_chat_id" in market_features[key].columns
#             and "telegram_chat_id" in graph_features[key].columns
#         ):
#             # Perform an inner join on 'telegram_chat_id'
#             combined_df = pd.merge(
#                 market_features[key],
#                 graph_features[key],
#                 on="telegram_chat_id",
#                 how="inner",
#             )
#         else:
#             # If key is not in dfs or 'telegram_chat_id' is missing in either DataFrame, use an empty DataFrame
#             combined_df = pd.DataFrame()
#         combined_data[key] = combined_df
#     return combined_data

# def prepare_data(graphs, features, label_mapping):

#     prepared_data = []
#     for key, graph in graphs.items():
#         # Extract features for nodes present in the graph
#         features_buffer = features[key]
#         features_buffer = features_buffer[
#             features_buffer["telegram_chat_id"].isin(graph.nodes)
#         ]
#         # Specify feature columns and normalize them
#         feature_columns = [
#             "average_increase_percentage",
#             "number_of_signals",
#             # "average_speed",
#             # "sum_targets_achieved",
#             # "rating",
#             # "in_ratio",
#             # "out_ratio",
#             # "eff_size",
#             # "efficiency",
#         ]
#         features_to_process = features_buffer[feature_columns]
#         normalized_features = (
#             features_to_process - features_to_process.mean()
#         ) / features_to_process.std()
#         nan_columns = normalized_features.columns[
#             normalized_features.isnull().any()
#         ].tolist()
#         if nan_columns:
#             print(f"NaN values detected in key: {key}, Columns: {nan_columns}")
#         node_attributes = torch.tensor(normalized_features.values, dtype=torch.float)
#         # Map node IDs to indices
#         id_to_index = {
#             telegram_chat_id: index
#             for index, telegram_chat_id in enumerate(
#                 features_buffer["telegram_chat_id"]
#             )
#         }
#         # Initialize lists for edge index and weights
#         edge_index_list = []
#         edge_weight_list = []
#         features_df = features_buffer.set_index("telegram_chat_id")
#         # c = features_df
#         features_df = features_df[feature_columns].dropna()
#         # cc = features_df
#         # features_df = features_df.dropna(inplace=True)
#         # ccc = features_df
#         # Calculate cosine similarity for each pair of rows
#         print(key)
#         similarities = {}
#         for chat_id_1, chat_id_2 in combinations(features_df.index, 2):
#             fi = features_df.loc[chat_id_1].values.reshape(1, -1)
#             fj = features_df.loc[chat_id_2].values.reshape(1, -1)
#             similarity = cosine_similarity(fi, fj)[0][0]
#             similarities[(chat_id_1, chat_id_2)] = similarity
#         # Populate edge index and weights
#         for (source_node, target_node), weight in similarities.items():
#             source = id_to_index.get(source_node)
#             target = id_to_index.get(target_node)
#             if source is not None and target is not None:  # Ensure both nodes are in the id_to_index mapping
#                 edge_index_list.append([source, target])
#                 edge_weight_list.append(weight)
#         # Convert lists to PyTorch tensors
#         edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
#         edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)
#         # Populate the lists
#         num_labels = 3
#         # # Prepare labels
#         # labels = [
#         #     label_mapping[key].get(node_id, 0)
#         #     for node_id in features_buffer["telegram_chat_id"]
#         # ]
#         # labels = torch.tensor(labels, dtype=torch.long)
#         # # Create a Data object
#         # data = Data(x=node_attributes, edge_index=edge_index, y=labels)
#         # Prepare labels
#         labels = []
#         for node_id in features_buffer["telegram_chat_id"]:
#             node_labels = label_mapping[key].get(node_id, 0)
#             if not isinstance(node_labels, list):
#                 node_labels = [node_labels]  # Convert to list for consistency
#             # Convert to one-hot encoded format
#             label_vector = [0] * num_labels
#             for label in node_labels:
#                 if label < num_labels:
#                     label_vector[label] = 1
#             labels.append(label_vector)
#         # Convert list of labels to a tensor
#         labels_tensor = torch.tensor(labels, dtype=torch.float)
#         # Create a Data object
#         data = Data(x=node_attributes, edge_index=edge_index, edge_weight=edge_weight , y=labels_tensor)
#         prepared_data.append(data)
#     # prepared_data = []
#     # for key, graph in graphs.items():
#     #     # Extract features for nodes present in the graph
#     #     features_buffer = features[key]
#     #     features_buffer = features_buffer[
#     #         features_buffer["telegram_chat_id"].isin(graph.nodes)
#     #     ]
#     #     # Specify feature columns and normalize them
#     #     feature_columns = [
#     #         "average_increase_percentage",
#     #         "number_of_signals",
#     #         # "average_speed",
#     #         # "sum_targets_achieved",
#     #         # "rating",
#     #         # "in_ratio",
#     #         # "out_ratio",
#     #         # "eff_size",
#     #         # "efficiency",
#     #     ]
#     #     features_to_process = features_buffer[feature_columns]
#     #     normalized_features = (
#     #         features_to_process - features_to_process.mean()
#     #     ) / features_to_process.std()
#     #     nan_columns = normalized_features.columns[
#     #         normalized_features.isnull().any()
#     #     ].tolist()
#     #     if nan_columns:
#     #         print(f"NaN values detected in key: {key}, Columns: {nan_columns}")
#     #     node_attributes = torch.tensor(normalized_features.values, dtype=torch.float)
#     #     # Map node IDs to indices
#     #     id_to_index = {
#     #         telegram_chat_id: index
#     #         for index, telegram_chat_id in enumerate(
#     #             features_buffer["telegram_chat_id"]
#     #         )
#     #     }
#     #     # Initialize lists for edge index and weights
#     #     edge_index_list = []
#     #     edge_weight_list = []
#     #     features_df = features_buffer.set_index("telegram_chat_id")
#     #     features_df = features_df[feature_columns]
#     #     # Calculate cosine similarity for each pair of rows
#     #     print(key)
#     #     similarities = {}
#     #     for chat_id_1, chat_id_2 in combinations(features_df.index, 2):
#     #         fi = features_df.loc[chat_id_1].values.reshape(1, -1)
#     #         fj = features_df.loc[chat_id_2].values.reshape(1, -1)
#     #         similarity = cosine_similarity(fi, fj)[0][0]
#     #         similarities[(chat_id_1, chat_id_2)] = similarity
#     #     # Populate edge index and weights
#     #     for (source_node, target_node), weight in similarities.items():
#     #         source = id_to_index.get(source_node)
#     #         target = id_to_index.get(target_node)
#     #         if source is not None and target is not None:  # Ensure both nodes are in the id_to_index mapping
#     #             edge_index_list.append([source, target])
#     #             edge_weight_list.append(weight)
#     #     # Convert lists to PyTorch tensors
#     #     edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
#     #     edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)
#     #     # Populate the lists
#     #     num_labels = 3
#     #     # # Prepare labels
#     #     # labels = [
#     #     #     label_mapping[key].get(node_id, 0)
#     #     #     for node_id in features_buffer["telegram_chat_id"]
#     #     # ]
#     #     # labels = torch.tensor(labels, dtype=torch.long)
#     #     # # Create a Data object
#     #     # data = Data(x=node_attributes, edge_index=edge_index, y=labels)
#     #     # Prepare labels
#     #     labels = []
#     #     for node_id in features_buffer["telegram_chat_id"]:
#     #         node_labels = label_mapping[key].get(node_id, 0)
#     #         if not isinstance(node_labels, list):
#     #             node_labels = [node_labels]  # Convert to list for consistency
#     #         # Convert to one-hot encoded format
#     #         label_vector = [0] * num_labels
#     #         for label in node_labels:
#     #             if label < num_labels:
#     #                 label_vector[label] = 1
#     #         labels.append(label_vector)
#     #     # Convert list of labels to a tensor
#     #     labels_tensor = torch.tensor(labels, dtype=torch.float)
#     #     # Create a Data object
#     #     data = Data(x=node_attributes, edge_index=edge_index, edge_weight=edge_weight , y=labels_tensor)
#     #     prepared_data.append(data)
#     return prepared_data

# def get_new_data_loader():
#     train_signals = get_train_scored_signals()
#     processed_train_signals = process_dataframe(train_signals)
#     ided_train_signals = assign_event_ids(processed_train_signals)
#     train_cascade, train_no_nodes, train_id_mapping = aggregate_data(ided_train_signals)
#     train_gs, results, As, P_dict = get_graphs(
#         train_cascade, train_no_nodes, train_id_mapping
#     )
#     train_f = graph_features(train_gs)
#     train_market_features = features_engineer(processed_train_signals)
#     train_c = combine_features(train_market_features, train_f)
#     # train_ngs = get_similar_graph(train_c)
#     train_label_mapping = create_label_mapping(3, "train")
#     print("train_data_prepared")

#     signals = get_scored_signals()
#     processed_signals = process_dataframe(signals)
#     ided_signals = assign_event_ids(processed_signals)
#     cascade, no_nodes, id_mapping = aggregate_data(ided_signals)
#     gs, results, As, P_dict = get_graphs(cascade, no_nodes, id_mapping)
#     f = graph_features(gs)
#     market_features = features_engineer(processed_signals)
#     c = combine_features(market_features, f)
#     # ngs = get_similar_graph(c)
#     label_mapping = create_label_mapping(3, "test")

#     # Prepare data for training and testing sets
#     train_data = prepare_data(train_gs, train_market_features, train_label_mapping)
#     test_data = prepare_data(gs, market_features, label_mapping)

#     train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
#     test_loader = DataLoader(test_data, batch_size=1, shuffle=False)


#     return train_loader, test_loader


# if __name__ == "__main__":
#     a, b = get_new_data_loader()


from os import path
import networkx as nx
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import random
from torch_geometric.loader import DataLoader
from perseus.settings import PROJECT_ROOT
from perseus.dataset.extract.cloudburst_connection import (
    get_masterminds,
    get_train_masterminds,
    get_scored_signals,
    get_train_scored_signals,
)

from perseus.dataset.preprocess.process import (
    aggregate_data,
    assign_event_ids,
    process_dataframe,
    features_engineer,
    get_graphs,
)

from perseus.dataset.preprocess.groudtruth_labeling import create_label_mapping

# from clotho.dataset.gnn_dataset_preparation import (
#     graph_features,
#     combine_features,
#     prepare_data,
# )

from os import path
import pickle
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from perseus.dataset.extract.cloudburst_connection import (
    get_scored_signals,
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
from perseus.dataset.extract.cloudburst_connection import get_direct_link
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity

from torch_geometric.utils import to_undirected, is_undirected


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


def graph_features(gs):
    dfs = {}
    # Looping through each key in the scores dictionary
    for key in gs.keys():
        # Creating lists to store node_ids, in-ego ratios, and out-ego ratios
        node_ids = []
        in_ratios = []
        out_ratios = []
        out_nodes = []
        eff_sizes = []
        efficiencies = []
        density = []
        clustering_coeffs = []
        for i in gs[key].nodes():
            node_id = i
            node_ids.append(node_id)
            # Calculating the in-ego ratio
            inn = len(in_ego_graph(gs[key], node_id).nodes) / len(gs[key])
            in_ratios.append(inn)
            # Calculating the out-ego ratio
            outt = len(out_ego_graph(gs[key], node_id).nodes) / len(gs[key])
            out_ratios.append(outt)
            out_nodes.append(len(out_ego_graph(gs[key], node_id).nodes))
            # Calculate the density of the ego network
            den = nx.density(out_ego_graph(gs[key], node_id))
            density.append(den)
            eff_size, efficiency = calculate_effsize_efficiency(gs[key], node_id)
            eff_sizes.append(eff_size)
            efficiencies.append(efficiency)
            # Calculate clustering coefficient for the node
            clustering_coeff = nx.clustering(gs[key], node_id)
            clustering_coeffs.append(clustering_coeff)
        # Creating a DataFrame for each key and storing it in the dfs dictionary
        dfs[key] = pd.DataFrame(
            {
                "telegram_chat_id": node_ids,
                "in_ratio": in_ratios,
                "out_ratio": out_ratios,
                "out_nodes": out_nodes,
                "eff_size": eff_sizes,
                "efficiency": efficiencies,
                "density": density,
                "clustering_coeff": clustering_coeffs,
            }
        )
    return dfs


def combine_features(market_features, graph_features):
    combined_data = {}
    for key in market_features.keys():
        # Check if the key exists in dfs and 'telegram_chat_id' exists in both DataFrames
        if (
            key in graph_features
            and "telegram_chat_id" in market_features[key].columns
            and "telegram_chat_id" in graph_features[key].columns
        ):
            # Perform an inner join on 'telegram_chat_id'
            combined_df = pd.merge(
                market_features[key],
                graph_features[key],
                on="telegram_chat_id",
                how="inner",
            )
        else:
            # If key is not in dfs or 'telegram_chat_id' is missing in either DataFrame, use an empty DataFrame
            combined_df = pd.DataFrame()
        combined_data[key] = combined_df
    return combined_data


def prepare_data(graphs, features, label_mapping):
    prepared_data = []
    for key, graph in graphs.items():
        # Extract features for nodes present in the graph
        features_buffer = features[key]
        features_buffer = features_buffer[
            features_buffer["telegram_chat_id"].isin(graph.nodes)
        ]
        # Specify feature columns and normalize them
        feature_columns = [
            "average_increase_percentage",
            "number_of_signals",
            # "average_speed",
            # "sum_targets_achieved",
            # "rating",
            # "in_ratio",
            # "out_ratio",
            # "eff_size",
            # "efficiency",
        ]
        features_to_process = features_buffer[feature_columns]
        normalized_features = (
            features_to_process - features_to_process.mean()
        ) / features_to_process.std()
        nan_columns = normalized_features.columns[
            normalized_features.isnull().any()
        ].tolist()
        if nan_columns:
            print(f"NaN values detected in key: {key}, Columns: {nan_columns}")
        node_attributes = torch.tensor(normalized_features.values, dtype=torch.float)
        # Map node IDs to indices
        id_to_index = {
            telegram_chat_id: index
            for index, telegram_chat_id in enumerate(
                features_buffer["telegram_chat_id"]
            )
        }
        # Initialize lists for edge index and weights
        edge_index_list = []
        edge_weight_list = []
        features_df = features_buffer.set_index("telegram_chat_id")
        # c = features_df
        features_df = features_df[feature_columns].dropna()
        # cc = features_df
        # features_df = features_df.dropna(inplace=True)
        # ccc = features_df
        # Calculate cosine similarity for each pair of rows
        print(key)
        similarities = {}
        for chat_id_1, chat_id_2 in combinations(features_df.index, 2):
            fi = features_df.loc[chat_id_1].values.reshape(1, -1)
            fj = features_df.loc[chat_id_2].values.reshape(1, -1)
            similarity = cosine_similarity(fi, fj)[0][0]
            similarities[(chat_id_1, chat_id_2)] = similarity
        # Populate edge index and weights
        for (source_node, target_node), weight in similarities.items():
            source = id_to_index.get(source_node)
            target = id_to_index.get(target_node)
            if (
                source is not None and target is not None
            ):  # Ensure both nodes are in the id_to_index mapping
                edge_index_list.append([source, target])
                edge_weight_list.append(weight)
        # Convert lists to PyTorch tensors
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)
        # Populate the lists
        num_labels = 3
        # # Prepare labels
        # labels = [
        #     label_mapping[key].get(node_id, 0)
        #     for node_id in features_buffer["telegram_chat_id"]
        # ]
        # labels = torch.tensor(labels, dtype=torch.long)
        # # Create a Data object
        # data = Data(x=node_attributes, edge_index=edge_index, y=labels)
        # Prepare labels
        labels = []
        for node_id in features_buffer["telegram_chat_id"]:
            node_labels = label_mapping[key].get(node_id, 0)
            if not isinstance(node_labels, list):
                node_labels = [node_labels]  # Convert to list for consistency
            # Convert to one-hot encoded format
            label_vector = [0] * num_labels
            for label in node_labels:
                if label < num_labels:
                    label_vector[label] = 1
            labels.append(label_vector)
        # Convert list of labels to a tensor
        labels_tensor = torch.tensor(labels, dtype=torch.float)
        # Create a Data object
        data = Data(
            x=node_attributes,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=labels_tensor,
        )

        undirect_edge_index = to_undirected(data.edge_index, num_nodes=data.x.size(0))

        # Create a Data object
        data = Data(
            x=node_attributes,
            edge_index=undirect_edge_index,
            edge_weight=edge_weight,
            y=labels_tensor,
        )

        prepared_data.append(data)
    # prepared_data = []
    # for key, graph in graphs.items():
    #     # Extract features for nodes present in the graph
    #     features_buffer = features[key]
    #     features_buffer = features_buffer[
    #         features_buffer["telegram_chat_id"].isin(graph.nodes)
    #     ]
    #     # Specify feature columns and normalize them
    #     feature_columns = [
    #         "average_increase_percentage",
    #         "number_of_signals",
    #         # "average_speed",
    #         # "sum_targets_achieved",
    #         # "rating",
    #         # "in_ratio",
    #         # "out_ratio",
    #         # "eff_size",
    #         # "efficiency",
    #     ]
    #     features_to_process = features_buffer[feature_columns]
    #     normalized_features = (
    #         features_to_process - features_to_process.mean()
    #     ) / features_to_process.std()
    #     nan_columns = normalized_features.columns[
    #         normalized_features.isnull().any()
    #     ].tolist()
    #     if nan_columns:
    #         print(f"NaN values detected in key: {key}, Columns: {nan_columns}")
    #     node_attributes = torch.tensor(normalized_features.values, dtype=torch.float)
    #     # Map node IDs to indices
    #     id_to_index = {
    #         telegram_chat_id: index
    #         for index, telegram_chat_id in enumerate(
    #             features_buffer["telegram_chat_id"]
    #         )
    #     }
    #     # Initialize lists for edge index and weights
    #     edge_index_list = []
    #     edge_weight_list = []
    #     features_df = features_buffer.set_index("telegram_chat_id")
    #     features_df = features_df[feature_columns]
    #     # Calculate cosine similarity for each pair of rows
    #     print(key)
    #     similarities = {}
    #     for chat_id_1, chat_id_2 in combinations(features_df.index, 2):
    #         fi = features_df.loc[chat_id_1].values.reshape(1, -1)
    #         fj = features_df.loc[chat_id_2].values.reshape(1, -1)
    #         similarity = cosine_similarity(fi, fj)[0][0]
    #         similarities[(chat_id_1, chat_id_2)] = similarity
    #     # Populate edge index and weights
    #     for (source_node, target_node), weight in similarities.items():
    #         source = id_to_index.get(source_node)
    #         target = id_to_index.get(target_node)
    #         if source is not None and target is not None:  # Ensure both nodes are in the id_to_index mapping
    #             edge_index_list.append([source, target])
    #             edge_weight_list.append(weight)
    #     # Convert lists to PyTorch tensors
    #     edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    #     edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)
    #     # Populate the lists
    #     num_labels = 3
    #     # # Prepare labels
    #     # labels = [
    #     #     label_mapping[key].get(node_id, 0)
    #     #     for node_id in features_buffer["telegram_chat_id"]
    #     # ]
    #     # labels = torch.tensor(labels, dtype=torch.long)
    #     # # Create a Data object
    #     # data = Data(x=node_attributes, edge_index=edge_index, y=labels)
    #     # Prepare labels
    #     labels = []
    #     for node_id in features_buffer["telegram_chat_id"]:
    #         node_labels = label_mapping[key].get(node_id, 0)
    #         if not isinstance(node_labels, list):
    #             node_labels = [node_labels]  # Convert to list for consistency
    #         # Convert to one-hot encoded format
    #         label_vector = [0] * num_labels
    #         for label in node_labels:
    #             if label < num_labels:
    #                 label_vector[label] = 1
    #         labels.append(label_vector)
    #     # Convert list of labels to a tensor
    #     labels_tensor = torch.tensor(labels, dtype=torch.float)
    #     # Create a Data object
    #     data = Data(x=node_attributes, edge_index=edge_index, edge_weight=edge_weight , y=labels_tensor)
    #     prepared_data.append(data)
    return prepared_data


def get_new_data_loader():
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
    # train_ngs = get_similar_graph(train_c)
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
    # ngs = get_similar_graph(c)
    label_mapping = create_label_mapping(3, "test")
    print("train_data_prepared")


    

    # Prepare data for training and testing sets
    train_data = prepare_data(train_gs, train_market_features, train_label_mapping)
    test_data = prepare_data(gs, market_features, label_mapping)

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    a, b = get_new_data_loader()
