from os import path
import pickle
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from clotho.dataset.extract.cloudburst_connection import (
    get_scored_signals,
    get_train_scored_signals,
)
from clotho.settings import PROJECT_ROOT
from clotho.dataset.preprocess.process import (
    aggregate_data,
    assign_event_ids,
    process_dataframe,
    features_engineer,
    get_graphs,
)
from clotho.dataset.preprocess.groudtruth_labeling import create_label_mapping
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from clotho.dataset.extract.cloudburst_connection import get_direct_link
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


def prepare_undirect_data(graphs, features, label_mapping):
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
            "in_ratio",
            "out_ratio",
            # "eff_size",
            # "efficiency",
        ]
        features_to_normalize = features_buffer[feature_columns]
        normalized_features = (
            features_to_normalize - features_to_normalize.mean()
        ) / features_to_normalize.std()

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

        # # Map node IDs to indices
        # id_to_index = {node_id: index for index, node_id in enumerate(features_buffer["telegram_chat_id"])}

        # Initialize lists for edge index and weights
        edge_index_list = []
        edge_weight_list = []

        features_df = features_buffer.set_index("telegram_chat_id")

        # Calculate cosine similarity for each pair of rows
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

        prepared_data.append(data)

    return prepared_data


# signals = get_scored_signals()
# with open(path.join(PROJECT_ROOT, "data", "signals.pkl"), "rb") as file:
#     signals = pickle.load(file)
# signals = get_scored_signals()


if __name__ == "__main__":
    signals = get_train_scored_signals()

    processed_signals = process_dataframe(signals)
    ided_signals = assign_event_ids(processed_signals)

    cascade, no_nodes, id_mapping = aggregate_data(ided_signals)
    gs, results, As, P_dict = get_graphs(cascade, no_nodes, id_mapping)
    f = graph_features(gs)
    market_features = features_engineer(processed_signals)
    c = combine_features(market_features, f)
    label_mapping = create_label_mapping(3, "train")
    data = prepare_undirect_data(gs, c, label_mapping)
    loader = DataLoader(data, batch_size=1, shuffle=True)


# def calculate_all_similarities(features_df):
#     # Ensure 'telegram_chat_id' is present
#     if "telegram_chat_id" not in features_df.columns:
#         raise KeyError("Column 'telegram_chat_id' not found in DataFrame")

#     # Set 'telegram_chat_id' as the index
#     features_df = features_df.set_index("telegram_chat_id")

#     # Calculate cosine similarity for each pair of rows
#     similarities = {}
#     for chat_id_1, chat_id_2 in combinations(features_df.index, 2):
#         fi = features_df.loc[chat_id_1].values.reshape(1, -1)
#         fj = features_df.loc[chat_id_2].values.reshape(1, -1)

#         similarity = cosine_similarity(fi, fj)[0][0]
#         similarities[(chat_id_1, chat_id_2)] = similarity

#     # sorted_edges = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

#     # # Keeping half of the edges based on high weight
#     # half_edge_count = len(sorted_edges) // 2
#     # edges_to_keep = sorted_edges[:half_edge_count]

#     # # Create a new graph with filtered edges
#     # filtered_G = nx.Graph()
#     # for (node1, node2), _ in edges_to_keep:
#     #     filtered_G.add_edge(node1, node2)

#     return similarities


# similarities = calculate_all_similarities(c["1INCH"])

# sql_output = get_direct_link()


# def count_linked_chat_pairs(sql_output):
#     linked_chat_counts = {}

#     for _, row in sql_output.iterrows():
#         chat_id_1 = row["chat_id_1"]
#         chat_id_2 = row["chat_id_2"]

#         # Sort the chat IDs to ensure consistent counting
#         pair = tuple(sorted([chat_id_1, chat_id_2]))

#         if pair in linked_chat_counts:
#             linked_chat_counts[pair] += 1
#         else:
#             linked_chat_counts[pair] = 1

#     return linked_chat_counts


# # Example usage
# linked_chat_counts = count_linked_chat_pairs(sql_output)
