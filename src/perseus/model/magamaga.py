from os import path
import pickle
import random
import pandas as pd
import networkx as nx
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from perseus.dataset.preprocess.train_test_validate import (
    get_test_scored_signals,
    get_train_scored_signals,
    get_valid_scored_signals,
)
from perseus.dataset.preprocess.process import (
    features_engineer,
    get_graphs,
    process_dataframe,
    aggregate_data,
    assign_event_ids,
)
from perseus.dataset.preprocess.groudtruth_labeling import (
    read_labeling_csv_back_to_dict,
)
from perseus.settings import PROJECT_ROOT


def calculate_effsize_efficiency(G, ego):
    # Get the ego network
    ego_net = nx.ego_graph(G, ego, undirected=False)

    # Get alters in the ego network (excluding ego)
    alters = set(ego_net.nodes()) - {ego}
    num_alters = len(alters)
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
            continue

        node_attributes = torch.tensor(normalized_features.values, dtype=torch.float)

        # Map node IDs to indices
        id_to_index = {
            telegram_chat_id: index
            for index, telegram_chat_id in enumerate(
                features_buffer["telegram_chat_id"]
            )
        }

        # Create edge index
        edge_index = []
        for source_node, target_node in graph.edges():
            source = id_to_index[source_node]
            target = id_to_index[target_node]
            edge_index.append([source, target])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        num_labels = 2

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
        data = Data(x=node_attributes, edge_index=edge_index, y=labels_tensor)

        prepared_data.append(data)

    return prepared_data


def filter_data_by_keys(data_list, keys):
    """
    Filters data in data_list to include only items with keys in 'keys'.
    This is a placeholder function; its implementation depends on how the data
    is structured.
    """
    # Assuming data_list contains dictionaries or similar structures
    filtered_data = [
        {k: v for k, v in data_item.items() if k in keys} for data_item in data_list
    ]
    return filtered_data


def split_data(options: str):
    """
    Common keys enabled
    """
    # Initial data loading and processing
    train_signals = get_train_scored_signals()
    test_signals = get_test_scored_signals()
    validate_signals = get_valid_scored_signals()

    datasets = [train_signals, test_signals, validate_signals]
    gs_ls = []
    features_ls = []
    market_features_ls = []
    P_dicts_ls = []
    label_mapping_ls = []

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

    label_mapping_ls = [
        read_labeling_csv_back_to_dict("train"),
        read_labeling_csv_back_to_dict("test"),
        read_labeling_csv_back_to_dict("valid"),
    ]

    # Now, let's find the common keys and filter each dataset accordingly
    # Assuming each item in the ls lists and label_mapping_ls is a dict or can be filtered by keys
    for i in range(3):
        common_keys = set(
            label_mapping_ls[i].keys()
        )  # Assuming label_mapping_ls[i] is a dict with relevant keys

        # Filter gs_ls, features_ls, market_features_ls, and P_dicts_ls based on common_keys
        # This step is highly dependent on the structure of your data. You need to implement the filtering logic based on your actual data structure.
        # The following are placeholders to illustrate the process:
        gs_ls[i] = {key: gs_ls[i][key] for key in common_keys if key in gs_ls[i]}
        features_ls[i] = {
            key: features_ls[i][key] for key in common_keys if key in features_ls[i]
        }
        market_features_ls[i] = {
            key: market_features_ls[i][key]
            for key in common_keys
            if key in market_features_ls[i]
        }
        P_dicts_ls[i] = {
            key: P_dicts_ls[i][key] for key in common_keys if key in P_dicts_ls[i]
        }

    if options == "DDINA":
        train_data = prepare_data(gs_ls[0], features_ls[0], label_mapping_ls[0])
        test_data = prepare_data(gs_ls[1], features_ls[1], label_mapping_ls[1])
        validate_data = prepare_data(gs_ls[2], features_ls[2], label_mapping_ls[2])

        # train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        # test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
        # validate_loader = DataLoader(validate_data, batch_size=1, shuffle=True)
    elif options == "COSS":
        train_data = prepare_cos_data(
            gs_ls[0], market_features_ls[0], label_mapping_ls[0]
        )
        test_data = prepare_cos_data(
            gs_ls[1], market_features_ls[1], label_mapping_ls[1]
        )
        validate_data = prepare_cos_data(
            gs_ls[2], market_features_ls[2], label_mapping_ls[2]
        )

        # train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        # test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
        # validate_loader = DataLoader(validate_data, batch_size=1, shuffle=True)
    elif options == "DDM":
        train_data = prepare_ddm_data(
            gs_ls[0], market_features_ls[0], label_mapping_ls[0], P_dicts_ls[0]
        )
        test_data = prepare_ddm_data(
            gs_ls[1], market_features_ls[1], label_mapping_ls[1], P_dicts_ls[1]
        )
        validate_data = prepare_ddm_data(
            gs_ls[2], market_features_ls[2], label_mapping_ls[2], P_dicts_ls[2]
        )

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
    validate_loader = DataLoader(validate_data, batch_size=1, shuffle=True)

    return train_loader, test_loader, validate_loader


def split_data_noloader(options: str):
    # Initial data loading and processing
    train_signals = get_train_scored_signals()
    test_signals = get_test_scored_signals()
    validate_signals = get_valid_scored_signals()

    datasets = [train_signals, test_signals, validate_signals]
    gs_ls = []
    features_ls = []
    market_features_ls = []
    P_dicts_ls = []
    label_mapping_ls = []

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

    label_mapping_ls = [
        read_labeling_csv_back_to_dict("train"),
        read_labeling_csv_back_to_dict("test"),
        read_labeling_csv_back_to_dict("valid"),
    ]

    # Now, let's find the common keys and filter each dataset accordingly
    # Assuming each item in the ls lists and label_mapping_ls is a dict or can be filtered by keys
    for i in range(3):
        common_keys = set(
            label_mapping_ls[i].keys()
        )  # Assuming label_mapping_ls[i] is a dict with relevant keys

        # Filter gs_ls, features_ls, market_features_ls, and P_dicts_ls based on common_keys
        # This step is highly dependent on the structure of your data. You need to implement the filtering logic based on your actual data structure.
        # The following are placeholders to illustrate the process:
        gs_ls[i] = {key: gs_ls[i][key] for key in common_keys if key in gs_ls[i]}
        features_ls[i] = {
            key: features_ls[i][key] for key in common_keys if key in features_ls[i]
        }
        market_features_ls[i] = {
            key: market_features_ls[i][key]
            for key in common_keys
            if key in market_features_ls[i]
        }
        P_dicts_ls[i] = {
            key: P_dicts_ls[i][key] for key in common_keys if key in P_dicts_ls[i]
        }

    if options == "DDINA":
        train_data = prepare_data(gs_ls[0], features_ls[0], label_mapping_ls[0])
        test_data = prepare_data(gs_ls[1], features_ls[1], label_mapping_ls[1])
        validate_data = prepare_data(gs_ls[2], features_ls[2], label_mapping_ls[2])

        # train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        # test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
        # validate_loader = DataLoader(validate_data, batch_size=1, shuffle=True)
    elif options == "COSS":
        train_data = prepare_cos_data(
            gs_ls[0], market_features_ls[0], label_mapping_ls[0]
        )
        test_data = prepare_cos_data(
            gs_ls[1], market_features_ls[1], label_mapping_ls[1]
        )
        validate_data = prepare_cos_data(
            gs_ls[2], market_features_ls[2], label_mapping_ls[2]
        )

        # train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        # test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
        # validate_loader = DataLoader(validate_data, batch_size=1, shuffle=True)
    elif options == "DDM":
        train_data = prepare_ddm_data(
            gs_ls[0], market_features_ls[0], label_mapping_ls[0], P_dicts_ls[0]
        )
        test_data = prepare_ddm_data(
            gs_ls[1], market_features_ls[1], label_mapping_ls[1], P_dicts_ls[1]
        )
        validate_data = prepare_ddm_data(
            gs_ls[2], market_features_ls[2], label_mapping_ls[2], P_dicts_ls[2]
        )

    # train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    # test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
    # validate_loader = DataLoader(validate_data, batch_size=1, shuffle=True)

    return train_data, test_data, validate_data


def prepare_ddm_data(graphs, features, label_mapping, P_dict):
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
            continue
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

        # Populate edge index and weights
        for (source_node, target_node), weight in P_dict[key].items():
            if weight > 0:  # Ensure both nodes are in the id_to_index mapping
                source = id_to_index.get(source_node)
                target = id_to_index.get(target_node)
                edge_index_list.append([source, target])
                edge_weight_list.append(weight)
        # Convert lists to PyTorch tensors
        print(key)
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)
        # Populate the lists
        num_labels = 2
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

        # undirect_edge_index = to_undirected(data.edge_index, num_nodes=data.x.size(0))

        # # Create a Data object
        # data = Data(
        #     x=node_attributes,
        #     edge_index=undirect_edge_index,
        #     edge_weight=edge_weight,
        #     y=labels_tensor,
        # )

        prepared_data.append(data)
    return prepared_data


def prepare_cos_data(graphs, features, label_mapping):
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
            continue

        node_attributes = torch.tensor(normalized_features.values, dtype=torch.float)
        # Map node IDs to indices
        id_to_index = {
            telegram_chat_id: index
            for index, telegram_chat_id in enumerate(
                features_buffer["telegram_chat_id"]
            )
        }

        # Set index and drop NaN values in the DataFrame
        features_df = features_buffer.set_index("telegram_chat_id")[
            feature_columns
        ].dropna()

        # Assuming features_df is a DataFrame where each row is a node's features
        # Assuming id_to_index is a dictionary mapping chat_id to a node index

        edge_index_list = []
        edge_weight_list = []

        # Calculate cosine similarity for each pair of nodes
        for i, chat_id_1 in enumerate(
            features_df.index[:-1]
        ):  # No need to include the last index in the outer loop
            for j, chat_id_2 in enumerate(
                features_df.index[i + 1 :]
            ):  # Start from i+1 to avoid self-similarities
                fi = features_df.loc[chat_id_1].values.reshape(1, -1)
                fj = features_df.loc[chat_id_2].values.reshape(1, -1)
                similarity = cosine_similarity(fi, fj)[0][0]
                source = id_to_index[chat_id_1]
                target = id_to_index[chat_id_2]

                edge_index_list.append([source, target])
                edge_index_list.append(
                    [target, source]
                )  # Add the reverse direction as well
                edge_weight_list.append(similarity)
                edge_weight_list.append(
                    similarity
                )  # Same weight for the reverse direction

        # Convert lists to PyTorch tensors
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)

        # Populate the lists
        num_labels = 2

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

    return prepared_data


def get_data_set(options: str):
    # with open("signals2.pkl", "rb") as file:
    #     test_signals = pickle.load(file)
    # with open("signals3.pkl", "rb") as file:
    #     validate_signals = pickle.load(file)
    # with open("signals1.pkl", "rb") as file:
    #     train_signals = pickle.load(file)
    train_signals = get_train_scored_signals()
    test_signals = get_test_scored_signals()
    validate_signals = get_valid_scored_signals()

    gs_ls = []
    features_ls = []
    market_features_ls = []
    P_dicts_ls = []

    for i in [train_signals, test_signals, validate_signals]:
        processed_signals = process_dataframe(i)
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

    label_mapping_ls = []
    # label_mapping_ls.append(create_label_mapping(3, "train", features_ls[0]))
    # label_mapping_ls.append(create_label_mapping(3, "test", features_ls[1]))
    # label_mapping_ls.append(create_label_mapping(3, "validate", features_ls[2]))

    label_mapping_ls.append(read_labeling_csv_back_to_dict("train"))
    label_mapping_ls.append(read_labeling_csv_back_to_dict("test"))
    label_mapping_ls.append(read_labeling_csv_back_to_dict("valid"))

    if options == "DDINA":
        train_data = prepare_data(gs_ls[0], features_ls[0], label_mapping_ls[0])
        test_data = prepare_data(gs_ls[1], features_ls[1], label_mapping_ls[1])
        validate_data = prepare_data(gs_ls[2], features_ls[2], label_mapping_ls[2])

        # train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        # test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
        # validate_loader = DataLoader(validate_data, batch_size=1, shuffle=True)
    elif options == "COSS":
        train_data = prepare_cos_data(
            gs_ls[0], market_features_ls[0], label_mapping_ls[0]
        )
        test_data = prepare_cos_data(
            gs_ls[1], market_features_ls[1], label_mapping_ls[1]
        )
        validate_data = prepare_cos_data(
            gs_ls[2], market_features_ls[2], label_mapping_ls[2]
        )

        # train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        # test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
        # validate_loader = DataLoader(validate_data, batch_size=1, shuffle=True)
    elif options == "DDM":
        train_data = prepare_ddm_data(
            gs_ls[0], market_features_ls[0], label_mapping_ls[0], P_dicts_ls[0]
        )
        test_data = prepare_ddm_data(
            gs_ls[1], market_features_ls[1], label_mapping_ls[1], P_dicts_ls[1]
        )
        validate_data = prepare_ddm_data(
            gs_ls[2], market_features_ls[2], label_mapping_ls[2], P_dicts_ls[2]
        )

        # train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        # test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
        # validate_loader = DataLoader(validate_data, batch_size=1, shuffle=True)

    return train_data, test_data, validate_data


def get_data_pickle(options: str):

    if options == "DDINA":
        with open(path.join(PROJECT_ROOT, "data", "DDINA_data.pkl"), "rb") as file:
            data = pickle.load(file)

    elif options == "COSS":
        with open(path.join(PROJECT_ROOT, "data", "COSS_data.pkl"), "rb") as file:
            data = pickle.load(file)

    elif options == "DDM":
        with open(path.join(PROJECT_ROOT, "data", "DDM_data.pkl"), "rb") as file:
            data = pickle.load(file)

    train_loader = data[0]
    test_loader = data[1]
    validate_loader = data[2]

    return train_loader, test_loader, validate_loader


def get_train_test_validate_data_pickle(options: str):

    if options == "DDINA":
        with open(path.join(PROJECT_ROOT, "data", "DDINA_data_T.pkl"), "rb") as file:
            data = pickle.load(file)

    elif options == "COSS":
        with open(path.join(PROJECT_ROOT, "data", "COSS_data_T.pkl"), "rb") as file:
            data = pickle.load(file)

    elif options == "DDM":
        with open(path.join(PROJECT_ROOT, "data", "DDM_data_T.pkl"), "rb") as file:
            data = pickle.load(file)

    train_loader = data[0]
    test_loader = data[1]
    validate_loader = data[2]

    return train_loader, test_loader, validate_loader


def get_train_test_validate_data(options: str):

    if options == "DDINA":
        data = split_data_noloader("DDINA")

        # with open(path.join(PROJECT_ROOT, "data", "DDINA_data.pkl"), "rb") as file:
        #     data = pickle.load(file)

    elif options == "COSS":
        data = split_data_noloader("COSS")

        # with open(path.join(PROJECT_ROOT, "data", "COSS_data.pkl"), "rb") as file:
        #     data = pickle.load(file)

    elif options == "DDM":
        data = split_data_noloader("DDM")

        # with open(path.join(PROJECT_ROOT, "data", "DDM_data.pkl"), "rb") as file:
        #     data = pickle.load(file)

    dataset = data[0] + data[1] + data[2]

    # Shuffle the dataset to ensure it's randomly ordered
    random.shuffle(dataset)

    # Calculate the size of each set
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - (train_size + val_size)

    # Split the data
    train_data = dataset[:train_size]
    val_data = dataset[train_size : (train_size + val_size)]
    test_data = dataset[(train_size + val_size) :]

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
    validate_loader = DataLoader(val_data, batch_size=1, shuffle=True)

    # train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    # test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
    # validate_loader = DataLoader(validate_data, batch_size=1, shuffle=True)

    return train_loader, test_loader, validate_loader


if __name__ == "__main__":
    # a,b,c = get_data_loader("DDINA")
    # aa,bb,cc = get_data_loader("COSS")
    # aaa,bbb,ccc = get_data_loader("DDM")
    # save the data
    a = split_data("DDINA")
    # b = split_data("COSS")
    # c = split_data("DDM")
    # with open(path.join(PROJECT_ROOT, "data", "DDINA_data.pkl"), "wb") as file:
    #     pickle.dump(a, file)
    # with open(path.join(PROJECT_ROOT, "data", "COSS_data.pkl"), "wb") as file:
    #     pickle.dump(b, file)
    # with open(path.join(PROJECT_ROOT, "data", "DDM_data.pkl"), "wb") as file:
    #     pickle.dump(c, file)
    # a,b,c, = split_data_noloader("DDINA")

    # aa, bb, cc =split_data_noloader("DDM")

    # aaa,bbb,ccc = split_data_noloader("COSS")

    # a = get_train_test_validate_data("DDINA")
    # with open(path.join(PROJECT_ROOT, "data", "DDINA_data_T.pkl"), "wb") as file:
    #     pickle.dump(a, file)
    # b = get_train_test_validate_data("COSS")
    # with open(path.join(PROJECT_ROOT, "data", "COSS_data_T.pkl"), "wb") as file:
    #     pickle.dump(b, file)
    # c = get_train_test_validate_data("DDM")
    # with open(path.join(PROJECT_ROOT, "data", "DDM_data_T.pkl"), "wb") as file:
    #     pickle.dump(c, file)
    # all = get_all_range_data("DDINA")

    # dina = split_data("DDINA")
    # cos = split_data("COSS")
    # ddm = split_data("DDM")
