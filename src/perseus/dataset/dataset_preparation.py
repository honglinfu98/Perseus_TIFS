"""
This script is used to prepare the dataset for the model training. It includes the following functions:
    - prepare_data: Prepare the data for the Directed DINA training
    - prepare_ddm_data: Prepare the data for the Weighted DINA model
    - prepare_cos_data: Prepare the data for the Cosine similarity model
    - split_data: Split the data into temporal tasks 
    - split_data_noloader: Split the data into temporal tasks, but without the DataLoader
    - get_split_data_pickle: Load the split data from the pickle file
    - get_train_test_validate_data_pickle: Load the train, test, and validate data from the pickle file
    - get_train_test_validate_data: Split the data into non-temporal tasks
"""

from os import path
import pickle

# import random
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
from perseus.settings import PROJECT_ROOT


def prepare_data(graphs: dict, features: dict, label_mapping: dict):
    """
    Prepare the data for the Directed DINA model
    """
    prepared_data = []
    for key, graph in graphs.items():
        # Extract features for nodes present in the graph
        features_buffer = features[key]
        features_buffer = features_buffer[
            features_buffer["telegram_chat_id"].isin(graph.nodes)
        ]

        # Selecting feature columns for normalization
        feature_columns = [
            # "average_speed",  # market
            # "sum_total_targets",  # osn
            "average_increase_percentage",  # market
            "number_of_signals",  # osn
            "sum_targets_achieved",  # osn
            "rating",  # topological
            # "in_ratio",  # topological
            # "out_ratio",  # topological
            # "out_nodes",  # topological
            # "density",  # topological
            # "clustering_coeff",  # topological
            # "closeness_centrality",  # topological
            # "eff_size",  # topological
            # "efficiency",  # topological
            # "betweenness_centrality",
            # "pagerank",
            "ego_in_ratio",
            "ego_out_ratio",
            "ego_out_nodes",
            "eff_size",
            "efficiency",
            "density",
            "clustering_coeff",
            "closeness_centrality",
            "pagerank",
            "betweenness_centrality",
            # "ego_weighted_in_ratio",
            # "ego_weighted_out_ratio",
            # "ego_out_weights",
            # "weighted_closeness_centrality",
            # "weighted_betweenness_centrality",
            # "weighted_pagerank",
            # "ego_weighted_eff_size",
            # "ego_weighted_efficiency",
            # "weighted_clustering_coefficient",
            # "ego_weighted_density",
        ]

        features_to_normalize = features_buffer[feature_columns]
        std = features_to_normalize.std()
        mean = features_to_normalize.mean()

        # Normalize, but handle cases where std is zero
        normalized_features = features_to_normalize.copy()

        # Only normalize features where std is non-zero
        non_zero_std = std != 0
        normalized_features.loc[:, non_zero_std] = (
            features_to_normalize.loc[:, non_zero_std] - mean[non_zero_std]
        ) / std[non_zero_std]

        # For features with zero std, assign them a constant value (like 0)
        normalized_features.loc[:, ~non_zero_std] = 0

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

        # Prepare labels
        labels = [
            label_mapping[key].get(node_id, 0)
            for node_id in features_buffer["telegram_chat_id"]
        ]
        labels_tensor = torch.tensor(labels, dtype=torch.float).unsqueeze(1)

        # Create a Data object
        data = Data(x=node_attributes, edge_index=edge_index, y=labels_tensor)

        prepared_data.append(data)

    return prepared_data


def prepare_ddm_data(graphs: dict, features: dict, label_mapping: dict, P_dict: dict):
    """
    Prepare the data for the Weighted DINA model
    """
    prepared_data = []
    for key, graph in graphs.items():
        # Extract features for nodes present in the graph
        features_buffer = features[key]
        features_buffer = features_buffer[
            features_buffer["telegram_chat_id"].isin(graph.nodes)
        ]
        # Selecting feature columns for normalization
        feature_columns = [
            # "average_speed",  # market
            # "sum_total_targets",  # osn
            "average_increase_percentage",  # market
            "number_of_signals",  # osn
            "sum_targets_achieved",  # osn
            "rating",  # topological
            # "weighted_in_ratio",
            # "weighted_out_ratio",
            # "weighted_out_degree",
            # "weighted_closeness_centrality",
            # "weighted_betweenness_centrality",
            # "weighted_pagerank",
            # "weighted_eff_size",
            # "weighted_efficiency",
            # "weighted_clustering_coefficient",
            # "weighted_density",
            "ego_weighted_in_ratio",
            "ego_weighted_out_ratio",
            "ego_out_weights",
            "weighted_closeness_centrality",
            "weighted_betweenness_centrality",
            "weighted_pagerank",
            "ego_weighted_eff_size",
            "ego_weighted_efficiency",
            "weighted_clustering_coefficient",
            "ego_weighted_density",
            # "ego_in_ratio",
            # "ego_out_ratio",
            # "ego_out_nodes",
            # "eff_size",
            # "efficiency",
            # "density",
            # "clustering_coeff",
            # "closeness_centrality",
            # "pagerank",
            # "betweenness_centrality",
        ]

        features_to_process = features_buffer[feature_columns]

        std = features_to_process.std()
        mean = features_to_process.mean()

        # Normalize, but handle cases where std is zero
        normalized_features = features_to_process.copy()

        # Only normalize features where std is non-zero
        non_zero_std = std != 0
        normalized_features.loc[:, non_zero_std] = (
            features_to_process.loc[:, non_zero_std] - mean[non_zero_std]
        ) / std[non_zero_std]

        # For features with zero std, assign them a constant value (like 0)
        normalized_features.loc[:, ~non_zero_std] = 0

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
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)

        # Prepare labels
        labels = [
            label_mapping[key].get(node_id, 0)
            for node_id in features_buffer["telegram_chat_id"]
        ]
        labels_tensor = torch.tensor(labels, dtype=torch.float).unsqueeze(1)

        # Create a Data object
        data = Data(
            x=node_attributes,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=labels_tensor,
        )

        prepared_data.append(data)
    return prepared_data


def prepare_cos_data(graphs: dict, features: dict, label_mapping: dict):
    """
    Prepare the data for the Cosine similarity model
    """
    prepared_data = []
    for key, graph in graphs.items():
        # Extract features for nodes present in the graph
        features_buffer = features[key]
        features_buffer = features_buffer[
            features_buffer["telegram_chat_id"].isin(graph.nodes)
        ]
        # Specify feature columns and normalize them
        feature_columns = [
            # "average_speed",  # market
            # "sum_total_targets",  # osn
            "average_increase_percentage",  # market
            "number_of_signals",  # osn
            "sum_targets_achieved",  # osn
            "rating",  # topological
            # "weighted_in_ratio",
            # "weighted_out_ratio",
            # "weighted_out_degree",
            # "weighted_closeness_centrality",
            # "weighted_betweenness_centrality",
            # "weighted_pagerank",
            # "weighted_eff_size",
            # "weighted_efficiency",
            # "weighted_clustering_coefficient",
            # "weighted_density",
            "ego_weighted_in_ratio",
            "ego_weighted_out_ratio",
            "ego_out_weights",
            "weighted_closeness_centrality",
            "weighted_betweenness_centrality",
            "weighted_pagerank",
            "ego_weighted_eff_size",
            "ego_weighted_efficiency",
            "weighted_clustering_coefficient",
            "ego_weighted_density",
            # "ego_in_ratio",
            # "ego_out_ratio",
            # "ego_out_nodes",
            # "eff_size",
            # "efficiency",
            # "density",
            # "clustering_coeff",
            # "closeness_centrality",
            # "pagerank",
            # "betweenness_centrality",
        ]

        features_to_process = features_buffer[feature_columns]

        std = features_to_process.std()
        mean = features_to_process.mean()

        # Normalize, but handle cases where std is zero
        normalized_features = features_to_process.copy()

        # Only normalize features where std is non-zero
        non_zero_std = std != 0
        normalized_features.loc[:, non_zero_std] = (
            features_to_process.loc[:, non_zero_std] - mean[non_zero_std]
        ) / std[non_zero_std]

        # For features with zero std, assign them a constant value (like 0)
        normalized_features.loc[:, ~non_zero_std] = 0

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

        # Prepare labels
        labels = [
            label_mapping[key].get(node_id, 0)
            for node_id in features_buffer["telegram_chat_id"]
        ]
        labels_tensor = torch.tensor(labels, dtype=torch.float).unsqueeze(1)

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


def split_data(options: str, loader: bool = True):
    """
    Split the data into train, test, and validate sets for temporal tasks
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
    P_com_ls = []

    for dataset in datasets:
        processed_signals = process_dataframe(dataset)
        ided_signals = assign_event_ids(processed_signals)
        cascade, no_nodes, id_mapping, cascade_labeling = aggregate_data(ided_signals)
        gs, results, As, P_dict, P_com = get_graphs(cascade, no_nodes, id_mapping)
        graph_feature = graph_features(gs)
        market_feature = features_engineer(processed_signals)
        weighted_feature = compute_weighted_graph_features(P_com)

        combine_feature = combine_features(
            market_feature, graph_feature, weighted_feature
        )

        gs_ls.append(gs)
        features_ls.append(combine_feature)
        market_features_ls.append(market_feature)
        P_dicts_ls.append(P_dict)
        P_com_ls.append(P_com)

    label_mapping_ls = [
        read_labeling_csv_back_to_dict("train"),
        read_labeling_csv_back_to_dict("test"),
        read_labeling_csv_back_to_dict("valid"),
    ]

    for i in range(3):
        common_keys = set(
            label_mapping_ls[i].keys()
        )  # Assuming label_mapping_ls[i] is a dict with relevant keys
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
        P_com_ls[i] = {
            key: P_com_ls[i][key] for key in common_keys if key in P_com_ls[i]
        }

    if options == "DDINA":
        train_data = prepare_data(gs_ls[0], features_ls[0], label_mapping_ls[0])
        test_data = prepare_data(gs_ls[1], features_ls[1], label_mapping_ls[1])
        validate_data = prepare_data(gs_ls[2], features_ls[2], label_mapping_ls[2])

    elif options == "COSS":
        train_data = prepare_cos_data(gs_ls[0], features_ls[0], label_mapping_ls[0])
        test_data = prepare_cos_data(gs_ls[1], features_ls[1], label_mapping_ls[1])
        validate_data = prepare_cos_data(gs_ls[2], features_ls[2], label_mapping_ls[2])
    elif options == "DDM":
        train_data = prepare_ddm_data(
            gs_ls[0], features_ls[0], label_mapping_ls[0], P_com_ls[0]
        )
        test_data = prepare_ddm_data(
            gs_ls[1], features_ls[1], label_mapping_ls[1], P_com_ls[1]
        )
        validate_data = prepare_ddm_data(
            gs_ls[2], features_ls[2], label_mapping_ls[2], P_com_ls[2]
        )

    if loader:
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
        validate_loader = DataLoader(validate_data, batch_size=1, shuffle=True)

        return train_loader, test_loader, validate_loader

    else:

        return train_data, test_data, validate_data


def get_split_data_pickle(options: str):
    """
    Load the data for temporal tasks using the pickle file
    """
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


def get_split_data_pickle_f(options: str):
    """
    Load the data for temporal tasks using the pickle file
    """
    if options == "DDINA":
        with open(path.join(PROJECT_ROOT, "data", "DDINA_data_f.pkl"), "rb") as file:
            data = pickle.load(file)

    elif options == "COSS":
        with open(path.join(PROJECT_ROOT, "data", "COSS_data_f.pkl"), "rb") as file:
            data = pickle.load(file)

    elif options == "DDM":
        with open(path.join(PROJECT_ROOT, "data", "DDM_data_f.pkl"), "rb") as file:
            data = pickle.load(file)

    train_loader = data[0]
    test_loader = data[1]
    validate_loader = data[2]

    return train_loader, test_loader, validate_loader


def get_split_data_pickle_t(options: str):
    """
    Load the data for temporal tasks using the pickle file
    """
    if options == "DDINA":
        with open(path.join(PROJECT_ROOT, "data", "DDINA_data_t.pkl"), "rb") as file:
            data = pickle.load(file)

    elif options == "COSS":
        with open(path.join(PROJECT_ROOT, "data", "COSS_data_t.pkl"), "rb") as file:
            data = pickle.load(file)

    elif options == "DDM":
        with open(path.join(PROJECT_ROOT, "data", "DDM_data_t.pkl"), "rb") as file:
            data = pickle.load(file)

    train_loader = data[0]
    test_loader = data[1]
    validate_loader = data[2]

    return train_loader, test_loader, validate_loader


def get_split_data_pickle_fv(options: str):
    """
    Load the data for temporal tasks using the pickle file
    """
    if options == "DDINA":
        with open(path.join(PROJECT_ROOT, "data", "DDINA_data_fv.pkl"), "rb") as file:
            data = pickle.load(file)

    elif options == "COSS":
        with open(path.join(PROJECT_ROOT, "data", "COSS_data_fv.pkl"), "rb") as file:
            data = pickle.load(file)

    elif options == "DDM":
        with open(path.join(PROJECT_ROOT, "data", "DDM_data_fv.pkl"), "rb") as file:
            data = pickle.load(file)

    train_loader = data[0]
    test_loader = data[1]
    validate_loader = data[2]

    return train_loader, test_loader, validate_loader


def get_split_data_pickle_tr(options: str):
    """
    Load the data for temporal tasks using the pickle file
    """
    if options == "DDINA":
        with open(path.join(PROJECT_ROOT, "data", "DDINA_data_tr.pkl"), "rb") as file:
            data = pickle.load(file)

    elif options == "COSS":
        with open(path.join(PROJECT_ROOT, "data", "COSS_data_tr.pkl"), "rb") as file:
            data = pickle.load(file)

    elif options == "DDM":
        with open(path.join(PROJECT_ROOT, "data", "DDM_data_tr.pkl"), "rb") as file:
            data = pickle.load(file)

    train_loader = data[0]
    test_loader = data[1]
    validate_loader = data[2]

    return train_loader, test_loader, validate_loader


def get_split_data_pickle_e(options: str):
    """
    Load the data for temporal tasks using the pickle file
    """
    if options == "DDINA":
        with open(path.join(PROJECT_ROOT, "data", "DDINA_data_e.pkl"), "rb") as file:
            data = pickle.load(file)

    elif options == "COSS":
        with open(path.join(PROJECT_ROOT, "data", "COSS_data_e.pkl"), "rb") as file:
            data = pickle.load(file)

    elif options == "DDM":
        with open(path.join(PROJECT_ROOT, "data", "DDM_data_e.pkl"), "rb") as file:
            data = pickle.load(file)

    train_loader = data[0]
    test_loader = data[1]
    validate_loader = data[2]

    return train_loader, test_loader, validate_loader


def get_split_data_pickle_s(options: str):
    """
    Load the data for temporal tasks using the pickle file
    """
    if options == "DDINA":
        with open(path.join(PROJECT_ROOT, "data", "DDINA_data_s.pkl"), "rb") as file:
            data = pickle.load(file)

    elif options == "COSS":
        with open(path.join(PROJECT_ROOT, "data", "COSS_data_s.pkl"), "rb") as file:
            data = pickle.load(file)

    elif options == "DDM":
        with open(path.join(PROJECT_ROOT, "data", "DDM_data_s.pkl"), "rb") as file:
            data = pickle.load(file)

    train_loader = data[0]
    test_loader = data[1]
    validate_loader = data[2]

    return train_loader, test_loader, validate_loader


def get_split_data_pickle_m(options: str):
    """
    Load the data for temporal tasks using the pickle file
    """
    if options == "DDINA":
        with open(path.join(PROJECT_ROOT, "data", "DDINA_data_m.pkl"), "rb") as file:
            data = pickle.load(file)

    elif options == "COSS":
        with open(path.join(PROJECT_ROOT, "data", "COSS_data_m.pkl"), "rb") as file:
            data = pickle.load(file)

    elif options == "DDM":
        with open(path.join(PROJECT_ROOT, "data", "DDM_data_m.pkl"), "rb") as file:
            data = pickle.load(file)

    train_loader = data[0]
    test_loader = data[1]
    validate_loader = data[2]

    return train_loader, test_loader, validate_loader


def get_split_data_pickle_l(options: str):
    """
    Load the data for temporal tasks using the pickle file
    """
    if options == "DDINA":
        with open(path.join(PROJECT_ROOT, "data", "DDINA_data_l.pkl"), "rb") as file:
            data = pickle.load(file)

    elif options == "COSS":
        with open(path.join(PROJECT_ROOT, "data", "COSS_data_l.pkl"), "rb") as file:
            data = pickle.load(file)

    elif options == "DDM":
        with open(path.join(PROJECT_ROOT, "data", "DDM_data_l.pkl"), "rb") as file:
            data = pickle.load(file)

    train_loader = data[0]
    test_loader = data[1]
    validate_loader = data[2]

    return train_loader, test_loader, validate_loader


def get_split_data_pickle_l_wc(options: str):
    """
    Load the data for temporal tasks using the pickle file
    """
    if options == "DDINA":
        with open(path.join(PROJECT_ROOT, "data", "DDINA_data_l_wc.pkl"), "rb") as file:
            data = pickle.load(file)

    elif options == "COSS":
        with open(path.join(PROJECT_ROOT, "data", "COSS_data_l_wc.pkl"), "rb") as file:
            data = pickle.load(file)

    elif options == "DDM":
        with open(path.join(PROJECT_ROOT, "data", "DDM_data_l_wc.pkl"), "rb") as file:
            data = pickle.load(file)

    train_loader = data[0]
    test_loader = data[1]
    validate_loader = data[2]

    return train_loader, test_loader, validate_loader


if __name__ == "__main__":
    # a = get_split_data_pickle_t("DDM")

    a = split_data("DDINA", loader=True)
    # save it in pickle
    with open(path.join(PROJECT_ROOT, "data", "DDINA_data_l_wc.pkl"), "wb") as file:
        pickle.dump(a, file)
    b = split_data("COSS", loader=True)
    # save it in pickle
    with open(path.join(PROJECT_ROOT, "data", "COSS_data_l_wc.pkl"), "wb") as file:
        pickle.dump(b, file)
    c = split_data("DDM", loader=True)
    # save it in pickle
    with open(path.join(PROJECT_ROOT, "data", "DDM_data_l_wc.pkl"), "wb") as file:
        pickle.dump(c, file)
