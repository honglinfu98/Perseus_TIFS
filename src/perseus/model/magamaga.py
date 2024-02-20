from os import path
import pickle 
from torch_geometric.loader import DataLoader
from perseus.dataset.compare_paper_graph import combine_features, graph_features
from perseus.dataset.extract.cloudburst_connection import get_scored_signals, get_train_scored_signals, get_valid_scored_signals
from perseus.dataset.preprocess.groudtruth_labeling import create_label_mapping
from perseus.dataset.preprocess.process import features_engineer, get_graphs, process_dataframe, aggregate_data, assign_event_ids
from perseus.dataset.gnn_dataset_preparation import (
    prepare_data,
)

from perseus.dataset.preprocess.groudtruth_labeling import create_label_mapping
from perseus.settings import PROJECT_ROOT
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
# from itertools import combinations
# from sklearn.metrics.pairwise import cosine_similarity

from torch_geometric.utils import to_undirected, is_undirected
import torch
from torch_geometric.data import Data
# from clotho.model.magamaga import get_data_loader


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
                edge_index_list.append(
                    [source, target]
                )
                edge_weight_list.append(weight)
        # Convert lists to PyTorch tensors
        print(key)
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


def get_data_loader(options: str):
    # with open("signals2.pkl", "rb") as file:
    #     test_signals = pickle.load(file)
    # with open("signals3.pkl", "rb") as file:
    #     validate_signals = pickle.load(file)
    # with open("signals1.pkl", "rb") as file:
    #     train_signals = pickle.load(file)
    train_signals = get_train_scored_signals()
    test_signals = get_scored_signals()
    validate_signals = get_valid_scored_signals()

    gs_ls = []
    features_ls = []
    market_features_ls = []
    P_dicts_ls = []

    for i in [train_signals,test_signals,validate_signals]:
        processed_signals = process_dataframe(i)
        ided_signals = assign_event_ids(processed_signals)
        cascade, no_nodes, id_mapping = aggregate_data(ided_signals)
        gs, results, As, P_dict = get_graphs(cascade, no_nodes, id_mapping)
        graph_feature = graph_features(gs)
        market_feature = features_engineer(processed_signals)
        combine_feature = combine_features(market_feature, graph_feature)
        gs_ls.append(gs)
        features_ls.append(combine_feature)
        market_features_ls.append(market_feature)
        P_dicts_ls.append(P_dict)

    label_mapping_ls = []
    label_mapping_ls.append(create_label_mapping(3, "train"))
    label_mapping_ls.append(create_label_mapping(3, "test"))
    label_mapping_ls.append(create_label_mapping(3, "validate"))

    if options == "DDINA":
        train_data = prepare_data(gs_ls[0], features_ls[0], label_mapping_ls[0])
        test_data = prepare_data(gs_ls[1], features_ls[1], label_mapping_ls[1])
        validate_data = prepare_data(gs_ls[2], features_ls[2], label_mapping_ls[2])


        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
        validate_loader = DataLoader(validate_data, batch_size=1, shuffle=True)
    elif options == "COSS":
        train_data = prepare_cos_data(gs_ls[0], market_features_ls[0], label_mapping_ls[0])
        test_data = prepare_cos_data(gs_ls[1], market_features_ls[1], label_mapping_ls[1])
        validate_data = prepare_cos_data(gs_ls[2], market_features_ls[2], label_mapping_ls[2])


        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
        validate_loader = DataLoader(validate_data, batch_size=1, shuffle=True)
    elif options == "DDM":
        train_data = prepare_ddm_data(gs_ls[0], market_features_ls[0], label_mapping_ls[0], P_dicts_ls[0])
        test_data = prepare_ddm_data(gs_ls[1], market_features_ls[1], label_mapping_ls[1], P_dicts_ls[1])
        validate_data = prepare_ddm_data(gs_ls[2], market_features_ls[2], label_mapping_ls[2], P_dicts_ls[2])


        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
        validate_loader = DataLoader(validate_data, batch_size=1, shuffle=True)


    return train_loader, test_loader, validate_loader



if __name__ == "__main__":
    # a,b,c = get_data_loader("DDINA")
    # a,b,c = get_data_loader("COSS")
    aa,bb,cc = get_data_loader("DDM")

