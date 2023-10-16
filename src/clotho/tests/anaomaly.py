import pandas as pd
from torch_geometric.data import Data

# from torch_geometric.utils import to_networkx
# import torch_geometric.transforms as T

# import matplotlib.pyplot as plt
# import random
# import numpy as np
# from collections import Counter

import torch

# import pandas as pd
# import networkx as nx
# from pygod.utils import load_data
from pygod.detector import DOMINANT, AnomalyDAE, CONAD

from clotho.extract.cloudburst_connection import get_scored_signals
from clotho.post_processing.graph_inferring import get_graphs
from clotho.pre_processing_summary.scored_signals import (
    aggregate_data,
    assign_event_ids,
    process_dataframe,
    features_engineer,
)


# import pickle

# # Load features.pkl
# with open('features.pkl', 'rb') as file:
#     features = pickle.load(file)

# # Load graph.pkl
# with open('graph.pkl', 'rb') as file:
#     graph_nx = pickle.load(file)


def anamoly_detection(graphs_dict: dict, features_dict: dict):
    mastermind_score = {}

    for coin, graph_nx in graphs_dict.items():
        features_all = features_dict[coin]
        features = features_all[features_all["telegram_chat_id"].isin(graph_nx.nodes)]

        # Selecting feature columns for normalization
        feature_columns = [
            "average_increase_percentage",
            # "average_speed",
            "number_of_signals",
            # "latest_chat_crowd_score",
            # "rating",
        ]

        # Extracting features to be normalized
        features_to_normalize = features[feature_columns]

        # Normalization (Min-Max Scaling)
        # normalized_features_min_max = (features_to_normalize - features_to_normalize.min()) / (features_to_normalize.max() - features_to_normalize.min())

        # OR Normalization (Z-score Normalization)
        normalized_features_z_score = (
            features_to_normalize - features_to_normalize.mean()
        ) / features_to_normalize.std()

        # Convert normalized features DataFrame to tensor
        node_attributes = torch.tensor(
            normalized_features_z_score.values, dtype=torch.float
        )  # Use normalized_features_z_score for Z-score Normalization

        # Mapping telegram_chat_id to row index in DataFrame
        id_to_index = {
            telegram_chat_id: index
            for index, telegram_chat_id in enumerate(features["telegram_chat_id"])
        }

        edge_index = []
        # Assuming graph is a NetworkX graph object
        for edge in graph_nx.edges(data=True):
            source = id_to_index[edge[0]]
            target = id_to_index[edge[1]]
            edge_index.append((source, target))

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        data = Data(x=node_attributes, edge_index=edge_index)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        graph = data.to(device)

        model = DOMINANT(num_layers=4)
        # model = AnomalyDAE()
        # model = OCGNN()
        # model = CONAD()

        model = model.fit(graph)

        scores = model.decision_score_.numpy()

        # Create a DataFrame with telegram_chat_id and their corresponding scores
        score_df = pd.DataFrame(
            {"telegram_chat_id": features["telegram_chat_id"], "score": scores}
        )

        # Sort the DataFrame by scores
        sorted_score_df = score_df.sort_values(by="score", ascending=False)

        mastermind_score[coin] = sorted_score_df

    return mastermind_score


def map_the_id_back_cascade(cascade, id_mapping):
    new_cascades = {}
    for algo, mappings in cascade.items():
        new_cascades[algo] = []
        for mapping in mappings:
            new_mapping = {}
            for key, value in mapping.items():
                if key == "T":
                    new_mapping[key] = value
                else:
                    new_mapping[id_mapping[algo]["new_to_id"][key]] = value
            new_cascades[algo].append(new_mapping)

    return new_cascades


if __name__ == "__main__":
    signals = get_scored_signals()
    processed_signals = process_dataframe(signals)
    ided_signals = assign_event_ids(processed_signals)
    cascade, no_nodes, id_mapping = aggregate_data(ided_signals)
    gs = get_graphs(cascade, no_nodes, id_mapping)
    cascade_old_id = map_the_id_back_cascade(cascade, id_mapping)
    features = features_engineer(processed_signals)
    scores = anamoly_detection(gs, features)
