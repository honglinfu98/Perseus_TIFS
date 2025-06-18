import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from os import path
from perseus.settings import PROJECT_ROOT
from perseus.dataset.preprocess.train_test_validate import get_btc_test_scored_signals
from perseus.dataset.preprocess.process import (
    features_engineer,
    get_graphs,
    process_dataframe,
    aggregate_data,
    assign_event_ids,
    combine_features,
    graph_features,
    out_ego_graph,
    in_ego_graph,
    calculate_effsize_efficiency,
    compute_weighted_graph_features,
)
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy import stats
from sklearn.metrics import f1_score, precision_recall_curve
from perseus.dataset.preprocess.groudtruth_labeling import (
    read_labeling_csv_back_to_dict,
)

# --- Get feature columns for DDM from dataset_preparation.py ---
DDM_FEATURE_COLUMNS = [
    "average_btc_base_return",
    "average_increase_percentage",
    "sum_targets_achieved",
    "rating",
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
]

DDINA_FEATURE_COLUMNS = [
    "average_btc_base_return",
    "average_increase_percentage",  # market
    "sum_targets_achieved",  # osn
    "rating",  # topological
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
]

# --- Load the trained model and outputs ---
with open(path.join(PROJECT_ROOT, "data", "sp", "model_saved.pkl"), "rb") as f:
    results = pickle.load(f)
model_info = results["DDINA"]["GraphSAGE"]
batch_size_key = list(model_info.keys())[0]
model_dict = model_info[batch_size_key]
model = model_dict["model"]
model.load_state_dict(model_dict["model_weights"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# --- Find the best threshold using saved outputs ---
all_probs = model_dict["all_probs"]
all_labels = model_dict["all_labels"]

if all_probs.ndim > 1 and all_probs.shape[1] == 1:
    all_probs = all_probs.ravel()
    all_labels = all_labels.ravel()

precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)

# Set your desired minimum precision
min_precision = 0.80  # adjust as needed

# Find indices where precision is above the minimum
high_precision_indices = np.where(precisions[:-1] >= min_precision)[0]

if len(high_precision_indices) > 0:
    # Compute F1 for these thresholds
    f1s = []
    for idx in high_precision_indices:
        t = thresholds[idx]
        preds = (all_probs > t).astype(int)
        f1 = f1_score(all_labels, preds, average="macro", zero_division=0)
        f1s.append(f1)
    f1s = np.array(f1s)
    best_idx = high_precision_indices[f1s.argmax()]
    best_threshold = thresholds[best_idx]
    best_macro_f1 = f1s.max()
    print(
        f"Best threshold with precision >= {min_precision}: {best_threshold:.3f}, Macro F1: {best_macro_f1:.3f}, Precision: {precisions[best_idx]:.3f}, Recall: {recalls[best_idx]:.3f}"
    )
else:
    print(
        f"No threshold found with precision >= {min_precision}. Consider lowering min_precision."
    )

# --- Get new detection signals ---
signals = get_btc_test_scored_signals()

processed_signals = process_dataframe(signals)
ided_signals = assign_event_ids(processed_signals)
cascade_buffer, no_nodes_buffer, id_mapping_buffer, cascade_labeling = aggregate_data(
    ided_signals
)
gs, P_theta = get_graphs(cascade_buffer, no_nodes_buffer, id_mapping_buffer)
graph_feature = graph_features(gs)
market_feature = features_engineer(processed_signals)
weighted_feature = compute_weighted_graph_features(P_theta)
combine_feature = combine_features(market_feature, graph_feature, weighted_feature)

all_predictions = {}

# Prepare a list of (asset_key, Data, node_ids) for batching
batch_data = []
for k, v in gs.items():
    try:
        features_buffer = combine_feature[k]
        features_buffer = features_buffer[
            features_buffer["telegram_chat_id"].isin(v.nodes)
        ]
        features_to_normalize = features_buffer[DDINA_FEATURE_COLUMNS]
        std = features_to_normalize.std()
        mean = features_to_normalize.mean()
        normalized_features = features_to_normalize.copy()
        non_zero_std = std != 0
        normalized_features.loc[:, non_zero_std] = (
            features_to_normalize.loc[:, non_zero_std] - mean[non_zero_std]
        ) / std[non_zero_std]
        normalized_features.loc[:, ~non_zero_std] = 0
        nan_columns = normalized_features.columns[
            normalized_features.isnull().any()
        ].tolist()
        if nan_columns:
            print(f"NaN values detected in key: {k}, Columns: {nan_columns}")
            continue
        node_attributes = torch.tensor(normalized_features.values, dtype=torch.float)
        id_to_index = {
            telegram_chat_id: index
            for index, telegram_chat_id in enumerate(
                features_buffer["telegram_chat_id"]
            )
        }
        edge_index = []
        for source_node, target_node in v.edges():
            source = id_to_index[source_node]
            target = id_to_index[target_node]
            edge_index.append([source, target])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        data = Data(x=node_attributes, edge_index=edge_index)
        node_ids = features_buffer["telegram_chat_id"].tolist()
        batch_data.append((k, data, node_ids))
    except Exception as e:
        print(e)
        continue

# Process in batches of 8
batch_size = 8
for i in range(0, len(batch_data), batch_size):
    batch = batch_data[i : i + batch_size]
    data_list = [item[1].to(device) for item in batch]
    asset_keys = [item[0] for item in batch]
    node_ids_list = [item[2] for item in batch]
    with torch.no_grad():
        output_list = model(data_list)
        for j, output in enumerate(output_list):
            predictions = (output > best_threshold).int()
            all_predictions[asset_keys[j]] = {}
            for idx, pred in enumerate(predictions.cpu().numpy()):
                node_id = node_ids_list[j][idx]
                all_predictions[asset_keys[j]][node_id] = pred

# --- Continue with mastermind detection as before ---
# transformed_predictions = {}
# for asset, nodes in all_predictions.items():
#     transformed_predictions[asset] = {
#         node: prediction for node, prediction in nodes.items()
#     }

# mastermind_set = {}
# for asset, nodes in transformed_predictions.items():
#     mastermind_set[asset] = {
#         node for node, prediction in nodes.items() if prediction == 1
#     }

# output_dict = {}
# for key, marked_nodes in mastermind_set.items():
#     if key in gs:
#         nodes = [i for i in gs[key].nodes]
#         output_dict[key] = {node: (1 if node in marked_nodes else 0) for node in nodes}


# --- Save predictions and labels to pickle file (case_study style) ---
label_mapping = read_labeling_csv_back_to_dict("test")
combined_predictions = {}
for asset, nodes in all_predictions.items():
    combined_predictions[asset] = {}
    for node_id, pred in nodes.items():
        # Get label from label_mapping if available, else 0
        label = label_mapping.get(asset, {}).get(node_id, 0)
        # Save as tuple (prediction, label)
        combined_predictions[asset][node_id] = (int(pred), int(label))

with open(path.join(PROJECT_ROOT, "data", "sp", "predictions.pkl"), "wb") as file:
    pickle.dump(combined_predictions, file)
