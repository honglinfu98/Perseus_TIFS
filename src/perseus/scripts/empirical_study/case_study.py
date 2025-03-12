"""
This function is used to create a case study for the empirical study. It is used to demonstrate how to use the trained model to make predictions on a single cascade.
"""

from os import path
import pickle
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from perseus.dataset.preprocess.process import (
    combine_features,
    graph_features,
    compute_weighted_graph_features,
)
from perseus.model.gnn_model import GraphSAGENet
from perseus.dataset.preprocess.groudtruth_labeling import (
    read_labeling_csv_back_to_dict,
)
from perseus.dataset.preprocess.process import (
    features_engineer,
    get_graphs,
    process_dataframe,
    aggregate_data,
    assign_event_ids,
)
from perseus.dataset.preprocess.train_test_validate import (
    get_train_scored_signals,
    get_test_scored_signals,
)
from perseus.settings import PROJECT_ROOT


train_signals = get_test_scored_signals()
processed_signals = process_dataframe(train_signals)
ided_signals = assign_event_ids(processed_signals)
cascade, no_nodes, id_mapping, cascade_labeling = aggregate_data(ided_signals)
gs, results, As, P_dict, P_com = get_graphs(cascade, no_nodes, id_mapping)
graph_feature = graph_features(gs)
market_feature = features_engineer(processed_signals)
weighted_feature = compute_weighted_graph_features(P_com)
combine_feature = combine_features(market_feature, graph_feature, weighted_feature)
label_mapping = read_labeling_csv_back_to_dict("test")
all_predictions = {}

for k, v in gs.items():
    try:
        # Extract features for nodes present in the graph
        features_buffer = combine_feature[k]
        features_buffer = features_buffer[
            features_buffer["telegram_chat_id"].isin(v.nodes)
        ]

        # Selecting feature columns for normalization
        feature_columns = [
            # "average_speed",  # market
            # "sum_total_targets",  # osn
            "average_increase_percentage",  # market
            # "number_of_signals",  # osn
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
            print(f"NaN values detected in key: {k}, Columns: {nan_columns}")
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
        for source_node, target_node in v.edges():
            source = id_to_index[source_node]
            target = id_to_index[target_node]
            edge_index.append([source, target])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Prepare labels
        labels = [
            label_mapping[k].get(node_id, 0)
            for node_id in features_buffer["telegram_chat_id"]
        ]
        labels_tensor = torch.tensor(labels, dtype=torch.float).unsqueeze(1)

        # Create a Data object
        data = Data(x=node_attributes, edge_index=edge_index, y=labels_tensor)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # # Example after training your model
        # class Net(torch.nn.Module):
        #     def __init__(self, num_features=4, num_classes=2):
        #         super().__init__()
        #         self.conv1 = GATConv(num_features, 8, heads=2)
        #         self.lin1 = torch.nn.Linear(num_features, 2 * 8)
        #         self.conv2 = GATConv(2 * 8, 8, heads=2)
        #         self.lin2 = torch.nn.Linear(2 * 8, 2 * 8)
        #         self.conv3 = GATConv(2 * 8, num_classes, heads=2, concat=False)
        #         self.lin3 = torch.nn.Linear(2 * 8, num_classes)

        #     def forward(self, x, edge_index):
        #         if torch.isnan(x).any() or torch.isinf(x).any():
        #             print("NaN or Inf in input feature x")
        #         if torch.isnan(edge_index).any() or torch.isinf(edge_index).any():
        #             print("NaN or Inf in edge_index")
        #         x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        #         x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        #         x = self.conv3(x, edge_index) + self.lin3(x)
        #         return x
        model = GraphSAGENet(13, 8, 1)

        # model_1 = Net(4, 2)
        # # model_1.load_state_dict(
        # #     torch.load(path.join(PROJECT_ROOT, "data", "model_weights.pth"))
        # # )
        with open(path.join(PROJECT_ROOT, "data", "results_wn.pkl"), "rb") as file:
            results = pickle.load(file)
        model.load_state_dict(results["DDINA"]["GraphSAGE"]["model_weights"])

        model.to(device)

        # Extract node IDs in the order they are being processed
        node_ids = features_buffer["telegram_chat_id"].tolist()

        model.eval()  # Set the model to evaluation mode

        all_predictions[k] = {}

        with torch.no_grad():
            data = data.to(device)
            output = model(data.x, data.edge_index)
            predictions = (output[0] > 0.55).int()
            # Pair each prediction with its corresponding node ID and actual label
            for i, pred in enumerate(predictions.cpu().numpy()):
                node_id = node_ids[i]
                actual_label = label_mapping[k].get(
                    node_id, 0
                )  # Default to 0 if not found
                all_predictions[k][node_id] = (pred[0], actual_label)

        # Now `all_predictions` is a list of tuples, each containing (node_id, predicted_label, actual_label)
        print(all_predictions)
        print(k)
    except Exception as e:
        print(e)
        continue


# Save the predictions to a pickle file in data
with open(path.join(PROJECT_ROOT, "data", "buffer", "predictions.pkl"), "wb") as file:
    pickle.dump(all_predictions, file)
