import numpy as np
import torch
from torch_geometric.nn import GATConv
from os import path
import torch.nn.functional as F
from perseus.dataset.compare_paper_graph import combine_features, graph_features
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
from torch_geometric.data import Data


train_signals = get_train_scored_signals()

processed_signals = process_dataframe(train_signals)
ided_signals = assign_event_ids(processed_signals)
cascade, no_nodes, id_mapping, cascade_labeling = aggregate_data(ided_signals)
gs, results, As, P_dict = get_graphs(cascade, no_nodes, id_mapping)
graph_feature = graph_features(gs)
market_feature = features_engineer(processed_signals)
combine_feature = combine_features(market_feature, graph_feature)


label_mapping = read_labeling_csv_back_to_dict("train")

for k in gs.keys():
    try:

        features_buffer = combine_feature[k]
        features_buffer = features_buffer[
            features_buffer["telegram_chat_id"].isin(gs[k].nodes)
        ]

        # Specify feature columns and normalize them
        feature_columns = [
            "average_increase_percentage",
            "number_of_signals",
            "in_ratio",
            "out_ratio",
        ]
        features_to_normalize = features_buffer[feature_columns]
        normalized_features = (
            features_to_normalize - features_to_normalize.mean()
        ) / features_to_normalize.std()

        nan_columns = normalized_features.columns[
            normalized_features.isnull().any()
        ].tolist()

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
        for source_node, target_node in gs[k].edges():
            source = id_to_index[source_node]
            target = id_to_index[target_node]
            edge_index.append([source, target])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        num_labels = 2

        # Prepare labels
        labels = []
        for node_id in features_buffer["telegram_chat_id"]:
            node_labels = label_mapping[k].get(node_id, 0)
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Example after training your model
        class Net(torch.nn.Module):
            def __init__(self, num_features=4, num_classes=2):
                super().__init__()
                self.conv1 = GATConv(num_features, 8, heads=2)
                self.lin1 = torch.nn.Linear(num_features, 2 * 8)
                self.conv2 = GATConv(2 * 8, 8, heads=2)
                self.lin2 = torch.nn.Linear(2 * 8, 2 * 8)
                self.conv3 = GATConv(2 * 8, num_classes, heads=2, concat=False)
                self.lin3 = torch.nn.Linear(2 * 8, num_classes)

            def forward(self, x, edge_index):
                if torch.isnan(x).any() or torch.isinf(x).any():
                    print("NaN or Inf in input feature x")
                if torch.isnan(edge_index).any() or torch.isinf(edge_index).any():
                    print("NaN or Inf in edge_index")
                x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
                x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
                x = self.conv3(x, edge_index) + self.lin3(x)
                return x

        model_1 = Net(4, 2)
        model_1.load_state_dict(
            torch.load(path.join(PROJECT_ROOT, "data", "model_weights.pth"))
        )
        model_1.to(device)

        # Extract node IDs in the order they are being processed
        node_ids = features_buffer["telegram_chat_id"].tolist()

        model_1.eval()  # Set the model to evaluation mode

        all_predictions = []

        with torch.no_grad():
            data = data.to(device)
            output = model_1(data.x, data.edge_index)
            predictions = output.argmax(dim=1)  # Get the class with the highest score
            # Pair each prediction with its corresponding node ID and actual label
            for i, pred in enumerate(predictions.cpu().numpy()):
                node_id = node_ids[i]
                actual_label = label_mapping[k].get(
                    node_id, 0
                )  # Default to 0 if not found
                all_predictions.append((node_id, pred, actual_label))

        # Now `all_predictions` is a list of tuples, each containing (node_id, predicted_label, actual_label)
        print(all_predictions)
        print(k)
    except Exception as e:
        print(e)
        continue
