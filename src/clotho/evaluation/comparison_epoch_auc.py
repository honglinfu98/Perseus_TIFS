from os import path
import pickle
import torch
import random
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from clotho.model.data_loader import get_data_loader

# from torch_geometric.loader import DataLoader
# from clotho.settings import PROJECT_ROOT
# from clotho.dataset.preprocess.process import (
#     aggregate_data,
#     assign_event_ids,
#     process_dataframe,
#     features_engineer,
#     get_graphs,
# )
# from clotho.dataset.preprocess.groudtruth_labeling import create_label_mapping
# from clotho.dataset.gnn_dataset_preparation import (
#     graph_features,
#     combine_features,
#     prepare_data,
# )


# GCN Implementation
class GCNNet(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# GraphSAGE Implementation
class GraphSAGENet(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GraphSAGENet, self).__init__()
        self.conv1 = SAGEConv(num_features, 16)
        self.conv2 = SAGEConv(16, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        num_features = 4
        num_classes = 3
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


# with open(path.join(PROJECT_ROOT, "data", "signals.pkl"), "rb") as file:
#     signals = pickle.load(file)
# processed_signals = process_dataframe(signals)
# ided_signals = assign_event_ids(processed_signals)

# cascade, no_nodes, id_mapping = aggregate_data(ided_signals)
# gs, results, As, P_dict = get_graphs(cascade, no_nodes, id_mapping)
# f = graph_features(gs)
# market_features = features_engineer(processed_signals)
# c = combine_features(market_features, f)
# label_mapping = create_label_mapping(3)

# # Split gs into training and testing sets
# all_keys = list(gs.keys())
# random.shuffle(all_keys)
# split_index = int(len(all_keys) * 0.8)  # 80% for training

# train_keys = set(all_keys[:split_index])
# test_keys = set(all_keys[split_index:])

# train_graphs = {key: gs[key] for key in train_keys}
# test_graphs = {key: gs[key] for key in test_keys}

# # Prepare data for training and testing sets
# train_data = prepare_data(train_graphs, c, label_mapping)
# test_data = prepare_data(test_graphs, c, label_mapping)

# train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
# test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
train_loader, test_loader = get_data_loader()


# Training and Testing loop
def run_experiment(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    loss_op = torch.nn.BCEWithLogitsLoss()

    def train():
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data.x, data.edge_index)
            # Check output range
            if torch.isnan(output).any() or torch.isinf(output).any():
                print("NaN or Inf in model output")
                continue
            loss = loss_op(output, data.y.float())
            if torch.isnan(loss) or torch.isinf(loss):
                print("NaN or Inf in loss")
                continue
            total_loss += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
        return total_loss / len(train_loader.dataset)

    @torch.no_grad()
    def test(loader):
        model.eval()
        all_probs, all_labels = [], []
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            all_probs.append(out.cpu())
            all_labels.append(data.y.cpu())
        return all_probs, all_labels

    # Training and Testing loop
    average_auc_scores = []
    for epoch in range(1, 101):
        train()
        probs, labels = test(test_loader)

        probs = torch.cat(probs, dim=0).sigmoid().numpy()
        labels = torch.cat(labels, dim=0).numpy()

        # Compute AUC for each label and average
        auc_scores = []
        for i in range(
            labels.shape[1]
        ):  # Assuming labels is a 2D array: [samples, labels]
            fpr, tpr, _ = roc_curve(labels[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            auc_scores.append(roc_auc)
        average_auc = np.mean(auc_scores)
        average_auc_scores.append(average_auc)

        print(f"Epoch: {epoch:03d}, Average AUC: {average_auc:.4f}")

    # Plotting average AUC scores
    plt.plot(range(1, 101), average_auc_scores, label="Average AUC")
    plt.xlabel("Epoch")
    plt.ylabel("Average AUC")
    plt.title("Average AUC per Epoch")
    plt.legend()
    plt.show()

    return average_auc_scores


# Run experiments for each model
num_features = 4  # Set this according to your dataset
num_classes = 3  # Set this according to your dataset

print("Running GAT Experiment")
gat_auc_scores = run_experiment(Net())

print("\nRunning GCN Experiment")
gcn_auc_scores = run_experiment(GCNNet(num_features, num_classes))

print("\nRunning GraphSAGE Experiment")
graphsage_auc_scores = run_experiment(GraphSAGENet(num_features, num_classes))


plt.figure(figsize=(10, 6))
epochs = range(1, 101)
plt.plot(epochs, gat_auc_scores, label="GAT")
plt.plot(epochs, gcn_auc_scores, label="GCN")
plt.plot(epochs, graphsage_auc_scores, label="GraphSAGE")

plt.xlabel("Epoch")
plt.ylabel("Average AUC")
plt.title("Average AUC per Epoch for GAT, GCN, and GraphSAGE")
plt.legend()
plt.show()
