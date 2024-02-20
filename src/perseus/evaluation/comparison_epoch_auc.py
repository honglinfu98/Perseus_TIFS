from os import path
import pickle
import torch
import random
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from perseus.model.data_loader import get_data_loader
from perseus.model.new_data_loader import get_new_data_loader

train_loader, test_loader = get_data_loader()


# def sample_one_degree_ego_network(edge_index, num_samples=10):
#     ego_networks = {}
#     for source_node, target_node in edge_index.t():
#         if source_node.item() not in ego_networks:
#             ego_networks[source_node.item()] = set()
#         ego_networks[source_node.item()].add(target_node.item())
#     for node in ego_networks:
#         ego_networks[node] = list(ego_networks[node])
#     return ego_networks
def sample_ego_network(edge_index, num_hops=1, num_neighbors=10):
    # Initialize ego networks dictionary
    ego_networks = {node.item(): set() for node in edge_index.unique()}

    # Function to add neighbors
    def add_neighbors(node, current_hop):
        if current_hop > num_hops:
            return
        neighbors = edge_index[1][edge_index[0] == node]
        # Sample a fixed number of neighbors if there are too many
        if len(neighbors) > num_neighbors:
            neighbors = neighbors[torch.randperm(len(neighbors))[:num_neighbors]]
        ego_networks[node].update(neighbors.tolist())
        for neighbor in neighbors:
            add_neighbors(neighbor.item(), current_hop + 1)

    # Build ego networks
    for node in ego_networks:
        add_neighbors(node, 1)

    # Convert sets to lists
    for node in ego_networks:
        ego_networks[node] = list(ego_networks[node])

    return ego_networks


class CustomGraphSAGENet(torch.nn.Module):
    def __init__(self, num_features, num_classes, aggregation_type="max"):
        super(CustomGraphSAGENet, self).__init__()
        self.conv1 = SAGEConv(num_features, 16, aggr=aggregation_type)
        self.conv2 = SAGEConv(16, num_classes, aggr=aggregation_type)


class CustomGraphSAGENet(torch.nn.Module):
    # [Constructor remains unchanged]

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        ego_networks = sample_ego_network(edge_index, num_hops=2, num_neighbors=10)
        aggregated_features = torch.zeros_like(x)
        for node, neighbors in ego_networks.items():
            neighbors_tensor = torch.tensor(
                neighbors, dtype=torch.long, device=x.device
            )
            aggregated_features[node] = x[neighbors_tensor].mean(dim=0)
        return aggregated_features


class GCNNet(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


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
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        x = self.conv3(x, edge_index) + self.lin3(x)
        return x


# new_train_loader, new_test_loader = get_new_data_loader()


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

    average_auc_scores = []
    for epoch in range(1, 101):
        train()
        probs, labels = test(test_loader)
        probs = torch.cat(probs, dim=0).sigmoid().numpy()
        labels = torch.cat(labels, dim=0).numpy()
        auc_scores = []
        for i in range(labels.shape[1]):
            fpr, tpr, _ = roc_curve(labels[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            auc_scores.append(roc_auc)
        average_auc = np.mean(auc_scores)
        average_auc_scores.append(average_auc)
        print(f"Epoch: {epoch:03d}, Average AUC: {average_auc:.4f}")

    plt.plot(range(1, 101), average_auc_scores, label="Average AUC")
    plt.xlabel("Epoch")
    plt.ylabel("Average AUC")
    plt.title("Average AUC per Epoch")
    plt.legend()
    plt.show()
    return average_auc_scores


num_features = 4  # Adjust this based on your dataset
num_classes = 3  # Adjust this based on your dataset

print("Running GAT Experiment")
gat_auc_scores = run_experiment(Net())

print("\nRunning GCN Experiment")
gcn_auc_scores = run_experiment(GCNNet(num_features, num_classes))

print("\nRunning GraphSAGE Experiment")
graphsage_auc_scores = run_experiment(GraphSAGENet(num_features, num_classes))

print("\nRunning Custom GraphSAGE Experiment")
custom_graphsage_auc_scores = run_experiment(
    CustomGraphSAGENet(num_features, num_classes, aggregation_type="max")
)

plt.figure(figsize=(10, 6))
epochs = range(1, 101)
plt.plot(epochs, gat_auc_scores, label="GAT")
plt.plot(epochs, gcn_auc_scores, label="GCN")
plt.plot(epochs, graphsage_auc_scores, label="GraphSAGE")
plt.plot(epochs, custom_graphsage_auc_scores, label="Custom GraphSAGE")
plt.xlabel("Epoch")
plt.ylabel("Average AUC")
plt.title("Average AUC per Epoch for GAT, GCN, GraphSAGE, and Custom GraphSAGE")
plt.legend()
plt.show()
