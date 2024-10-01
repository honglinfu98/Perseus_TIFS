import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class GCNNet(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        embeddings = F.dropout(x, training=self.training)
        x = self.conv2(embeddings, edge_index)
        x = torch.sigmoid(x)  # Sigmoid activation for binary classification

        return x, embeddings


class GraphSAGENet(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GraphSAGENet, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index))
        embeddings = F.dropout(x, training=self.training)
        x = self.conv2(embeddings, edge_index)
        x = torch.sigmoid(x)  # Sigmoid for binary

        return x, embeddings


class Net(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=8):
        super(Net, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=2, concat=True)
        self.conv2 = GATConv(
            hidden_channels * 2, num_classes, heads=2, concat=False
        )  # Multi-head attention reduces to the number of classes

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        embeddings = F.dropout(x, training=self.training)
        x = self.conv2(embeddings, edge_index)
        x = torch.sigmoid(x)  # Sigmoid activation for binary classification
        return x, embeddings


# class GCNNet(torch.nn.Module):
#     def __init__(self, num_features, hidden_channels, num_classes):
#         super(GCNNet, self).__init__()
#         self.conv1 = GCNConv(num_features, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, num_classes)

#     def forward(self, x, edge_index, edge_weight=None):
#         x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
#         embeddings = x  # Save embeddings without dropout
#         x = self.conv2(embeddings, edge_index)
#         x = torch.sigmoid(x)  # Sigmoid activation for binary classification
#         return x, embeddings


# class GraphSAGENet(torch.nn.Module):
#     def __init__(self, num_features, hidden_channels, num_classes):
#         super(GraphSAGENet, self).__init__()
#         self.conv1 = SAGEConv(num_features, hidden_channels)
#         self.conv2 = SAGEConv(hidden_channels, num_classes)

#     def forward(self, x, edge_index, edge_weight=None):
#         x = F.relu(self.conv1(x, edge_index))
#         embeddings = x  # Save embeddings without dropout
#         x = self.conv2(embeddings, edge_index)
#         x = torch.sigmoid(x)  # Sigmoid for binary classification
#         return x, embeddings


# class Net(torch.nn.Module):
#     def __init__(self, num_features, num_classes, hidden_channels=8):
#         super(Net, self).__init__()
#         self.conv1 = GATConv(num_features, hidden_channels, heads=2, concat=True)
#         self.conv2 = GATConv(hidden_channels * 2, num_classes, heads=2, concat=False)

#     def forward(self, x, edge_index):
#         x = F.elu(self.conv1(x, edge_index))
#         embeddings = x  # Save embeddings without dropout
#         x = self.conv2(embeddings, edge_index)
#         x = torch.sigmoid(x)  # Sigmoid activation for binary classification
#         return x, embeddings


# class Net(torch.nn.Module):
#     def __init__(self, num_features, num_classes):
#         super().__init__()
#         self.conv1 = GATConv(num_features, 8, heads=2)
#         self.lin1 = torch.nn.Linear(num_features, 16)
#         self.conv2 = GATConv(16, 8, heads=2)
#         self.lin2 = torch.nn.Linear(16, 16)
#         self.conv3 = GATConv(16, num_classes, heads=2, concat=False)
#         self.lin3 = torch.nn.Linear(16, num_classes)

#     def forward(self, x, edge_index):
#         x1 = F.elu(self.conv1(x, edge_index) + self.lin1(x))
#         embeddings = F.elu(self.conv2(x1, edge_index) + self.lin2(x1))
#         x = self.conv3(embeddings, edge_index) + self.lin3(embeddings)
#         x = torch.sigmoid(x)  # Sigmoid for binary

#         return x, embeddings
