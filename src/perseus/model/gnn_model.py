import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


# class GCNNet(torch.nn.Module):
#     def __init__(self, num_features, hidden_channels, num_classes):
#         super(GCNNet, self).__init__()
#         self.conv1 = GCNConv(num_features, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, num_classes)

#     def forward(self, x, edge_index, edge_weight=None):
#         x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
#         embeddings = F.dropout(x, training=self.training)
#         x = self.conv2(embeddings, edge_index)
#         x = torch.sigmoid(x)  # Sigmoid activation for binary classification

#         return x, embeddings


# class GCNNet(torch.nn.Module):
#     def __init__(self, num_features, hidden_channels, num_classes):
#         super(GCNNet, self).__init__()
#         self.conv1 = GCNConv(num_features, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, num_classes)

#         # Batch normalization layer to avoid over-smoothing
#         self.batch_norm1 = torch.nn.BatchNorm1d(hidden_channels)

#     def forward(self, x, edge_index, edge_weight=None):
#         # Ensure edge weights are non-negative (optional)
#         if edge_weight is not None:
#             edge_weight = F.relu(edge_weight)  # Ensure edge weights are non-negative

#         # First GCN layer with batch normalization
#         x = self.conv1(x, edge_index, edge_weight=edge_weight)
#         x = self.batch_norm1(x)  # Add batch normalization
#         x = F.relu(x)  # ReLU activation

#         # Dropout to prevent overfitting
#         embeddings = F.dropout(x, training=self.training, p=0.5)

#         # Second GCN layer
#         x = self.conv2(embeddings, edge_index)

#         # If binary classification, use sigmoid
#         # If multi-class classification, use softmax
#         x = torch.sigmoid(x)  # For binary classification
#         # x = torch.softmax(x, dim=1)  # Uncomment this for multi-class classification

#         return x, embeddings


class GCNNet(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

        # Add batch normalization to prevent over-smoothing
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_channels)
        self.batch_norm2 = torch.nn.BatchNorm1d(num_classes)

        # Introducing residual connection
        self.skip_connection = torch.nn.Linear(num_features, num_classes)

    def forward(self, x, edge_index, edge_weight=None):
        # Ensure edge weights are scaled appropriately
        if edge_weight is not None:
            edge_weight = torch.abs(edge_weight)  # Make sure edge weights are positive
            edge_weight = edge_weight / torch.max(edge_weight)  # Scale to [0, 1]

        # First GCN layer with batch normalization and LeakyReLU activation
        x1 = self.conv1(x, edge_index, edge_weight=edge_weight)
        x1 = self.batch_norm1(x1)
        x1 = F.leaky_relu(x1)  # Use Leaky ReLU to allow negative slopes

        # Dropout for regularization
        x1 = F.dropout(x1, training=self.training, p=0.5)

        # Second GCN layer
        x2 = self.conv2(x1, edge_index)
        x2 = self.batch_norm2(x2)

        # Skip connection (directly from input to output)
        skip_x = self.skip_connection(x)

        # Combine the GCN output with the skip connection
        x = x2 + skip_x

        # If binary classification, use sigmoid
        # For multi-class, switch to softmax
        x = torch.sigmoid(x)  # Binary classification
        # x = torch.softmax(x, dim=1)  # Uncomment this line for multi-class classification

        return x, x1


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
