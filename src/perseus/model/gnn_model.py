import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Dropout, Linear
from torch_geometric.nn import SAGEConv, global_max_pool


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


class SageEncode(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, embedding_size):
        super(SageEncode, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, embedding_size)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        embeddings = F.dropout(x, training=self.training)
        out = self.conv2(embeddings, edge_index)
        return out, embeddings


class GraphClassifierWithMaxPooling(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, embedding_size):
        super(GraphClassifierWithMaxPooling, self).__init__()

        self.encoder = SageEncode(num_features, hidden_channels, embedding_size)
        self.classifier = Linear(embedding_size, 1)

    def forward(self, batch):
        # batch.x:   [total_nodes,  num_features]
        # batch.batch: [total_nodes] → graph-idx for each node
        out, embeddings = self.encoder(batch.x, batch.edge_index)
        # global_max_pool will now return [batch.num_graphs, embedding_size]
        graph_repr = global_max_pool(embeddings, batch.batch)
        logits = self.classifier(graph_repr)  # → [batch.num_graphs, 1]
        return torch.sigmoid(logits).view(-1)  # → [batch.num_graphs]
