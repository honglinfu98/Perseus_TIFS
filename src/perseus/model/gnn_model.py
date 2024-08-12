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
        return x, embeddings  # Return both output and embeddings


class GraphSAGENet(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GraphSAGENet, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index))
        embeddings = F.dropout(x, training=self.training)
        x = self.conv2(embeddings, edge_index)
        return x, embeddings  # Return both output and embeddings


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
        x1 = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        embeddings = F.elu(self.conv2(x1, edge_index) + self.lin2(x1))
        x = self.conv3(embeddings, edge_index) + self.lin3(embeddings)
        return x, embeddings  # Return both output and embeddings
