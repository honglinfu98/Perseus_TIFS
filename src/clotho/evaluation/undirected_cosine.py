import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assuming get_new_data_loader is correctly defined elsewhere
# from clotho.model.new_data_loader import get_new_data_loader
from clotho.model.magamaga import get_data_loader


train_loader, test_loader, c = get_data_loader("COSS")


class GCN(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x


class GAT(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def test(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)
        all_probs.append(torch.sigmoid(out).cpu())  # Convert logits to probabilities
        all_labels.append(data.y.cpu())
    return torch.cat(all_probs, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy()


def compute_metrics(probs, labels):
    preds = probs > 0.5
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return accuracy, precision, recall, f1


def run_experiment(model, train_loader, test_loader, num_epochs=100):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    loss_op = torch.nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data.x, data.edge_index)
            loss = loss_op(output, data.y.float())
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

    probs, labels = test(model, test_loader)
    return compute_metrics(probs, labels)


# Define your dataset's specifics
num_features = 2
num_classes = 3

# Initialize models
models = {
    "GCN": GCN(num_features, 16, num_classes),
    "GAT": GAT(num_features, 16, num_classes),
    "GraphSAGE": GraphSAGE(num_features, 16, num_classes),
}

# Run experiments
for name, model in models.items():
    print(f"Running {name} experiment...")
    accuracy, precision, recall, f1 = run_experiment(
        model, train_loader, test_loader, num_epochs=100
    )
    print(
        f"{name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}\n"
    )
