import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from sklearn.metrics import accuracy_score, precision_score
import matplotlib.pyplot as plt
from collections import defaultdict
from perseus.model.data_loader import get_data_loader

from perseus.model.new_data_loader import get_new_data_loader
from perseus.model.undirect_weighed_data_loader import get_undirect_weighted_data_loader

# Assuming your data loaders are defined as follows:
# from your_data_loader_module import get_new_data_loader, get_undirect_weighted_data_loader, get_data_loader
train_loader1, test_loader1 = get_new_data_loader()
train_loader2, test_loader2 = get_undirect_weighted_data_loader()
train_loader3, test_loader3 = get_data_loader()

# Define the model classes (GCN, GAT, GraphSAGE) as you provided in your code snippets


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
    # Your GAT model definition here

class GraphSAGE(nn.Module):
    # Your GraphSAGE model definition here

# Helper function to evaluate the model and compute metrics
@torch.no_grad()
def test(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)
        all_probs.append(torch.sigmoid(out).cpu())  # Convert logits to probabilities
        all_labels.append(data.y.cpu())
    accuracy, precision = compute_metrics(torch.cat(all_probs, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy())
    return accuracy, precision

def compute_metrics(probs, labels):
    preds = probs > 0.5
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    return accuracy, precision

def run_experiment(model, train_loader, test_loader, num_epochs, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    loss_op = torch.nn.BCEWithLogitsLoss()
    metrics = defaultdict(list)

    for epoch in range(num_epochs):
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data.x, data.edge_index)
            loss = loss_op(output, data.y.float())
            loss.backward()
            optimizer.step()

        accuracy, precision = test(model, test_loader, device)
        metrics['accuracy'].append(accuracy)
        metrics['precision'].append(precision)
        print(f"Epoch {epoch+1}/{num_epochs}, Accuracy: {accuracy}, Precision: {precision}")

    return metrics

def plot_metrics(models_metrics, title):
    plt.figure(figsize=(20, 7))
    for metric in ['accuracy', 'precision']:
        plt.subplot(1, 2, ('accuracy', 'precision').index(metric) + 1)
        for model_name, metrics in models_metrics.items():
            epochs = list(range(1, len(metrics[metric]) + 1))
            plt.plot(epochs, metrics[metric], label=model_name)
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} Comparison')
        plt.legend()
    plt.suptitle(title)
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your models
num_features = 2  # Adjust according to your dataset
num_classes = 3   # Adjust according to your dataset
models = {
    "GCN": GCN(num_features, 16, num_classes),
    "GAT": GAT(num_features, 16, num_classes),
    "GraphSAGE": GraphSAGE(num_features, 16, num_classes),
}

# Example of running an experiment with the first loader
loaders = [(train_loader1, test_loader1), (train_loader2, test_loader2), (train_loader3, test_loader3)]
loader_names = ['Loader 1', 'Loader 2', 'Loader 3']

for i, (train_loader, test_loader) in enumerate(loaders):
    models_metrics = {}
    for name, model in models.items():
        print(f"Running {name} experiment with {loader_names[i]}...")
        metrics = run_experiment(model, train_loader, test_loader, num_epochs=100, device=device)
        models_metrics[name] = metrics
    plot_metrics(models_metrics, f'GNN Model Comparison with {loader_names[i]}')
