import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

from os import path
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
# from clotho.model.data_loader import get_data_loader
from perseus.model.magamaga import get_data_loader






# Assuming get_data_loader is correctly defined elsewhere
# from your_custom_data_loader_file import get_data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your model classes (GCNNet, GraphSAGENet, Net for GAT, etc.) here as you have in your original script.
# GCN Implementation
class GCNNet(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# GraphSAGE Implementation
class GraphSAGENet(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GraphSAGENet, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_weight=None):
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


class GAT(torch.nn.Module):
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

# Define the run_experiment function here as you have in your original script.


def run_experiment(model,train_loader,test_loader, num_epochs=100):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    loss_op = torch.nn.BCEWithLogitsLoss()

    def train():
        for epoch in range(num_epochs):
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
            print(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader.dataset)}"
            )

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

    train()
    probs, labels = test(test_loader)
    probs = torch.cat(probs, dim=0).sigmoid().numpy()
    labels = torch.cat(labels, dim=0).numpy()

    # Compute ROC for each label and store
    fpr_dict, tpr_dict, thresholds_dict = {}, {}, {}
    for i in range(labels.shape[1]):  # Assuming labels is a 2D array: [samples, labels]
        fpr, tpr, thresholds = roc_curve(labels[:, i], probs[:, i])
        fpr_dict[i], tpr_dict[i], thresholds_dict[i] = fpr, tpr, thresholds

    return model, fpr_dict, tpr_dict, thresholds_dict


# Updated Metrics Calculation Functions
def calculate_accuracy(labels, preds):
    return accuracy_score(labels, preds)


def calculate_precision(labels, preds):
    return precision_score(labels, preds, average="macro", zero_division=0)


def calculate_recall(labels, preds):
    return recall_score(labels, preds, average="macro", zero_division=0)


def calculate_f1_score(labels, preds):
    return f1_score(labels, preds, average="macro", zero_division=0)


#     return accuracy, precision, recall, f1
@torch.no_grad()
def compute_metrics(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)
        all_probs.append(out.cpu())
        all_labels.append(data.y.cpu())
    probs = torch.cat(all_probs, dim=0).sigmoid().numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    preds = probs.argmax(axis=1)
    labels = labels.argmax(axis=1)

    # Calculating metrics
    accuracy = calculate_accuracy(labels, preds)
    precision = calculate_precision(labels, preds)
    recall = calculate_recall(labels, preds)
    f1 = calculate_f1_score(labels, preds)

    # Additional Metrics
    cm = confusion_matrix(labels, preds)
    specificity = np.mean(
        [
            cm[i][i] / (cm[i][i] + np.sum(cm[:, i]) - cm[i][i])
            for i in range(cm.shape[0])
            if np.sum(cm[:, i]) - cm[i][i] != 0
        ]
    )
    prevalence = np.mean([np.sum(cm[i]) / np.sum(cm) for i in range(cm.shape[0])])
    detection_rate = np.mean(
        [cm[i][i] / np.sum(cm[i]) for i in range(cm.shape[0]) if np.sum(cm[i]) != 0]
    )
    detection_prevalence = np.mean(
        [np.sum(cm[:, i]) / np.sum(cm) for i in range(cm.shape[0])]
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "prevalence": prevalence,
        "detection_rate": detection_rate,
        "detection_prevalence": detection_prevalence,
    }

# Initialize a structure to store results
results = {
    'DDINA': {},
    'COSS': {},
    'DDM': {}
}

datasets = ['DDINA', 'COSS', 'DDM']
models = ['GAT', 'GCN', 'GraphSAGE']

# Run experiments for each model
num_features = 2  # Set this according to your dataset
num_classes = 3  # Set this according to your dataset
hidden_channels = 16



for dataset in datasets:
    train_loader, test_loader, _ = get_data_loader(dataset)
    for model_name in models:
        if model_name == 'GAT':
            if dataset == 'DDINA':
                model = Net()
            else:
                num_features = 2
                model = GAT(num_features, hidden_channels, num_classes)
        elif model_name == 'GCN':
            if dataset == 'DDINA':
                num_features = 4
            else:
                num_features = 2
            model = GCNNet(num_features, hidden_channels, num_classes)
        elif model_name == 'GraphSAGE':
            if dataset == 'DDINA':
                num_features = 4
            else:
                num_features = 2
            model = GraphSAGENet(num_features, hidden_channels, num_classes)
        
        print(f"Running {model_name} Experiment on {dataset}")
        _, fpr, tpr, _ = run_experiment(model, train_loader, test_loader, num_epochs=100)
        print(model)
        metrics = compute_metrics(model, test_loader)
        
        # Store results
        results[dataset][model_name] = {
            'metrics': metrics,
            'fpr': fpr,
            'tpr': tpr
        }

# Example of accessing and printing metrics
for dataset in datasets:
    for model_name in models:
        metrics = results[dataset][model_name]['metrics']
        print(f"{model_name} Model Metrics on {dataset}:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value}")
        print("\n")

# Plotting ROC Curves for a specific label across datasets and models
label = 0  # Adjust this based on the label you're interested in
plt.figure(figsize=(10, 6))
for dataset in datasets:
    for model_name in models:
        fpr = results[dataset][model_name]['fpr'][label]
        tpr = results[dataset][model_name]['tpr'][label]
        plt.plot(fpr, tpr, label=f"{model_name} on {dataset}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves Comparison Across Datasets and Models")
plt.legend()
plt.show()
