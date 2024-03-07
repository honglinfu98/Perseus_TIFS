import time
import pandas as pd 
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, ReLU, Dropout
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

from perseus.model.magamaga import get_data_pickle
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Modified GCNNet with variable layers
class GCNNet(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, num_layers=2):
        super(GCNNet, self).__init__()
        self.layers = ModuleList()
        self.layers.append(GCNConv(num_features, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))
        self.layers.append(GCNConv(hidden_channels, num_classes))
        self.dropout = Dropout(0.5)

    def forward(self, x, edge_index, edge_weight=None):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x, edge_index, edge_weight=edge_weight))
            x = self.dropout(x)
        x = self.layers[-1](x, edge_index)
        return x

# Modified GraphSAGENet with variable layers
class GraphSAGENet(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, num_layers=2):
        super(GraphSAGENet, self).__init__()
        self.layers = ModuleList()
        self.layers.append(SAGEConv(num_features, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_channels, hidden_channels))
        self.layers.append(SAGEConv(hidden_channels, num_classes))
        self.dropout = Dropout(0.5)

    def forward(self, x, edge_index, edge_weight=None):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x, edge_index))
            x = self.dropout(x)
        x = self.layers[-1](x, edge_index)
        return x

# Modified Net with a more flexible structure for GAT layers
class Net(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, num_layers=2):
        super().__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=2)
        self.lin1 = Linear(num_features, 2 * hidden_channels)

        self.middle_convs = ModuleList()
        self.middle_lins = ModuleList()
        for _ in range(num_layers - 2):
            self.middle_convs.append(GATConv(2 * hidden_channels, hidden_channels, heads=2))
            self.middle_lins.append(Linear(2 * hidden_channels, 2 * hidden_channels))

        self.conv_last = GATConv(2 * hidden_channels, num_classes, heads=2, concat=False)
        self.lin_last = Linear(2 * hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        for conv, lin in zip(self.middle_convs, self.middle_lins):
            x = F.elu(conv(x, edge_index) + lin(x))
        x = self.conv_last(x, edge_index) + self.lin_last(x)
        return x

# Modified GAT with variable layers
class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, num_layers=2):
        super(GAT, self).__init__()
        self.initial_conv = GATConv(num_features, hidden_channels, heads=2)
        self.final_conv = GATConv(hidden_channels * 2, num_classes, heads=1, concat=False)

        self.middle_convs = ModuleList()
        for _ in range(num_layers - 2):
            self.middle_convs.append(GATConv(hidden_channels * 2, hidden_channels, heads=2))
        
        self.dropout = Dropout(0.5)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.initial_conv(x, edge_index))
        x = self.dropout(x)
        for conv in self.middle_convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        x = self.final_conv(x, edge_index)
        return x
    


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



    
datasets = ['DDINA', 'COSS', 'DDM']
models = ['GAT', 'GCNNet', 'GraphSAGENet']  # Ensure these match your class names exactly

def run_experiments():
    learning_rates = [1e-4, 5e-4, 1e-3]
    hidden_channels_list = [16, 32, 64]
    num_layers_options = [2, 3, 4]
    results = []

    for dataset_name in datasets:
        train_loader, test_loader, _ = get_data_pickle(dataset_name)
        dataset_results = {}  # Dictionary for storing results per dataset

        for model_name in models:
            # Dynamic setting of num_features and num_classes based on dataset
            if dataset_name == 'DDINA':
                num_features, num_classes = 4, 3
            else:
                num_features, num_classes = 2, 3

            for lr in learning_rates:
                for hidden_channels in hidden_channels_list:
                    for num_layers in num_layers_options:
                        print(f"Running {model_name} Experiment on {dataset_name} with lr={lr}, hidden_channels={hidden_channels}, num_layers={num_layers}")
                        model_class = globals()[model_name]
                        model = model_class(num_features, hidden_channels, num_classes, num_layers=num_layers).to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                        model, fpr, tpr, thresholds, train_times = run_experiment(model, train_loader, test_loader, optimizer, num_epochs=3)
                        metrics = compute_metrics(model, test_loader)
                        # Store results for this configuration
                        config_key = f"{model_name}_lr{lr}_h{hidden_channels}_layers{num_layers}"
                        dataset_results[config_key] = {
                            'metrics': metrics,
                            'fpr': fpr,
                            'tpr': tpr,
                            'train_times': train_times
                        }

        results.append(dataset_results)

    return results

def run_experiment(model, train_loader, test_loader, optimizer, num_epochs=3):
    model = model.to(device)
    loss_op = torch.nn.BCEWithLogitsLoss()
    train_times = []

    def train():
        model.train()
        for epoch in range(num_epochs):
            start_time = time.time()
            total_loss = 0
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                output = model(data.x, data.edge_index)
                loss = loss_op(output, data.y.float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * data.num_graphs
            end_time = time.time()
            train_times.append(end_time - start_time)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader.dataset)}, Time: {train_times[-1]:.2f}s")

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
    for i in range(labels.shape[1]):
        fpr, tpr, thresholds = roc_curve(labels[:, i], probs[:, i])
        fpr_dict[i], tpr_dict[i], thresholds_dict[i] = fpr, tpr, thresholds

    return model, fpr_dict, tpr_dict, thresholds_dict, train_times



results = run_experiments()

# # Convert results to DataFrame for easier analysis
# all_results = []
# for dataset, configs in results.items():
#     for config, res in configs.items():
#         res['config'] = config
#         res['dataset'] = dataset
#         all_results.append(res)

# results_df = pd.DataFrame(all_results)
# print(results_df)

# # Optionally, save the results to a CSV file
# results_df.to_csv("hyperparameter_tuning_results.csv", index=False)