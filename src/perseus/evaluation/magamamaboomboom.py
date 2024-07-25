import time
import torch
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from sklearn.metrics import roc_curve, balanced_accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns
from perseus.settings import PROJECT_ROOT


from os import path
import torch
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

# from clotho.model.data_loader import get_data_loader
from perseus.model.magamaga import (
    get_data_pickle,
    split_data,
    get_train_test_validate_data,
    get_train_test_validate_data_pickle,
)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import random

# Set a seed value
seed = 724

# Python's `random` module
random.seed(seed)

# NumPy
np.random.seed(seed)

# PyTorch
torch.manual_seed(seed)


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
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        x = self.conv3(x, edge_index) + self.lin3(x)
        return x


# class GAT(torch.nn.Module):
#     def __init__(self, num_features, hidden_channels, num_classes):
#         super(GAT, self).__init__()
#         self.conv1 = GATConv(num_features, hidden_channels)
#         self.conv2 = GATConv(hidden_channels, num_classes)

#     def forward(self, x, edge_index, edge_weight=None):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)
#         return x


def run_experiment(model, train_loader, test_loader, num_epochs=100):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    loss_op = torch.nn.BCEWithLogitsLoss()

    train_times = []  # To store training times for each epoch

    def train():
        model.train()
        for epoch in range(num_epochs):
            start_time = time.time()  # Start timing the epoch
            total_loss = 0
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                output = model(data.x, data.edge_index)
                loss = loss_op(output, data.y.float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * data.num_graphs
            end_time = time.time()  # End timing the epoch
            epoch_time = end_time - start_time
            train_times.append(epoch_time)  # Store the time for this epoch
            print(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader.dataset)}, Time: {epoch_time:.2f}s"
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

    return model, fpr_dict, tpr_dict, thresholds_dict, train_times


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
    balanced_acc = balanced_accuracy_score(labels, preds)  # Calculate balanced accuracy

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
        "balanced_accuracy_score": balanced_acc,
        "specificity": specificity,
        "prevalence": prevalence,
        "detection_rate": detection_rate,
        "detection_prevalence": detection_prevalence,
    }


# Initialize a structure to store results
results = {"DDINA": {}, "COSS": {}, "DDM": {}}

datasets = ["DDINA", "COSS", "DDM"]
models = ["GAT", "GCN", "GraphSAGE"]

# Run experiments for each model
num_features = 2  # Set this according to your dataset
num_classes = 2  # Set this according to your dataset
hidden_channels = 8


# Comparison across different embedding for the same model
# Assuming 'results' contains your ROC curve data
models = ["GCN", "GAT", "GraphSAGE"]
dataset_colors = {"DDINA": "blue", "COSS": "green", "DDM": "red"}


# Non time dimension related experiments


for dataset in datasets:
    train_loader, test_loader, _ = split_data(dataset)
    for model_name in models:
        if model_name == "GAT":
            if dataset == "DDINA":
                num_features = 4
            else:
                num_features = 2
            model = Net(num_features, num_classes)
        elif model_name == "GCN":
            if dataset == "DDINA":
                num_features = 4
            else:
                num_features = 2
            model = GCNNet(num_features, hidden_channels, num_classes)
        elif model_name == "GraphSAGE":
            if dataset == "DDINA":
                num_features = 4
            else:
                num_features = 2
            model = GraphSAGENet(num_features, hidden_channels, num_classes)

        print(f"Running {model_name} Experiment on {dataset}")
        m, fpr, tpr, _, train_times = run_experiment(
            model, train_loader, test_loader, num_epochs=100
        )
        metrics = compute_metrics(model, test_loader)
        # Store results along with training times
        results[dataset][model_name] = {
            "model": m,
            "metrics": metrics,
            "fpr": fpr,
            "tpr": tpr,
            "train_times": train_times,  # Storing the training times
        }


# Example of accessing and printing metrics
for dataset in datasets:
    for model_name in models:
        metrics = results[dataset][model_name]["metrics"]
        print(f"{model_name} Model Metrics on {dataset}:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value}")
        print("\n")

# Plotting ROC Curves for a specific label across datasets and models


label = 1  # Adjust this based on the label you're interested in
plt.figure(figsize=(10, 6))
for dataset in datasets:
    for model_name in models:
        fpr = results[dataset][model_name]["fpr"][label]
        tpr = results[dataset][model_name]["tpr"][label]
        plt.plot(fpr, tpr, label=f"{model_name} on {dataset}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves Comparison Across Datasets and Models")
plt.legend()
plt.show()


# # Function to plot CDF
# def plot_cdf(data, ax, title):
#     for dataset, times in data.items():
#         sorted_times = np.sort(times)
#         cdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
#         ax.step(sorted_times, cdf, label=f'{dataset} CDF')
#     ax.set_xscale('log')
#     ax.set_xlabel('Time per Epoch (seconds)')
#     ax.set_ylabel('CDF')
#     ax.set_title(title)
#     ax.legend()
#     ax.grid(True)


def plot_cdf(data, ax, title, colors):
    for dataset, times in data.items():
        sorted_times = np.sort(times)
        cdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
        ax.step(
            sorted_times, cdf, label=f"{dataset} CDF", color=colors[dataset]
        )  # Use the colors dictionary for consistent colors
    ax.set_xscale("log")
    ax.set_xlabel("Time per Epoch (seconds)")
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)


# Example usage with your plots for GCN, GAT, and GraphSAGE
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Preparing data for GCN
GCN_train_times = {
    dataset: results[dataset]["GCN"]["train_times"] for dataset in datasets
}
plot_cdf(GCN_train_times, axs[0], "GCN Training Times", dataset_colors)

# Preparing data for GAT
GAT_train_times = {
    dataset: results[dataset]["GAT"]["train_times"] for dataset in datasets
}
plot_cdf(GAT_train_times, axs[1], "GAT Training Times", dataset_colors)

# Preparing data for GraphSAGE
GraphSAGE_train_times = {
    dataset: results[dataset]["GraphSAGE"]["train_times"] for dataset in datasets
}
plot_cdf(GraphSAGE_train_times, axs[2], "GraphSAGE Training Times", dataset_colors)

plt.tight_layout()
plt.show()


# Create a figure and a set of subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Loop through each model and plot its ROC curve on a different subplot
for i, model_name in enumerate(models):
    for dataset in datasets:
        fpr = results[dataset][model_name]["fpr"][label]
        tpr = results[dataset][model_name]["tpr"][label]
        axes[i].plot(fpr, tpr, label=f"{dataset}", color=dataset_colors[dataset])
    axes[i].set_title(f"ROC Curve of {model_name}")
    axes[i].set_xlabel("False Positive Rate")
    axes[i].set_ylabel("True Positive Rate")
    axes[i].legend(title="Dataset")

plt.tight_layout()
plt.show()


# Table for precision, accuracy, f1, recall
# Initialize an empty list to store your data
data = []
metrics_columns = sorted(results[datasets[0]][models[0]]["metrics"].keys())

# Loop through each dataset and model to gather metrics
for dataset in datasets:
    for model_name in models:
        # Check if the dataset and model_name keys exist in the results dictionary
        if dataset in results and model_name in results[dataset]:
            # Retrieve the stored metrics for the current dataset and model
            metrics = results[dataset][model_name]["metrics"]
            # Append a new record including the dataset, model, and all metrics
            data.append(
                [dataset, model_name] + [metrics[metric] for metric in sorted(metrics)]
            )
        else:
            # Handle the missing dataset/model combination by appending NaNs or placeholders
            data.append([dataset, model_name] + [None for _ in sorted(metrics_columns)])

# Assuming 'metrics_columns' is predefined or you define it based on your known metrics
columns = ["Dataset", "Model"] + metrics_columns
df = pd.DataFrame(data, columns=columns)

# Assuming 'df' is your original DataFrame

# Step 1: Keep only the specified columns
filtered_df = df[["Dataset", "Model", "accuracy", "f1", "precision", "recall"]]

# Step 2: Apply the .style.highlight_max() method
highlighted_df = filtered_df.style.highlight_max(
    subset=["accuracy", "f1", "precision", "recall"], color="yellow", axis=0
)

# Display the DataFrame with highlighted maximum values
highlighted_df


# Time dimension related experiments


results1 = {"DDINA": {}, "COSS": {}, "DDM": {}}


for dataset in datasets:
    train_loader, test_loader, _ = get_train_test_validate_data(dataset)
    for model_name in models:
        if model_name == "GAT":
            if dataset == "DDINA":
                num_features = 4
            else:
                num_features = 2
            model = Net(num_features, num_classes)
        elif model_name == "GCN":
            if dataset == "DDINA":
                num_features = 4
            else:
                num_features = 2
            model = GCNNet(num_features, hidden_channels, num_classes)
        elif model_name == "GraphSAGE":
            if dataset == "DDINA":
                num_features = 4
            else:
                num_features = 2
            model = GraphSAGENet(num_features, hidden_channels, num_classes)

        print(f"Running {model_name} Experiment on {dataset}")
        m, fpr, tpr, _, train_times = run_experiment(
            model, train_loader, test_loader, num_epochs=100
        )
        metrics = compute_metrics(model, test_loader)
        # Store results along with training times
        results1[dataset][model_name] = {
            "model": m,
            "metrics": metrics,
            "fpr": fpr,
            "tpr": tpr,
            "train_times": train_times,  # Storing the training times
        }


# Example of accessing and printing metrics
for dataset in datasets:
    for model_name in models:
        metrics = results1[dataset][model_name]["metrics"]
        print(f"{model_name} Model Metrics on {dataset}:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value}")
        print("\n")

# Plotting ROC Curves for a specific label across datasets and models


label = 1  # Adjust this based on the label you're interested in
plt.figure(figsize=(10, 6))
for dataset in datasets:
    for model_name in models:
        fpr = results1[dataset][model_name]["fpr"][label]
        tpr = results1[dataset][model_name]["tpr"][label]
        plt.plot(fpr, tpr, label=f"{model_name} on {dataset}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves Comparison Across Datasets and Models")
plt.legend()
plt.show()


def plot_cdf(data, ax, title, colors):
    for dataset, times in data.items():
        sorted_times = np.sort(times)
        cdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
        ax.step(
            sorted_times, cdf, label=f"{dataset} CDF", color=colors[dataset]
        )  # Use the colors dictionary for consistent colors
    ax.set_xscale("log")
    ax.set_xlabel("Time per Epoch (seconds)")
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)


# Example usage with your plots for GCN, GAT, and GraphSAGE
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Preparing data for GCN
GCN_train_times = {
    dataset: results1[dataset]["GCN"]["train_times"] for dataset in datasets
}
plot_cdf(GCN_train_times, axs[0], "GCN Training Times", dataset_colors)

# Preparing data for GAT
GAT_train_times = {
    dataset: results1[dataset]["GAT"]["train_times"] for dataset in datasets
}
plot_cdf(GAT_train_times, axs[1], "GAT Training Times", dataset_colors)

# Preparing data for GraphSAGE
GraphSAGE_train_times = {
    dataset: results1[dataset]["GraphSAGE"]["train_times"] for dataset in datasets
}
plot_cdf(GraphSAGE_train_times, axs[2], "GraphSAGE Training Times", dataset_colors)

plt.tight_layout()
plt.show()


# Comparison across different embedding for the same model
# Assuming 'results' contains your ROC curve data
datasets = ["DDINA", "COSS", "DDM"]
models = ["GCN", "GAT", "GraphSAGE"]
dataset_colors = {"DDINA": "blue", "COSS": "green", "DDM": "red"}


# Create a figure and a set of subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Loop through each model and plot its ROC curve on a different subplot
for i, model_name in enumerate(models):
    for dataset in datasets:
        fpr = results1[dataset][model_name]["fpr"][label]
        tpr = results1[dataset][model_name]["tpr"][label]
        axes[i].plot(fpr, tpr, label=f"{dataset}", color=dataset_colors[dataset])
    axes[i].set_title(f"ROC Curve of {model_name}")
    axes[i].set_xlabel("False Positive Rate")
    axes[i].set_ylabel("True Positive Rate")
    axes[i].legend(title="Dataset")

plt.tight_layout()
plt.show()


# Table for precision, accuracy, f1, recall
# Initialize an empty list to store your data
data = []
metrics_columns = sorted(results1[datasets[0]][models[0]]["metrics"].keys())

# Loop through each dataset and model to gather metrics
for dataset in datasets:
    for model_name in models:
        # Check if the dataset and model_name keys exist in the results dictionary
        if dataset in results1 and model_name in results1[dataset]:
            # Retrieve the stored metrics for the current dataset and model
            metrics = results1[dataset][model_name]["metrics"]
            # Append a new record including the dataset, model, and all metrics
            data.append(
                [dataset, model_name] + [metrics[metric] for metric in sorted(metrics)]
            )
        else:
            # Handle the missing dataset/model combination by appending NaNs or placeholders
            data.append([dataset, model_name] + [None for _ in sorted(metrics_columns)])

# Assuming 'metrics_columns' is predefined or you define it based on your known metrics
columns = ["Dataset", "Model"] + metrics_columns
df = pd.DataFrame(data, columns=columns)

# Assuming 'df' is your original DataFrame

# Step 1: Keep only the specified columns
filtered_df = df[["Dataset", "Model", "accuracy", "f1", "precision", "recall"]]

# Step 2: Apply the .style.highlight_max() method
highlighted_df1 = filtered_df.style.highlight_max(
    subset=["accuracy", "f1", "precision", "recall"], color="yellow", axis=0
)

# Display the DataFrame with highlighted maximum values
highlighted_df1


# import matplotlib.pyplot as plt

label = 1  # Adjust this based on the label you're interested in

# Set up the figure and subplots
plt.figure(figsize=(10, 12))  # Increase the height to accommodate two plots vertically

# First subplot
plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
for dataset in datasets:
    for model_name in models:
        fpr = results[dataset][model_name]["fpr"][label]
        tpr = results[dataset][model_name]["tpr"][label]
        plt.plot(fpr, tpr, label=f"{model_name} on {dataset}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves Comparison Across Datasets and Models (Results)")
plt.legend()

# Second subplot
plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
for dataset in datasets:
    for model_name in models:
        fpr = results1[dataset][model_name]["fpr"][label]
        tpr = results1[dataset][model_name]["tpr"][label]
        plt.plot(fpr, tpr, label=f"{model_name} on {dataset}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves Comparison Across Datasets and Models (Results1)")
plt.legend()

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


# Extract FPR and TPR for a specific model and task
fpr = results["DDINA"]["GAT"]["fpr"][1]
tpr = results["DDINA"]["GAT"]["tpr"][1]

# # Calculate AUC
# auc = metrics.auc(fpr, tpr)
# print(f"AUC: {auc}")


# def plot_cdf(data, ax, title, colors):
#     label_map = {
#         'DDINA': 'Directed DANI',
#         'COSS': 'Cosine Similarity',
#         'DDM': 'Weighted DANI'
#     }
#     for dataset, times in data.items():
#         sorted_times = np.sort(times)
#         cdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
#         dataset_label = label_map.get(dataset, dataset)  # Use mapped label if exists, otherwise use dataset name

#         ax.step(sorted_times, cdf, label=f'{dataset_label}', color=colors[dataset])

#     ax.set_xscale('log')

#     # Calculate the data range to determine tick settings
#     data_min, data_max = np.min([times for dataset, times in data.items() for times in times]), \
#                          np.max([times for dataset, times in data.items() for times in times])

#     # Adjust the number of ticks based on the data range
#     if data_max / data_min > 1000:
#         numticks = 3
#     else:
#         numticks = 5

#     # Set major locator and formatter for the log scale
#     ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs='all', numticks=numticks))
#     ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.3f}' if x < 1 else f'{int(x)}'))

#     # Set tick parameters and rotate x-ticks for better readability
#     ax.tick_params(axis='x', which='major', labelrotation=45)

#     ax.set_xlabel('Time per Epoch (seconds)')
#     ax.set_ylabel('CDF')
#     ax.set_title(title)
#     ax.legend()


# def create_plots(fontsize_title, fontsize_labels, fontsize_legend, fontsize_ticks):
#     # Create the main figure with desired size
#     fig = plt.figure(figsize=(36, 15))

#     # Create subfigures for each of the sections (a to d)
#     subfigs = fig.subfigures(2, 2, wspace=0.07, hspace=0.07)

#     # Helper function to set the styles
#     def set_style(ax, show_title=True, show_xlabel=True, show_ylabel=True):
#         if show_title:
#             ax.set_title(ax.get_title(), fontsize=fontsize_title)
#         else:
#             ax.set_title('')
#         if show_xlabel:
#             ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize_labels)
#         else:
#             ax.set_xlabel('')
#         if show_ylabel:
#             ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize_labels)
#         else:
#             ax.set_ylabel('')
#         ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
#         if ax.get_legend():
#             ax.legend(title=ax.get_legend().get_title().get_text(), fontsize=fontsize_legend, title_fontsize=fontsize_legend)

#     # Configure each subfigure
#     # Subfigure a (top left)
#     axs_a = subfigs[0, 0].subplots(1, 3)
#     GCN_train_times = {dataset: results[dataset]['GCN']['train_times'] for dataset in datasets}
#     GAT_train_times = {dataset: results[dataset]['GAT']['train_times'] for dataset in datasets}
#     GraphSAGE_train_times = {dataset: results[dataset]['GraphSAGE']['train_times'] for dataset in datasets}

#     plot_cdf(GCN_train_times, axs_a[0], 'GCN', dataset_colors)
#     plot_cdf(GAT_train_times, axs_a[1], 'GAT', dataset_colors)
#     plot_cdf(GraphSAGE_train_times, axs_a[2], 'GraphSAGE', dataset_colors)


#     set_style(axs_a[0], show_title=True, show_xlabel=False, show_ylabel=True)  # Only leftmost y-label
#     for ax in axs_a[1:]:
#         set_style(ax, show_title=True, show_xlabel=False, show_ylabel=False)  # No y-label

#     # Subfigure c (top right) - Adjusted to use all models
#     axs_c = subfigs[0, 1].subplots(1, len(models))  # Ensure the subplot layout matches the number of models
#     for i, model_name in enumerate(models):
#         print(model_name)

#         plot_fpr_tpr(results, model_name, axs_c[i], datasets, label, dataset_colors,
#                      title=f"{model_name}", xlabel="False Positive Rate", ylabel="True Positive Rate")
#         set_style(axs_c[i], show_title=True, show_xlabel=False, show_ylabel=(i == 0))


#     # Subfigure b (bottom left)
#     axs_b = subfigs[1, 0].subplots(1, 3)
#     GCN_train_times = {dataset: results1[dataset]['GCN']['train_times'] for dataset in datasets}
#     GAT_train_times = {dataset: results1[dataset]['GAT']['train_times'] for dataset in datasets}
#     GraphSAGE_train_times = {dataset: results1[dataset]['GraphSAGE']['train_times'] for dataset in datasets}

#     plot_cdf(GCN_train_times, axs_b[0], 'GCN', dataset_colors)
#     plot_cdf(GAT_train_times, axs_b[1], 'GAT', dataset_colors)
#     plot_cdf(GraphSAGE_train_times, axs_b[2], 'GraphSAGE', dataset_colors)


#     set_style(axs_b[0], show_title=False, show_xlabel=True, show_ylabel=True)  # Only leftmost y-label
#     for ax in axs_b[1:]:
#         set_style(ax, show_title=False, show_xlabel=True, show_ylabel=False)  # No y-label

#     # Subfigure d (bottom right) - Adjusted to use all models
#     axs_d = subfigs[1, 1].subplots(1, len(models))  # Ensure the subplot layout matches the number of models
#     for i, model_name in enumerate(models):
#         print(model_name)
#         plot_fpr_tpr(results1, model_name, axs_d[i], datasets, label, dataset_colors,
#                      title=f"{model_name}", xlabel="False Positive Rate", ylabel="True Positive Rate")
#         set_style(axs_d[i], show_title=False, show_xlabel=True, show_ylabel=(i == 0))

#     # Display the figure
#     # plt.tight_layout()  # Adjust layout

#     plt.savefig(path.join(PROJECT_ROOT, "data", 'comparisons.pdf'))
#     plt.show()

# def plot_fpr_tpr(results, model_name, ax, datasets, label, dataset_colors, title, xlabel, ylabel):
#     # Label mapping dictionary
#     label_map = {
#         'DDINA': 'Directed DANI',
#         'COSS': 'Cosine Similarity',
#         'DDM': 'Weighted DANI'
#     }

#     for dataset in datasets:
#         fpr = results[dataset][model_name]['fpr'][label]
#         tpr = results[dataset][model_name]['tpr'][label]

#         # Use mapped label if exists, otherwise use dataset name
#         dataset_label = label_map.get(dataset, dataset)

#         ax.plot(fpr, tpr, label=f"{dataset_label}", color=dataset_colors[dataset])

#     ax.set_title(title)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.legend()


# # Example usage:
# create_plots(fontsize_title=30, fontsize_labels=25, fontsize_legend=20, fontsize_ticks=20)


def get_global_min_max(results, results1, method):
    all_times = []
    for res in [results, results1]:
        for dataset in datasets:
            all_times.extend(res[dataset][method]["train_times"])
    return min(all_times), max(all_times)


global_min_max = {}
for method in ["GCN", "GAT", "GraphSAGE"]:
    global_min_max[method] = get_global_min_max(results, results1, method)


def plot_cdf(data, ax, title, colors, global_min, global_max):
    label_map = {
        "DDINA": "Directed DANI",
        "COSS": "Cosine Similarity",
        "DDM": "Weighted DANI",
    }
    for dataset, times in data.items():
        sorted_times = np.sort(times)
        cdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
        dataset_label = label_map.get(dataset, dataset)

        ax.step(sorted_times, cdf, label=f"{dataset_label}", color=colors[dataset])

    ax.set_xscale("log")
    ax.set_xlim(
        global_min, global_max
    )  # Set consistent x-limits based on global min and max

    # Set consistent number of ticks across plots
    if global_max / global_min > 1000:
        numticks = 3
    else:
        numticks = 5

    ax.xaxis.set_major_locator(
        ticker.LogLocator(base=10.0, subs="all", numticks=numticks)
    )
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{x:.3f}" if x < 1 else f"{int(x)}")
    )

    ax.tick_params(axis="x", which="major", labelrotation=45)

    ax.set_xlabel("Time per Epoch (seconds)")
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.legend()


def create_plots(fontsize_title, fontsize_labels, fontsize_legend, fontsize_ticks):
    # Create the main figure with desired size
    fig = plt.figure(figsize=(36, 15))

    # Create subfigures for each of the sections (a to d)
    subfigs = fig.subfigures(2, 2, wspace=0.07, hspace=0.07)

    # Helper function to set the styles
    def set_style(ax, show_title=True, show_xlabel=True, show_ylabel=True):
        if show_title:
            ax.set_title(ax.get_title(), fontsize=fontsize_title)
        else:
            ax.set_title("")
        if show_xlabel:
            ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize_labels)
        else:
            ax.set_xlabel("")
        if show_ylabel:
            ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize_labels)
        else:
            ax.set_ylabel("")
        ax.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
        if ax.get_legend():
            ax.legend(
                title=ax.get_legend().get_title().get_text(),
                fontsize=fontsize_legend,
                title_fontsize=fontsize_legend,
            )

    # Configure each subfigure
    # Subfigure a (top left)
    axs_a = subfigs[0, 0].subplots(1, 3)
    for j, method in enumerate(["GCN", "GAT", "GraphSAGE"]):
        method_train_times = {
            dataset: results[dataset][method]["train_times"] for dataset in datasets
        }
        plot_cdf(
            method_train_times,
            axs_a[j],
            method,
            dataset_colors,
            *global_min_max[method],
        )

    set_style(
        axs_a[0], show_title=True, show_xlabel=False, show_ylabel=True
    )  # Only leftmost y-label
    for ax in axs_a[1:]:
        set_style(
            ax, show_title=True, show_xlabel=False, show_ylabel=False
        )  # No y-label

    # Subfigure c (top right) - Adjusted to use all models
    axs_c = subfigs[0, 1].subplots(
        1, len(models)
    )  # Ensure the subplot layout matches the number of models
    for i, model_name in enumerate(models):
        print(model_name)

        plot_fpr_tpr(
            results,
            model_name,
            axs_c[i],
            datasets,
            label,
            dataset_colors,
            title=f"{model_name}",
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
        )
        set_style(axs_c[i], show_title=True, show_xlabel=False, show_ylabel=(i == 0))

    # Subfigure b (bottom left)
    axs_b = subfigs[1, 0].subplots(1, 3)
    for j, method in enumerate(["GCN", "GAT", "GraphSAGE"]):
        method_train_times = {
            dataset: results1[dataset][method]["train_times"] for dataset in datasets
        }
        plot_cdf(
            method_train_times,
            axs_b[j],
            method,
            dataset_colors,
            *global_min_max[method],
        )

    set_style(
        axs_b[0], show_title=False, show_xlabel=True, show_ylabel=True
    )  # Only leftmost y-label
    for ax in axs_b[1:]:
        set_style(
            ax, show_title=False, show_xlabel=True, show_ylabel=False
        )  # No y-label

    # Subfigure d (bottom right) - Adjusted to use all models
    axs_d = subfigs[1, 1].subplots(
        1, len(models)
    )  # Ensure the subplot layout matches the number of models
    for i, model_name in enumerate(models):
        print(model_name)
        plot_fpr_tpr(
            results1,
            model_name,
            axs_d[i],
            datasets,
            label,
            dataset_colors,
            title=f"{model_name}",
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
        )
        set_style(axs_d[i], show_title=False, show_xlabel=True, show_ylabel=(i == 0))

    # Display the figure
    # plt.tight_layout()  # Adjust layout

    plt.savefig(path.join(PROJECT_ROOT, "data", "comparisons.pdf"))
    plt.show()


def plot_fpr_tpr(
    results, model_name, ax, datasets, label, dataset_colors, title, xlabel, ylabel
):
    # Label mapping dictionary
    label_map = {
        "DDINA": "Directed DANI",
        "COSS": "Cosine Similarity",
        "DDM": "Weighted DANI",
    }

    for dataset in datasets:
        fpr = results[dataset][model_name]["fpr"][label]
        tpr = results[dataset][model_name]["tpr"][label]

        # Use mapped label if exists, otherwise use dataset name
        dataset_label = label_map.get(dataset, dataset)

        ax.plot(fpr, tpr, label=f"{dataset_label}", color=dataset_colors[dataset])

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()


create_plots(
    fontsize_title=30, fontsize_labels=25, fontsize_legend=20, fontsize_ticks=20
)
