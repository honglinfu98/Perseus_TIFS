"""
This script is used to tune the parameters on the models and datasets. It is used to run the experiments for the empirical study and evaluate the performance of the models on the test set.
"""

from os import path
import time
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, Dropout
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
from perseus.settings import PROJECT_ROOT
from perseus.dataset.dataset_preparation import get_split_data_pickle, split_data


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
            self.middle_convs.append(
                GATConv(2 * hidden_channels, hidden_channels, heads=2)
            )
            self.middle_lins.append(Linear(2 * hidden_channels, 2 * hidden_channels))

        self.conv_last = GATConv(
            2 * hidden_channels, num_classes, heads=2, concat=False
        )
        self.lin_last = Linear(2 * hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        for conv, lin in zip(self.middle_convs, self.middle_lins):
            x = F.elu(conv(x, edge_index) + lin(x))
        x = self.conv_last(x, edge_index) + self.lin_last(x)
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
    """
    Compute the evaluation metrics for the model on the given
    """
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


datasets = ["DDINA", "COSS", "DDM"]
models = [
    "Net",
    "GCNNet",
    "GraphSAGENet",
]  # Ensure these match your class names exactly


def run_experiments():
    """
    Run experiments for all models on all datasets with different
    """
    learning_rates = [1e-5, 5e-5, 1e-4, 5e-4]
    hidden_channels_list = [8, 16, 32, 64]
    num_layers_options = [2, 3, 4, 5]

    results = []

    for dataset_name in datasets:
        train_loader, test_loader, _ = get_split_data_pickle(dataset_name)
        dataset_results = {}  # Dictionary for storing results per dataset

        for model_name in models:
            # Dynamic setting of num_features and num_classes based on dataset
            if dataset_name == "DDINA":
                num_features, num_classes = 4, 2
            else:
                num_features, num_classes = 2, 2

            for lr in learning_rates:
                for hidden_channels in hidden_channels_list:
                    for num_layers in num_layers_options:
                        print(
                            f"Running {model_name} Experiment on {dataset_name} with lr={lr}, hidden_channels={hidden_channels}, num_layers={num_layers}"
                        )
                        model_class = globals()[model_name]
                        model = model_class(
                            num_features,
                            hidden_channels,
                            num_classes,
                            num_layers=num_layers,
                        ).to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                        model, fpr, tpr, thresholds, train_times = run_experiment(
                            model, train_loader, test_loader, optimizer, num_epochs=100
                        )
                        metrics = compute_metrics(model, test_loader)
                        # Store results for this configuration
                        config_key = (
                            f"{model_name}_lr{lr}_h{hidden_channels}_layers{num_layers}"
                        )
                        dataset_results[config_key] = {
                            "metrics": metrics,
                            "fpr": fpr,
                            "tpr": tpr,
                            "train_times": train_times,
                        }

        results.append(dataset_results)

    return results


def run_experiment(model, train_loader, test_loader, optimizer, num_epochs=100):
    """
    Custom training loop for the model on the given dataset.
    """
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
            print(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader.dataset)}, Time: {train_times[-1]:.2f}s"
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
    for i in range(labels.shape[1]):
        fpr, tpr, thresholds = roc_curve(labels[:, i], probs[:, i])
        fpr_dict[i], tpr_dict[i], thresholds_dict[i] = fpr, tpr, thresholds

    return model, fpr_dict, tpr_dict, thresholds_dict, train_times


# Function to calculate AUC for each model configuration
def calculate_auc_for_models(dataset_results):
    """
    Calculate AUC for each model configuration in the dataset results.
    """
    auc_results = {}
    for model in models:
        for lr in learning_rates:
            for h in hidden_channels_list:
                for layers in num_layers_options:
                    model_name = f"{model}_lr{lr}_h{h}_layers{layers}"
                    if model_name in dataset_results:
                        if (
                            1 in dataset_results[model_name]["fpr"]
                            and 1 in dataset_results[model_name]["tpr"]
                        ):
                            fpr = dataset_results[model_name]["fpr"][1]
                            tpr = dataset_results[model_name]["tpr"][1]
                            auc_score = auc(fpr, tpr)
                            auc_results[model_name] = auc_score
    return auc_results


# Function to print sorted AUC results
def print_sorted_auc_results(dataset_name, auc_results):
    """
    Print the sorted AUC results for the given dataset.
    """
    print(f"Sorted AUC Results for {dataset_name}:")
    sorted_auc = sorted(auc_results.items(), key=lambda item: item[1], reverse=True)
    for model_name, auc_score in sorted_auc:
        print(f"{model_name}: {auc_score:.4f}")


# Function to convert AUC results dictionary into a DataFrame
def convert_to_df(auc_results):
    """
    Convert the AUC results dictionary into a DataFrame.
    """
    data = []
    for model_name, auc_score in auc_results.items():
        parts = model_name.split("_")
        model, lr, h, layers = parts[0], parts[1][2:], parts[2][1:], parts[3][6:]
        data.append(
            {
                "Model": model,
                "Learning Rate": lr,
                "Hidden Channels": h,
                "Layers": layers,
                "AUC Score": auc_score,
            }
        )
    df = pd.DataFrame(data)
    df["Learning Rate"] = pd.to_numeric(df["Learning Rate"])
    df["Hidden Channels"] = pd.to_numeric(df["Hidden Channels"])
    df["Layers"] = pd.to_numeric(df["Layers"])
    return df


if __name__ == "__main__":

    with open(path.join(PROJECT_ROOT, "data", "parameter_results.pkl"), "rb") as file:
        results = pickle.load(file)

    title_fontsize = 35
    label_fontsize = 35
    tick_fontsize = 30

    datasets_dict = {
        "DDINA": results[0],
        "COSS": results[1],
        "DDM": results[2],
    }

    learning_rates = [1e-5, 5e-5, 1e-4, 5e-4]
    hidden_channels_list = [8, 16, 32, 64]
    num_layers_options = [2, 3, 4, 5]
    models = ["Net", "GCNNet", "GraphSAGENet"]

    auc_results_ls = []

    # Calculate and print AUC for each dataset
    for dataset_name, dataset_results in datasets_dict.items():
        auc_results = calculate_auc_for_models(dataset_results)
        print_sorted_auc_results(dataset_name, auc_results)
        auc_results_ls.append(auc_results)
        print("\n")

    # Convert each AUC results dictionary to DataFrame
    dfs = [convert_to_df(auc_results) for auc_results in auc_results_ls]

    # Calculate global min and max AUC scores
    global_min_auc = min(df["AUC Score"].min() for df in dfs)
    global_max_auc = max(df["AUC Score"].max() for df in dfs)

    # Define your custom titles here
    titles = ["Directed DANI", "Cosine Similarity", "Weighted DANI"]

    # Mapping for renaming models
    model_mapping = {"Net": "GAT", "GCNNet": "GCN", "GraphSAGENet": "GraphSAGE"}

    highlight_coords = {
        "Model": "GAT",
        "Learning Rate": 0.001,
        "Hidden Channels": 8,
        "Layers": 2,
    }

    # Your existing setup
    fig, axes = plt.subplots(1, 3, figsize=(35, 10), sharey=True)
    norm = plt.Normalize(global_min_auc, global_max_auc)
    cmap = "inferno"

    captions = [
        "(a) AUC Scores for Directed DANI",
        "(b) AUC Scores for Weighted DANI",
        "(c) AUC Scores for Cosine Similarity",
    ]

    # Mapping and formatting for each subplot
    for i, (df, title) in enumerate(zip(dfs, titles)):
        df["Model"] = df["Model"].map(model_mapping)
        df["Model"] = pd.Categorical(
            df["Model"], categories=["GCN", "GraphSAGE", "GAT"], ordered=True
        )
        pivot_table = df.pivot_table(
            index=["Model", "Learning Rate"],
            columns=["Hidden Channels", "Layers"],
            values="AUC Score",
        )
        formatted_columns = [
            f"({hidden}, {layer})" for hidden, layer in pivot_table.columns
        ]
        pivot_table.columns = formatted_columns

        ax = axes[i]
        sns.heatmap(
            pivot_table, annot=False, fmt=".2f", cmap=cmap, norm=norm, ax=ax, cbar=False
        )
        ax.set_xlabel("Hidden Channels - Layers", fontsize=label_fontsize)
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        if i == 0:
            ax.set_ylabel("Model - Learning Rate", fontsize=label_fontsize)
        else:
            ax.set_ylabel("")  # Remove the y-axis label for the second and third plots
        ax.set_yticklabels(
            [f"{model}-{lr:.0e}" for model, lr in pivot_table.index],
            fontsize=tick_fontsize,
        )

        # Add specific highlights and annotations
        if title == "Directed DANI":
            num_rows = pivot_table.shape[0]
            row = num_rows - 2  # Second last row
            col = 0  # First column
            ax.plot([col, col + 1], [row, row + 1], color="red", lw=3)
            ax.plot([col, col + 1], [row + 1, row], color="red", lw=3)

        ax.text(
            0.5,
            -0.25,
            captions[i],
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=label_fontsize,
        )

    # Color bar and saving
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)
    plt.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05)
    plt.savefig(
        path.join(PROJECT_ROOT, "data", "auc_scores_combined.pdf"), bbox_inches="tight"
    )
    plt.show()
    plt.close(fig)
    print("Combined plot saved successfully.")
