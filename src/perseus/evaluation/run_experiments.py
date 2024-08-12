"""
This script is used to run the experiments for the empirical study. It is used to train the models on the datasets and evaluate the performance of the models on the test set.
"""

import time
from os import path
import random
import pickle
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import (
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    balanced_accuracy_score,
)
from perseus.dataset.dataset_preparation import (
    get_split_data_pickle,
    split_data,
    get_train_test_validate_data,
    get_train_test_validate_data_pickle,
)
from perseus.model.gnn_model import GCNNet, Net, GraphSAGENet
from perseus.settings import PROJECT_ROOT

# Set a seed value
seed = 725
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_experiment(model, train_loader, test_loader, num_epochs=100):
    """
    Run the experiment for the given model and dataset. Prepare for the training and testing of the model.
    """

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
                output, _ = model(data.x, data.edge_index)
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
        all_probs, all_labels, all_embs = [], [], []
        batch_times, num_nodes = [], []
        for data in loader:
            data = data.to(device)
            start_time = time.time()
            out, embs = model(
                data.x, data.edge_index
            )  # Make sure embs are the last layer embeddings
            batch_time = time.time() - start_time
            batch_times.append(batch_time)

            all_probs.append(out.cpu())
            all_labels.append(data.y.cpu())
            all_embs.append(embs.cpu())  # Collect embeddings

        return all_probs, all_labels, all_embs, batch_times, num_nodes

    train()
    print(len(test(test_loader)))  # Check the number of returned values

    probs, labels, embs, batch_times, num_nodes = test(test_loader)
    probs = torch.cat(probs, dim=0).sigmoid().numpy()
    labels = torch.cat(labels, dim=0).numpy()

    # Compute ROC for each label and store
    fpr_dict, tpr_dict, thresholds_dict = {}, {}, {}
    for i in range(labels.shape[1]):  # Assuming labels is a 2D array: [samples, labels]
        fpr, tpr, thresholds = roc_curve(labels[:, i], probs[:, i])
        fpr_dict[i], tpr_dict[i], thresholds_dict[i] = fpr, tpr, thresholds

    all_labels = [torch.cat(test(test_loader)[1])]

    return (
        model,
        fpr_dict,
        tpr_dict,
        thresholds_dict,
        train_times,
        batch_times,
        num_nodes,
        embs,
        all_labels,
    )


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
        out, _ = model(data.x, data.edge_index)
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


label = 1  # Adjust this based on the label you're interested in
datasets = ["DDINA", "COSS", "DDM"]
models = ["GAT", "GCN", "GraphSAGE"]
num_features = 2  # Set this according to your dataset
num_classes = 2  # Set this according to your dataset
hidden_channels = 8
dataset_colors = {"DDINA": "blue", "COSS": "green", "DDM": "red"}

# Non time dimension related experiments
results = {"DDINA": {}, "COSS": {}, "DDM": {}}

for dataset in datasets:
    train_loader, test_loader, _ = get_split_data_pickle(dataset)
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
        m, fpr, tpr, _, train_times, batch_times, num_nodes, embs, labels = (
            run_experiment(model, train_loader, test_loader, num_epochs=100)
        )
        metrics = compute_metrics(model, test_loader)
        # Store results along with training times
        results[dataset][model_name] = {
            "model": m,
            "metrics": metrics,
            "fpr": fpr,
            "tpr": tpr,
            "train_times": train_times,  # Storing the training times
            "batch_times": batch_times,
            "num_nodes": num_nodes,
            "embs": embs,
            "labels": labels,
        }


results1 = {"DDINA": {}, "COSS": {}, "DDM": {}}


for dataset in datasets:
    train_loader, test_loader, _ = get_train_test_validate_data_pickle(dataset)
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
        m, fpr, tpr, _, train_times, batch_times, num_nodes, embs, labels = (
            run_experiment(model, train_loader, test_loader, num_epochs=100)
        )
        metrics = compute_metrics(model, test_loader)
        # Store results along with training times
        results1[dataset][model_name] = {
            "model": m,
            "metrics": metrics,
            "fpr": fpr,
            "tpr": tpr,
            "train_times": train_times,  # Storing the training times
            "batch_times": batch_times,
            "num_nodes": num_nodes,
            "embs": embs,
            "labels": labels,
        }

with open(path.join(PROJECT_ROOT, "data", "results.pkl"), "wb") as file:
    pickle.dump(results, file)


with open(path.join(PROJECT_ROOT, "data", "results1.pkl"), "wb") as file:
    pickle.dump(results1, file)
