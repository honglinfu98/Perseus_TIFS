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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# @torch.no_grad()
# def compute_metrics(model, loader):
#     model.eval()
#     all_probs, all_labels = [], []
#     for data in loader:
#         data = data.to(device)
#         out = model(data.x, data.edge_index)
#         all_probs.append(out.cpu())
#         all_labels.append(data.y.cpu())
#     probs = torch.cat(all_probs, dim=0).sigmoid().numpy()
#     labels = torch.cat(all_labels, dim=0).numpy()
#     preds = probs.argmax(axis=1)

#     # Converting labels for multi-class classification
#     labels = labels.argmax(axis=1)

#     # Calculating metrics
#     accuracy = calculate_accuracy(labels, preds)
#     precision = calculate_precision(labels, preds)
#     recall = calculate_recall(labels, preds)
#     f1 = calculate_f1_score(labels, preds)


# TODO: make the comparison 3 or 2
# TODO: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/neighbor_loader.html?highlight=sampling
# TODO: https://neo4j.com/blog/graph-algorithms-neo4j-betweenness-centrality/?utm_source=google&utm_medium=PaidSearch&utm_campaign=GDB&utm_content=APAC-X-Awareness-GDB-Text&utm_term=&gad_source=1&gclid=CjwKCAiA7t6sBhAiEiwAsaieYoEzQ8natFE0mCoqHrA2LhWq6050KWKz4wOIzGqxJ3d7zIIrcQTmZxoCC1IQAvD_BwE
# TODO: (why we use the above sampling method, in degree and betweeness centrality)
# TODO: experiment for time evaluation and accurary


# TODO: no slicing window for average, just each epoch
# TODO: validation, test; test not in validation and training.
# TODO: Network comparison, frst three year and last years
# TODO: Three dataset test, train, validate
# TODO: write the paper by specifying how we label the data (see the review reply)

# validation needs not to show

# GAT
# 邻居点的采样方法，为什么要这样采样，如何设计采样方式
# 调参数，随机数
# random seed 44

# TODO: experiment with dani and without dani.
# todo：more experiments the better, more comparison


# GCN Implementation
class GCNNet(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# GraphSAGE Implementation
class GraphSAGENet(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GraphSAGENet, self).__init__()
        self.conv1 = SAGEConv(num_features, 16)
        self.conv2 = SAGEConv(16, num_classes)

    def forward(self, x, edge_index):
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


# train_loader, test_loader = get_data_loader()
train_loader, test_loader, c = get_data_loader("DDINA")


def run_experiment(model, num_epochs=100):
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

    return (
        accuracy,
        precision,
        recall,
        f1,
        specificity,
        prevalence,
        detection_rate,
        detection_prevalence,
    )


# Run experiments for each model
num_features = 4  # Set this according to your dataset
num_classes = 3  # Set this according to your dataset

print("Running GAT Experiment")
gat_model, gat_fpr, gat_tpr, _ = run_experiment(Net(), num_epochs=100)

print("\nRunning GCN Experiment")
gcn_model, gcn_fpr, gcn_tpr, _ = run_experiment(
    GCNNet(num_features, num_classes), num_epochs=100
)

print("\nRunning GraphSAGE Experiment")
graphsage_model, graphsage_fpr, graphsage_tpr, _ = run_experiment(
    GraphSAGENet(num_features, num_classes), num_epochs=100
)

gat_metrics = compute_metrics(gat_model, test_loader)
print(
    f"GAT Model Metrics:\nAccuracy: {gat_metrics[0]}, Precision: {gat_metrics[1]}, Recall: {gat_metrics[2]}, F1 Score: {gat_metrics[3]}, Specificity: {gat_metrics[4]}, Prevalence: {gat_metrics[5]}, Detection Rate: {gat_metrics[6]}, Detection Prevalence: {gat_metrics[7]}\n"
)

gcn_metrics = compute_metrics(gcn_model, test_loader)
print(
    f"GAT Model Metrics:\nAccuracy: {gcn_metrics[0]}, Precision: {gcn_metrics[1]}, Recall: {gcn_metrics[2]}, F1 Score: {gcn_metrics[3]}, Specificity: {gcn_metrics[4]}, Prevalence: {gcn_metrics[5]}, Detection Rate: {gcn_metrics[6]}, Detection Prevalence: {gcn_metrics[7]}\n"
)

graphsage_metrics = compute_metrics(graphsage_model, test_loader)
print(
    f"GAT Model Metrics:\nAccuracy: {graphsage_metrics[0]}, Precision: {graphsage_metrics[1]}, Recall: {graphsage_metrics[2]}, F1 Score: {graphsage_metrics[3]}, Specificity: {graphsage_metrics[4]}, Prevalence: {graphsage_metrics[5]}, Detection Rate: {graphsage_metrics[6]}, Detection Prevalence: {graphsage_metrics[7]}\n"
)

# Plotting ROC Curves
plt.figure(figsize=(10, 6))
label = 0  # Assuming we are plotting for the first label
plt.plot(gat_fpr[label], gat_tpr[label], label="GAT")
plt.plot(gcn_fpr[label], gcn_tpr[label], label="GCN")
plt.plot(graphsage_fpr[label], graphsage_tpr[label], label="GraphSAGE")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves Comparison")
plt.legend()
plt.show()
