import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns


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


# def run_experiment(model,train_loader,test_loader, num_epochs=100):
#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
#     loss_op = torch.nn.BCEWithLogitsLoss()

#     def train():
#         for epoch in range(num_epochs):
#             model.train()
#             total_loss = 0
#             for data in train_loader:
#                 data = data.to(device)
#                 optimizer.zero_grad()
#                 output = model(data.x, data.edge_index)
#                 if torch.isnan(output).any() or torch.isinf(output).any():
#                     print("NaN or Inf in model output")
#                     continue
#                 loss = loss_op(output, data.y.float())
#                 if torch.isnan(loss) or torch.isinf(loss):
#                     print("NaN or Inf in loss")
#                     continue
#                 total_loss += loss.item() * data.num_graphs
#                 loss.backward()
#                 optimizer.step()
#             print(
#                 f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader.dataset)}"
#             )

#     @torch.no_grad()
#     def test(loader):
#         model.eval()
#         all_probs, all_labels = [], []
#         for data in loader:
#             data = data.to(device)
#             out = model(data.x, data.edge_index)
#             all_probs.append(out.cpu())
#             all_labels.append(data.y.cpu())
#         return all_probs, all_labels

#     train()
#     probs, labels = test(test_loader)
#     probs = torch.cat(probs, dim=0).sigmoid().numpy()
#     labels = torch.cat(labels, dim=0).numpy()
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
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader.dataset)}, Time: {epoch_time:.2f}s")

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
        _, fpr, tpr, _, train_times = run_experiment(model, train_loader, test_loader, num_epochs=100)
        metrics = compute_metrics(model, test_loader)
        # Store results along with training times
        results[dataset][model_name] = {
            'metrics': metrics,
            'fpr': fpr,
            'tpr': tpr,
            'train_times': train_times  # Storing the training times
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




label = 1  # Adjust this based on the label you're interested in
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















# Unit computation time for different embedding for the same model 
graphSAGE_train_times = {dataset: results[dataset]['GraphSAGE']['train_times'] for dataset in datasets}

# Flatten the dictionary to prepare for DataFrame creation
time_data = [(dataset, time) for dataset, times in graphSAGE_train_times.items() for time in times]
df_times = pd.DataFrame(time_data, columns=['Dataset', 'Time'])

# Plotting the distribution of training times
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_times, x='Dataset', y='Time')
# sns.stripplot(data=df_times, x='Dataset', y='Time', color='black', alpha=0.5)

plt.title('Distribution of Training Times for GraphSAGE Across Embedding Methods')
plt.ylabel('Time per Epoch (seconds)')
plt.xlabel('Dataset')








# Comparison across different embedding for the same model
# Assuming 'results' contains your ROC curve data
datasets = ['DDINA', 'COSS', 'DDM']
models = ['GCN', 'GAT', 'GraphSAGE']
dataset_colors = {
    'DDINA': 'blue',
    'COSS': 'green',
    'DDM': 'red'
}



# Create a figure and a set of subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Loop through each model and plot its ROC curve on a different subplot
for i, model_name in enumerate(models):
    for dataset in datasets:
        fpr = results[dataset][model_name]['fpr'][label]
        tpr = results[dataset][model_name]['tpr'][label]
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
metrics_columns = sorted(results[datasets[0]][models[0]]['metrics'].keys())

# Loop through each dataset and model to gather metrics
for dataset in datasets:
    for model_name in models:
        # Check if the dataset and model_name keys exist in the results dictionary
        if dataset in results and model_name in results[dataset]:
            # Retrieve the stored metrics for the current dataset and model
            metrics = results[dataset][model_name]['metrics']
            # Append a new record including the dataset, model, and all metrics
            data.append([dataset, model_name] + [metrics[metric] for metric in sorted(metrics)])
        else:
            # Handle the missing dataset/model combination by appending NaNs or placeholders
            data.append([dataset, model_name] + [None for _ in sorted(metrics_columns)])

# Assuming 'metrics_columns' is predefined or you define it based on your known metrics
columns = ['Dataset', 'Model'] + metrics_columns
df = pd.DataFrame(data, columns=columns)

# Assuming 'df' is your original DataFrame

# Step 1: Keep only the specified columns
filtered_df = df[['Dataset', 'Model', 'accuracy', 'f1', 'precision', 'recall']]

# Step 2: Apply the .style.highlight_max() method
highlighted_df = filtered_df.style.highlight_max(subset=['accuracy', 'f1', 'precision', 'recall'], color='yellow', axis=0)

# Display the DataFrame with highlighted maximum values
highlighted_df