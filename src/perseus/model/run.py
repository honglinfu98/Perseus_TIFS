import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from perseus.model.data_loader import get_data_loader


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


train_loader, test_loader = get_data_loader()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
loss_op = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)


def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        # Check output range
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
    return total_loss / len(train_loader.dataset)


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


# Training and Testing loop
average_auc_scores = []
for epoch in range(1, 101):
    train()
    probs, labels = test(test_loader)

    probs = torch.cat(probs, dim=0).sigmoid().numpy()
    labels = torch.cat(labels, dim=0).numpy()

    # Compute AUC for each label and average
    auc_scores = []
    for i in range(labels.shape[1]):  # Assuming labels is a 2D array: [samples, labels]
        fpr, tpr, _ = roc_curve(labels[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores.append(roc_auc)
    average_auc = np.mean(auc_scores)
    average_auc_scores.append(average_auc)

    print(f"Epoch: {epoch:03d}, Average AUC: {average_auc:.4f}")

# Plotting average AUC scores
plt.plot(range(1, 101), average_auc_scores, label="Average AUC")
plt.xlabel("Epoch")
plt.ylabel("Average AUC")
plt.title("Average AUC per Epoch")
plt.legend()
plt.show()
