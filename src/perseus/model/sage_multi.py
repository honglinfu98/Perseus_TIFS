from sklearn.metrics import roc_curve
import torch
from torch.nn import ModuleList, Dropout
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_experiment(model, train_loader, test_loader, lr, num_epochs=100):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_op = torch.nn.BCELoss()

    train_times = []

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
        all_probs, all_labels, all_embs = [], [], []
        batch_times, num_nodes = [], []
        for data in loader:
            data = data.to(device)
            start_time = time.time()
            num_nodes.append(data.x.size(0))
            out = model(
                data.x, data.edge_index
            )  # Make sure embs are the last layer embeddings
            batch_time = time.time() - start_time
            batch_times.append(batch_time)

            all_probs.append(out.cpu())
            all_labels.append(data.y.cpu())
            # all_embs.append(embs.cpu())  # Collect embeddings

        return all_probs, all_labels, batch_times, num_nodes

    train()

    probs, labels, batch_times, num_nodes = test(test_loader)
    probs = torch.cat(probs, dim=0).sigmoid().numpy()
    labels = torch.cat(labels, dim=0).numpy()

    # Compute ROC for each label and store
    fpr_dict, tpr_dict, thresholds_dict = {}, {}, {}
    for i in range(labels.shape[1]):  # Assuming labels is a 2D array: [samples, labels]
        fpr, tpr, thresholds = roc_curve(labels[:, i], probs[:, i])
        fpr_dict[i], tpr_dict[i], thresholds_dict[i] = fpr, tpr, thresholds

    all_labels = [torch.cat(test(test_loader)[1])]

    model_weights = model.state_dict()

    return (
        model,
        model_weights,
        fpr_dict,
        tpr_dict,
        thresholds_dict,
        train_times,
        batch_times,
        num_nodes,
        probs,
        labels,
        all_labels,
    )


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
        return torch.sigmoid(x)


# # Define the flexible GraphSAGENet model
# class GraphSAGENet(torch.nn.Module):
#     def __init__(self, num_features, hidden_channels, num_classes, num_layers=2):
#         super(GraphSAGENet, self).__init__()
#         self.layers = torch.nn.ModuleList()
#         self.layers.append(SAGEConv(num_features, hidden_channels))
#         for _ in range(num_layers - 2):
#             self.layers.append(SAGEConv(hidden_channels, hidden_channels))
#         self.layers.append(SAGEConv(hidden_channels, num_classes))

#     def forward(self, x, edge_index, edge_weight=None):
#         for layer in self.layers[:-1]:
#             x = F.relu(layer(x, edge_index))
#             x = F.dropout(x, training=self.training)
#         embeddings = x  # Extract embeddings from the last hidden layer
#         x = self.layers[-1](x, edge_index)
#         x = torch.sigmoid(x)  # Use sigmoid for binary classification
#         return x, embeddings
