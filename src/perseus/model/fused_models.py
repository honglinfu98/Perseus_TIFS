#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import (
    GATv2Conv,
    SAGEConv,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.nn.aggr import AttentionalAggregation
from sklearn.metrics import f1_score, precision_score, accuracy_score

# --- 1) Import your dataset & splitter here ---
# get_split_data_pickle_btc should return three Python iterables (lists) of Data objects
from perseus.dataset.dataset_preparation import get_split_data_pickle_btc_noloader


class MultiGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=8):
        super().__init__()
        # two-layer GATv2 encoder supporting edge features (edge_dim=1)
        self.conv1 = GATv2Conv(
            in_channels, hidden_channels, heads=heads, concat=True, edge_dim=1
        )
        # next layer reduces to hidden_channels (heads=1) for pooling
        self.conv2 = GATv2Conv(
            hidden_channels * heads, hidden_channels, heads=1, concat=False, edge_dim=1
        )
        # MLP on fused graph embeddings
        self.graph_mlp = torch.nn.Linear(hidden_channels, hidden_channels)
        # combine node + context
        self.node_mlp = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        # final classifier → one logit per node
        self.out_lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data_list):
        node_embs, graph_embs = [], []
        # encode & pool each graph
        for data in data_list:
            # prepare edge_attr for GATv2Conv
            edge_attr = getattr(data, "edge_weight", None)
            if edge_attr is not None:
                edge_attr = edge_attr.unsqueeze(-1)

            # first attention layer + activation
            h = self.conv1(x=data.x, edge_index=data.edge_index, edge_attr=edge_attr)
            h = F.elu(h)

            # second attention layer + activation
            h = self.conv2(x=h, edge_index=data.edge_index, edge_attr=edge_attr)
            h = F.elu(h)

            node_embs.append(h)
            # single-graph batch index
            batch_idx = data.x.new_zeros(h.size(0), dtype=torch.long)
            g_emb = global_max_pool(h, batch_idx)  # [1, hidden]
            graph_embs.append(g_emb)

        # # fuse all graph embeddings into one context
        # G = torch.cat(graph_embs, dim=0).max(dim=0)  # [hidden]
        # G = F.relu(self.graph_mlp(G))  # [hidden]

        # fuse all graph embeddings into one context
        G = torch.cat(graph_embs, dim=0).max(dim=0).values  # [hidden]
        G = F.relu(self.graph_mlp(G))  # [hidden]

        # inject context into nodes and classify
        logits_list = []
        for h in node_embs:
            n = h.size(0)
            G_rep = G.unsqueeze(0).expand(n, -1)  # [n, hidden]
            H_cat = torch.cat([h, G_rep], dim=1)  # [n, hidden*2]
            h2 = F.relu(self.node_mlp(H_cat))  # [n, hidden]
            logits = self.out_lin(h2).view(-1)  # [n]
            logits_list.append(logits)

        return logits_list


class MultiGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        # shared two-layer GraphSAGE encoder
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        # MLP on fused graph embeddings
        self.graph_mlp = torch.nn.Linear(hidden_channels, hidden_channels)
        # combine node + context
        self.node_mlp = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        # final classifier → one logit per node
        self.out_lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data_list):
        node_embs, graph_embs = [], []
        # encode & pool each graph
        for data in data_list:
            h = self.conv1(data.x, data.edge_index)
            h = F.relu(h)
            h = self.conv2(h, data.edge_index)
            h = F.relu(h)
            node_embs.append(h)
            # single-graph batch index
            batch_idx = data.x.new_zeros(h.size(0), dtype=torch.long)
            g_emb = global_max_pool(h, batch_idx)  # [1, hidden]
            graph_embs.append(g_emb)

        # # fuse all graph embeddings into one context
        # G = torch.cat(graph_embs, dim=0).max(dim=0)  # [hidden]
        # G = F.relu(self.graph_mlp(G))  # [hidden]

        # fuse all graph embeddings into one context
        G = torch.cat(graph_embs, dim=0).max(dim=0).values  # [hidden]
        G = F.relu(self.graph_mlp(G))  # [hidden]

        # inject context into nodes and classify
        logits_list = []
        for h in node_embs:
            n = h.size(0)
            G_rep = G.unsqueeze(0).expand(n, -1)  # [n, hidden]
            H_cat = torch.cat([h, G_rep], dim=1)  # [n, hidden*2]
            h2 = F.relu(self.node_mlp(H_cat))  # [n, hidden]
            logits = self.out_lin(h2).view(-1)  # [n]
            logits_list.append(logits)

        return logits_list


# class MeanMultiGAT(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, heads=8):
#         super().__init__()
#         # two-layer GATv2 encoder supporting edge features (edge_dim=1)
#         self.conv1 = GATv2Conv(
#             in_channels, hidden_channels, heads=heads, concat=True, edge_dim=1
#         )
#         # next layer reduces to hidden_channels (heads=1) for pooling
#         self.conv2 = GATv2Conv(
#             hidden_channels * heads, hidden_channels, heads=1, concat=False, edge_dim=1
#         )
#         # MLP on fused graph embeddings
#         self.graph_mlp = torch.nn.Linear(hidden_channels, hidden_channels)
#         # combine node + context
#         self.node_mlp = torch.nn.Linear(hidden_channels * 2, hidden_channels)
#         # final classifier → one logit per node
#         self.out_lin = torch.nn.Linear(hidden_channels, 1)

#     def forward(self, data_list):
#         node_embs, graph_embs = [], []
#         # encode & pool each graph
#         for data in data_list:
#             # prepare edge_attr for GATv2Conv
#             edge_attr = getattr(data, "edge_weight", None)
#             if edge_attr is not None:
#                 edge_attr = edge_attr.unsqueeze(-1)

#             # first attention layer + activation
#             h = self.conv1(x=data.x, edge_index=data.edge_index, edge_attr=edge_attr)
#             h = F.elu(h)

#             # second attention layer + activation
#             h = self.conv2(x=h, edge_index=data.edge_index, edge_attr=edge_attr)
#             h = F.elu(h)

#             node_embs.append(h)
#             # single-graph batch index
#             batch_idx = data.x.new_zeros(h.size(0), dtype=torch.long)
#             g_emb = global_mean_pool(h, batch_idx)  # [1, hidden]
#             graph_embs.append(g_emb)

#         # # fuse all graph embeddings into one context
#         # G = torch.cat(graph_embs, dim=0).max(dim=0)  # [hidden]
#         # G = F.relu(self.graph_mlp(G))  # [hidden]

#         # fuse all graph embeddings into one context
#         G = torch.cat(graph_embs, dim=0).mean(dim=0)  # [hidden]
#         G = F.relu(self.graph_mlp(G))  # [hidden]

#         # inject context into nodes and classify
#         logits_list = []
#         for h in node_embs:
#             n = h.size(0)
#             G_rep = G.unsqueeze(0).expand(n, -1)  # [n, hidden]
#             H_cat = torch.cat([h, G_rep], dim=1)  # [n, hidden*2]
#             h2 = F.relu(self.node_mlp(H_cat))  # [n, hidden]
#             logits = self.out_lin(h2).view(-1)  # [n]
#             logits_list.append(logits)

#         return logits_list


# class MeanMultiGraphSAGE(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels):
#         super().__init__()
#         # shared two-layer GraphSAGE encoder
#         self.conv1 = SAGEConv(in_channels, hidden_channels)
#         self.conv2 = SAGEConv(hidden_channels, hidden_channels)
#         # MLP on fused graph embeddings
#         self.graph_mlp = torch.nn.Linear(hidden_channels, hidden_channels)
#         # combine node + context
#         self.node_mlp = torch.nn.Linear(hidden_channels * 2, hidden_channels)
#         # final classifier → one logit per node
#         self.out_lin = torch.nn.Linear(hidden_channels, 1)

#     def forward(self, data_list):
#         node_embs, graph_embs = [], []
#         # encode & pool each graph
#         for data in data_list:
#             h = self.conv1(data.x, data.edge_index)
#             h = F.relu(h)
#             h = self.conv2(h, data.edge_index)
#             h = F.relu(h)
#             node_embs.append(h)
#             # single-graph batch index
#             batch_idx = data.x.new_zeros(h.size(0), dtype=torch.long)
#             g_emb = global_mean_pool(h, batch_idx)  # [1, hidden]
#             graph_embs.append(g_emb)

#         # # fuse all graph embeddings into one context
#         # G = torch.cat(graph_embs, dim=0).max(dim=0)  # [hidden]
#         # G = F.relu(self.graph_mlp(G))  # [hidden]

#         # fuse all graph embeddings into one context
#         G = torch.cat(graph_embs, dim=0).mean(dim=0)  # [hidden]
#         G = F.relu(self.graph_mlp(G))  # [hidden]

#         # inject context into nodes and classify
#         logits_list = []
#         for h in node_embs:
#             n = h.size(0)
#             G_rep = G.unsqueeze(0).expand(n, -1)  # [n, hidden]
#             H_cat = torch.cat([h, G_rep], dim=1)  # [n, hidden*2]
#             h2 = F.relu(self.node_mlp(H_cat))  # [n, hidden]
#             logits = self.out_lin(h2).view(-1)  # [n]
#             logits_list.append(logits)

#         return logits_list


# # def train_epoch(model, loader, optimizer, criterion, device):
# #     model.train()
# #     total_loss = total_correct = total_nodes = 0

# #     for data_list in loader:
# #         data_list = [d.to(device) for d in data_list]
# #         logits_list = model(data_list)
# #         y_list = [d.y.view(-1).float() for d in data_list]

# #         logits = torch.cat(logits_list, dim=0)
# #         y = torch.cat(y_list, dim=0)

# #         optimizer.zero_grad()
# #         loss = criterion(logits, y)
# #         loss.backward()
# #         optimizer.step()

# #         total_loss += loss.item() * y.size(0)
# #         preds = (logits.sigmoid() > 0.5).long()
# #         total_correct += (preds == y.long()).sum().item()
# #         total_nodes += y.size(0)

# #     return total_loss / total_nodes, total_correct / total_nodes


# # @torch.no_grad()
# # def eval_epoch(model, loader, criterion, device):
# #     model.eval()
# #     total_loss = total_correct = total_nodes = 0

# #     for data_list in loader:
# #         data_list = [d.to(device) for d in data_list]
# #         logits_list = model(data_list)
# #         y_list = [d.y.view(-1).float() for d in data_list]

# #         logits = torch.cat(logits_list, dim=0)
# #         y = torch.cat(y_list, dim=0)

# #         loss = criterion(logits, y)
# #         total_loss += loss.item() * y.size(0)
# #         preds = (logits.sigmoid() > 0.5).long()
# #         total_correct += (preds == y.long()).sum().item()
# #         total_nodes += y.size(0)

# #     return total_loss / total_nodes, total_correct / total_nodes


# # def find_optimal_threshold(probs, labels, step=0.01):
# #     """
# #     Find threshold between 0 and 1 that maximizes F1 score.

# #     Args:
# #         probs (np.ndarray): Predicted probabilities.
# #         labels (np.ndarray): True binary labels (0 or 1).
# #         step (float): Step size for threshold search.

# #     Returns:
# #         best_thresh (float), best_f1, best_prec, best_acc
# #     """
# #     best_thresh = 0.5
# #     best_f1 = 0.0
# #     thresholds = np.arange(0.0, 1.0 + step, step)

# #     for t in thresholds:
# #         preds = (probs > t).astype(int)
# #         f1 = f1_score(labels, preds)
# #         if f1 > best_f1:
# #             best_f1 = f1
# #             best_thresh = t

# #     best_preds = (probs > best_thresh).astype(int)
# #     best_prec = precision_score(labels, best_preds)
# #     best_acc = accuracy_score(labels, best_preds)

# #     return best_thresh, best_f1, best_prec, best_acc


# # if __name__ == "__main__":
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# #     data = "DDINA"
# #     # data = "DDM"

# #     # --- 2) Use get_split_data_pickle_btc to obtain lists of Data ---
# #     train_list, test_list, valid_list = get_split_data_pickle_btc(data)

# #     # --- 3) Wrap them with DataListLoader for cross-graph batches ---
# #     train_loader = DataListLoader(train_list, batch_size=8, shuffle=True)
# #     valid_loader = DataListLoader(valid_list, batch_size=8)
# #     test_loader = DataListLoader(test_list, batch_size=8)

# #     # hyperparameters
# #     hidden_channels = 8
# #     lr = 1e-4
# #     weight_decay = 5e-4
# #     epochs = 100

# #     # infer input dims from one graph
# #     in_channels = train_list[0].x.size(1)

# #     # Initialize model with GATv2 backbone supporting edge weights
# #     model = MultiGAT(
# #         in_channels=in_channels,
# #         hidden_channels=hidden_channels,
# #         heads=8,
# #     ).to(device)

# #     model = MultiGraphSAGE(
# #         in_channels=in_channels,
# #         hidden_channels=hidden_channels,
# #     ).to(device)
# #     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
# #     criterion = torch.nn.BCEWithLogitsLoss()

# #     best_val_acc = 0.0
# #     for epoch in range(1, epochs + 1):
# #         train_loss, train_acc = train_epoch(
# #             model, train_loader, optimizer, criterion, device
# #         )
# #         val_loss, val_acc = eval_epoch(model, valid_loader, criterion, device)

# #         if val_acc > best_val_acc:
# #             best_val_acc = val_acc
# #             torch.save(model.state_dict(), "best_model.pt")

# #         print(
# #             f"Epoch {epoch:02d} | "
# #             f"Train L={train_loss:.4f}, A={train_acc:.4f} | "
# #             f"Valid L/A={val_loss:.4f}/{val_acc:.4f}"
# #         )

# #     # final test evaluation
# #     model.load_state_dict(torch.load("best_model.pt"))
# #     test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
# #     print(f"Test Loss={test_loss:.4f}, Acc={test_acc:.4f}")

# #     # --- Find optimal threshold on test set for F1 ---
# #     all_probs = []
# #     all_labels = []
# #     model.eval()
# #     with torch.no_grad():
# #         for data_list in test_loader:
# #             data_list = [d.to(device) for d in data_list]
# #             logits_list = model(data_list)
# #             for logits, data in zip(logits_list, data_list):
# #                 probs = torch.sigmoid(logits).cpu().numpy()
# #                 labels = data.y.view(-1).cpu().numpy()
# #                 all_probs.extend(probs.tolist())
# #                 all_labels.extend(labels.tolist())

# #     all_probs = np.array(all_probs)
# #     all_labels = np.array(all_labels)
# #     best_thresh, best_f1, best_prec, best_acc = find_optimal_threshold(
# #         all_probs, all_labels, step=0.01
# #     )

# #     print(f"Optimal threshold: {best_thresh:.2f}")
# #     print(f"F1 at optimal threshold: {best_f1:.4f}")
# #     print(f"Precision at optimal threshold: {best_prec:.4f}")
# #     print(f"Accuracy at optimal threshold: {best_acc:.4f}")
