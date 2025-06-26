import torch
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv,
    SAGEConv,
    global_max_pool,
)


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
