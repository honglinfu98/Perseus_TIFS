import random

import torch


def custom_sampling(edge_index, num_samples):
    # edge_index: [2, E] tensor of edge indices
    # num_samples: number of neighbors to sample for each node

    node_indices = edge_index[0].unique()
    sampled_edges = []

    for node in node_indices:
        neighbors = edge_index[1][edge_index[0] == node]
        sampled_neighbors = random.sample(
            neighbors.tolist(), min(num_samples, len(neighbors))
        )
        for neighbor in sampled_neighbors:
            sampled_edges.append([node.item(), neighbor])

    return torch.tensor(sampled_edges).t()


def custom_aggregate(node_features, sampled_edges):
    # node_features: [N, F] tensor of node features
    # sampled_edges: [2, E] tensor of edges after sampling

    aggregated_features = torch.zeros_like(node_features)
    for node in range(node_features.size(0)):
        neighbors = sampled_edges[1][sampled_edges[0] == node]
        if len(neighbors) > 0:
            neighbor_features = node_features[neighbors]
            aggregated_features[node] = neighbor_features.mean(dim=0)

    return aggregated_features


class CustomGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomGraphSAGE, self).__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.act = torch.nn.ReLU()

    def forward(self, x, edge_index):
        # Custom sampling
        sampled_edges = custom_sampling(edge_index, num_samples=10)

        # Custom aggregation
        aggregated_x = custom_aggregate(x, sampled_edges)

        # Apply a linear transformation and activation
        return self.act(self.lin(aggregated_x))


# Example usage
model = CustomGraphSAGE(in_channels=data.num_node_features, out_channels=16)
out = model(data.x, data.edge_index)
