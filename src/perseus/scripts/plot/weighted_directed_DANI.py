"""
This script is used to plot the weighted directed DANI network graph
"""

from os import path
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch_geometric
from perseus.dataset.dataset_preparation import split_data
from perseus.settings import PROJECT_ROOT


def count_reciprocal_edges(edge_index: torch.Tensor):
    """
    Count the number of reciprocal edges in a graph
    """

    # Transpose for easier comparison
    edge_index_t = edge_index.t()
    # Set of tuples for edges
    edge_set = set(map(tuple, edge_index_t.tolist()))
    # Count reciprocals
    reciprocal_count = sum((edge[1], edge[0]) in edge_set for edge in edge_set)
    return reciprocal_count


def calculate_ratio(data: torch_geometric.data.data.Data):
    """
    Calculate the ratio of reciprocal edges in a graph
    """
    num_nodes = torch.max(data.edge_index) + 1  # Assuming node indices start at 0
    reciprocal_count = count_reciprocal_edges(data.edge_index)
    return reciprocal_count / num_nodes


def plot_weighted_edges(
    edge_index1: torch.Tensor,
    edge_weight1: torch.Tensor,
):
    """
    Plot the two graphs side by side
    """
    # Initialize two graphs
    G1 = nx.MultiDiGraph()
    # G2 = nx.MultiDiGraph()

    # Convert tensors if necessary and add edges for Graph 1
    if isinstance(edge_index1, torch.Tensor):
        edge_index1 = edge_index1.t().tolist()
    edge_weights1 = edge_weight1.tolist()
    for (u, v), weight in zip(edge_index1, edge_weights1):
        G1.add_edge(u, v, weight=weight)

    # Create a figure with two subplots
    plt.figure(figsize=(10, 10))  # Adjust overall figure size

    # Graph 1
    # plt.subplot(1, 2, 1)  # 1 row, 2 columns, subplot 1
    pos = nx.shell_layout(G1)
    edge_widths = [G1[u][v][0]["weight"] * 12 for u, v in G1.edges()]
    nx.draw_networkx_nodes(G1, pos, node_size=100, node_color="skyblue")
    nx.draw_networkx_edges(
        G1,
        pos,
        width=edge_widths,
        arrowstyle="-|>",
        arrowsize=50,
        edge_color="k",
        connectionstyle="arc3, rad = 0.1",
    )
    # plt.title(title1, fontsize=title_size)
    plt.axis("off")

    # Display the plot
    plt.savefig(path.join(PROJECT_ROOT, "data", "embedding1.pdf"))

    plt.show()


def plot_directed_edges(edge_index1: torch.Tensor):
    """
    Plot the two graphs side by side
    """
    # Initialize two graphs
    G1 = nx.MultiDiGraph()
    # G2 = nx.MultiDiGraph()

    # Convert tensors if necessary and add edges for Graph 1
    if isinstance(edge_index1, torch.Tensor):
        edge_index1 = edge_index1.t().tolist()
    # edge_weights1 = edge_weight1.tolist()
    for u, v in edge_index1:
        G1.add_edge(u, v)

    # Create a figure with two subplots
    plt.figure(figsize=(10, 10))  # Adjust overall figure size

    # Graph 1
    # plt.subplot(1, 2, 1)  # 1 row, 2 columns, subplot 1
    pos = nx.shell_layout(G1)
    # edge_widths = [G1[u][v][0]["weight"] * 12 for u, v in G1.edges()]
    nx.draw_networkx_nodes(G1, pos, node_size=100, node_color="skyblue")
    nx.draw_networkx_edges(
        G1,
        pos,
        width=30,
        arrowstyle="-|>",
        arrowsize=50,
        edge_color="k",
        # connectionstyle="arc3, rad = 0.1",
    )
    # plt.title(title1, fontsize=title_size)
    plt.axis("off")

    # Display the plot
    plt.savefig(path.join(PROJECT_ROOT, "data", "embedding2.pdf"))

    plt.show()


if __name__ == "__main__":

    aaa, _, _ = split_data("COSS")
    a, b, c = split_data("DDM")
    aa, bb, cc = split_data("DDINA")

    # sorted_data_list = sorted(a, key=calculate_ratio, reverse=True)
    indexed_data = list(enumerate(a))
    sorted_indexed_data = sorted(
        indexed_data, key=lambda x: calculate_ratio(x[1]), reverse=True
    )
    original_indices = [idx for idx, _ in sorted_indexed_data]

    # Example usage:
    # Assuming edge_index1, edge_weight1, edge_index2, and edge_weight2 are defined as tensors or lists
    plot_weighted_edges(a[0].edge_index, a[0].edge_weight)

    plot_directed_edges(aa[89].edge_index)
