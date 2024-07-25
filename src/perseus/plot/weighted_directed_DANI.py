from os import path
import matplotlib.pyplot as plt
import networkx as nx
import torch
from perseus.model.magamaga import split_data_noloader
from perseus.settings import PROJECT_ROOT

aaa, _, _ = split_data_noloader("COSS")


a, b, c = split_data_noloader("DDM")
aa, bb, cc = split_data_noloader("DDINA")


def count_reciprocal_edges(edge_index):
    # Transpose for easier comparison
    edge_index_t = edge_index.t()
    # Set of tuples for edges
    edge_set = set(map(tuple, edge_index_t.tolist()))
    # Count reciprocals
    reciprocal_count = sum((edge[1], edge[0]) in edge_set for edge in edge_set)
    return reciprocal_count


def calculate_ratio(data):
    num_nodes = torch.max(data.edge_index) + 1  # Assuming node indices start at 0
    reciprocal_count = count_reciprocal_edges(data.edge_index)
    return reciprocal_count / num_nodes


sorted_data_list = sorted(a, key=calculate_ratio, reverse=True)


def plot_graphs_side_by_side(
    edge_index1, edge_weight1, title1="Graph 1", title_size=16
):
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
    plt.figure(figsize=(50, 50))  # Adjust overall figure size

    # Graph 1
    # plt.subplot(1, 2, 1)  # 1 row, 2 columns, subplot 1
    pos = nx.shell_layout(G1)
    edge_widths = [G1[u][v][0]["weight"] * 12 for u, v in G1.edges()]
    nx.draw_networkx_nodes(G1, pos, node_size=1200, node_color="skyblue")
    nx.draw_networkx_edges(
        G1,
        pos,
        width=edge_widths,
        arrowstyle="-|>",
        arrowsize=50,
        edge_color="k",
        connectionstyle="arc3, rad = 0.1",
    )
    plt.title(title1, fontsize=title_size)
    plt.axis("off")

    # Display the plot
    plt.savefig(path.join(PROJECT_ROOT, "data", "embedding.pdf"))

    plt.show()


# Example usage:
# Assuming edge_index1, edge_weight1, edge_index2, and edge_weight2 are defined as tensors or lists
plot_graphs_side_by_side(
    aaa[0].edge_index,
    aaa[0].edge_weight,
    title1="Weighted DANI Network Graph",
    title_size=40,
)
