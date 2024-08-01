"""
This script is used to plot the number of nodes and edges in the graphs for the DANI and cosine similarity models.
"""

from os import path
import numpy as np
import matplotlib.pyplot as plt
from perseus.model.magamaga import split_data_noloader
from perseus.settings import PROJECT_ROOT


def extract_counts(graphs: list, count_type="nodes"):
    """
    Extract the number of nodes or edges from a list of graphs.
    """
    if count_type == "nodes":
        return [graph.num_nodes for graph in graphs]
    elif count_type == "edges":
        return [graph.num_edges for graph in graphs]
    else:
        raise ValueError("count_type must be 'nodes' or 'edges'")


def set_text_size(ax, title_size, label_size, tick_size):
    ax.title.set_size(title_size)
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)
    ax.tick_params(axis="both", labelsize=tick_size)


def plot_graph_summary(
    node_counts: list,
    edge_counts_directed_dani: list,
    edge_counts_weighted_dani: list,
    edge_counts_cosine: list,
    title_size=20,
    label_size=15,
    tick_size=10,
    legend_size=12,
):
    """
    This function is used to plot the number of nodes and edges in the graphs for the DANI and cosine similarity models.
    """
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))  # Single chart

    # Convert histogram to line plot for each category and plot on the same axis
    def plot_line_hist(data, ax, label, color):
        counts, bin_edges = np.histogram(data, bins=10)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax.plot(
            bin_centers,
            counts,
            linestyle="-",
            marker="o",
            color=color,
            label=label,
        )
        return bin_centers

    # Store original bin centers for x-ticks
    node_bin_centers = plot_line_hist(node_counts, ax, "Nodes", "skyblue")
    directed_dani_bin_centers = plot_line_hist(
        edge_counts_directed_dani, ax, "Edges for Directed DANI", "lightgreen"
    )
    weighted_dani_bin_centers = plot_line_hist(
        edge_counts_weighted_dani, ax, "Edges for Weighted DANI", "gold"
    )
    cosine_bin_centers = plot_line_hist(
        edge_counts_cosine, ax, "Edges for Cosine similarity", "salmon"
    )

    ax.set_xlabel("Number of Nodes or Edges", fontsize=label_size)
    ax.set_ylabel("Frequency", fontsize=label_size)
    ax.legend(fontsize=legend_size)

    # Use a logarithmic scale for x-axis and set evenly spaced ticks
    ax.set_xscale("log")
    min_bin = min(
        node_bin_centers.min(),
        directed_dani_bin_centers.min(),
        weighted_dani_bin_centers.min(),
        cosine_bin_centers.min(),
    )
    max_bin = max(
        node_bin_centers.max(),
        directed_dani_bin_centers.max(),
        weighted_dani_bin_centers.max(),
        cosine_bin_centers.max(),
    )
    ax.set_xticks(np.logspace(np.log10(min_bin), np.log10(max_bin), num=5))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    set_text_size(ax, title_size, label_size, tick_size)

    plt.tight_layout()
    plt.savefig(path.join(PROJECT_ROOT, "data", "graphs_summary_combined.pdf"))
    plt.show()


if __name__ == "__main__":
    # List of models
    model = ["DDINA", "COSS", "DDM"]

    # Collecting results
    results = []
    for i in model:
        a, b, c = split_data_noloader(i)
        results.append((a, b, c))

    # Sample graphs data would be needed here.
    directed_dani = results[0][0] + results[0][1] + results[0][2]
    cosine = results[1][0] + results[1][1] + results[1][2]
    weighted_dani = results[2][0] + results[2][1] + results[2][2]

    # Extracting data
    node_counts = extract_counts(directed_dani, "nodes")
    edge_counts_directed_dani = extract_counts(directed_dani, "edges")
    edge_counts_cosine = extract_counts(cosine, "edges")
    edge_counts_weighted_dani = extract_counts(weighted_dani, "edges")

    # Plotting the graph with customizable text sizes
    plot_graph_summary(
        node_counts,
        edge_counts_directed_dani,
        edge_counts_weighted_dani,
        edge_counts_cosine,
        title_size=20,
        label_size=20,
        tick_size=20,
        legend_size=20,
    )
