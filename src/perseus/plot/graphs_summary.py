import numpy as np
from perseus.model.magamaga import split_data_noloader
from os import path
from perseus.settings import PROJECT_ROOT
import matplotlib.pyplot as plt

# List of models
model = ["DDINA", "COSS", "DDM"]

# Collecting results
results = []
for i in model:
    a, b, c = split_data_noloader(i)
    results.append((a, b, c))


def extract_counts(graphs, count_type='nodes'):
    if count_type == 'nodes':
        return [graph.num_nodes for graph in graphs]
    elif count_type == 'edges':
        return [graph.num_edges for graph in graphs]
    else:
        raise ValueError("count_type must be 'nodes' or 'edges'")

def set_text_size(ax, title_size, label_size, tick_size):
    ax.title.set_size(title_size)
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)
    ax.tick_params(axis='both', labelsize=tick_size)

def plot_graph_summary(node_counts, edge_counts_directed_dani, edge_counts_weighted_dani, edge_counts_cosine,
                       title_size=20, label_size=15, tick_size=10, legend_size=12):
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 7))  # Single chart

    # Convert histogram to line plot for each category and plot on the same axis
    def plot_line_hist(data, ax, label, color):
        counts, bin_edges = np.histogram(data, bins=40)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax.plot(np.log1p(bin_centers), counts, linestyle='-', marker='o', color=color, label=label)

    # Plot each dataset
    plot_line_hist(node_counts, ax, 'Nodes for all embedding', 'skyblue')
    plot_line_hist(edge_counts_directed_dani, ax, 'Edges for Directed DANI', 'lightgreen')
    plot_line_hist(edge_counts_weighted_dani, ax, 'Edges for Weighted DANI', 'gold')
    plot_line_hist(edge_counts_cosine, ax, 'Edges for Cosine similarity', 'salmon')

    # Add labels and legend
    # ax.set_title('Graph Data Summary')
    ax.set_xlabel('Number of Nodes or Edges')
    ax.set_ylabel('Number of Graphs')
    set_text_size(ax, title_size, label_size, tick_size)
    ax.legend(fontsize=legend_size)

    # Set custom tick labels for x-axis
    xticks = ax.get_xticks()
    ax.set_xticklabels([int(np.expm1(tick)) for tick in xticks])

    plt.tight_layout()
    plt.savefig(path.join(PROJECT_ROOT, "data", 'graphs_summary_combined.pdf'))
    plt.show()

# Sample graphs data would be needed here.
directed_dani = results[0][0] + results[0][1] + results[0][2]
cosine = results[1][0] + results[1][1] + results[1][2]
weighted_dani = results[2][0] + results[2][1] + results[2][2]

# Extracting data
node_counts = extract_counts(directed_dani, 'nodes')
edge_counts_directed_dani = extract_counts(directed_dani, 'edges')
edge_counts_cosine = extract_counts(cosine, 'edges')
edge_counts_weighted_dani = extract_counts(weighted_dani, 'edges')

# Plotting the graph with customizable text sizes
plot_graph_summary(node_counts, edge_counts_directed_dani, edge_counts_weighted_dani, edge_counts_cosine,
                   title_size=25, label_size=25, tick_size=25, legend_size=15)
