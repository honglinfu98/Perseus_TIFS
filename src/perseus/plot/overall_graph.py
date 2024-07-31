from os import path
import networkx as nx
import matplotlib.pyplot as plt
from perseus.dataset.preprocess.process import (
    aggregate_data,
    assign_event_ids,
    get_graphs,
    process_dataframe,
)
from perseus.dataset.preprocess.train_test_validate import get_train_scored_signals
from perseus.settings import PROJECT_ROOT


def combine_and_plot_graphs(
    graph_dict, layout="spring", figsize=(12, 12), edge_alpha=0.5, edge_width=0.5
):
    # Create a new graph to combine all the individual graphs
    combined_graph = nx.Graph()

    # Iterate over all the graphs in the dictionary
    for graph in graph_dict.values():
        # Add nodes and edges from each graph to the combined graph
        combined_graph.add_nodes_from(graph.nodes(data=True))
        combined_graph.add_edges_from(graph.edges(data=True))

    # Choose layout based on the input parameter
    if layout == "spring":
        pos = nx.spring_layout(combined_graph)  # Spring layout
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(combined_graph)  # Kamada-Kawai layout
    elif layout == "circular":
        pos = nx.circular_layout(combined_graph)  # Circular layout
    elif layout == "random":
        pos = nx.random_layout(combined_graph)  # Random layout
    elif layout == "shell":
        pos = nx.shell_layout(combined_graph)  # Shell layout
    else:
        raise ValueError("Unsupported layout type")

    # Create the plot with a specified figure size
    plt.figure(figsize=figsize)
    # Plotting the combined graph with transparency in edges
    nx.draw(
        combined_graph,
        pos,
        with_labels=True,
        node_size=5000,
        node_color="red",
        edge_color="blue",
        width=edge_width,
        alpha=edge_alpha,
        font_size=10,
        font_weight="bold",
    )
    plt.title("Combined Graph using {} Layout".format(layout.capitalize()))
    plt.savefig(path.join(PROJECT_ROOT, "data", "overall_graph.pdf"))

    plt.show()


# Example usage:

if __name__ == "__main__":
    signals = get_train_scored_signals()
    processed_signals = process_dataframe(signals)
    ided_signals = assign_event_ids(processed_signals)
    cascade, no_nodes, id_mapping, cascade_labeling = aggregate_data(ided_signals)
    gs, results, As, P_dict = get_graphs(cascade, no_nodes, id_mapping)
    combine_and_plot_graphs(
        gs, layout="kamada_kawai", figsize=(100, 100), edge_alpha=1, edge_width=1.5
    )
