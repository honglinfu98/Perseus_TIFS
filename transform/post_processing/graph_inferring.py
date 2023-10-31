from matplotlib import pyplot as plt
import pandas as pd
import networkx as nx

from transform.pre_processing_summary.scored_signals import (
    assign_event_ids,
    process_dataframe,
    aggregate_data,
)
from extract.cloudburst_connection import get_scored_signals
from transform.DANI import DANI


def get_graphs(cascade: dict, no_nodes: dict, id_mapping: dict):
    """
    This function creates the graphs for the cascadeX
    :param cascade: the cascade
    :param no_nodes: the number of nodes for each commodity
    :param id_mapping: the mapping between the commodity and the id
    :return: the graphs
    """
    ensure_graph_learned = {}
    # Filter the no_nodes that have more than 3 nodes and have more than no nodes cascade
    for key, value in no_nodes.items():
        if value > 3 and value < len(cascade[key]):
            ensure_graph_learned[key] = value

    # Create the graphs for each commodity using moer_than_three
    graphs = {}
    for key, value in ensure_graph_learned.items():
        graphs[key], _, _ = DANI(ensure_graph_learned[key], cascade[key])
        graphs[key] = nx.relabel_nodes(graphs[key], id_mapping[key]["new_to_id"])

    return graphs


def btw_cen_vs_avg_return_by_node(processed_signals: pd.DataFrame, graphs: dict):
    # Group by commodity and telegram_chat_id
    grouped = processed_signals.groupby(["commodity", "telegram_chat_id"])

    # Define aggregation functions
    aggregations = {
        "increase_percentage": "mean",  # average increase
    }

    # Perform groupby and aggregation
    feature_df = grouped.agg(aggregations).reset_index()

    # Rename columns
    feature_df.rename(
        columns={
            "increase_percentage": "average_increase",
        },
        inplace=True,
    )

    # Loop through each commodity and its corresponding graph
    for key, g in graphs.items():
        # Compute betweenness centrality
        btw_cen = nx.betweenness_centrality(g)

        # Initialize lists to store values for plotting
        plot_btw_cen = []
        plot_avg_increases = []
        plot_labels = []

        for node in g.nodes():
            # Check if node has a corresponding entry in feature_df
            node_data = feature_df[
                (feature_df["commodity"] == key)
                & (feature_df["telegram_chat_id"] == node)
            ]
            if not node_data.empty:
                avg_increase = node_data["average_increase"].values[0]
                plot_btw_cen.append(btw_cen[node])
                plot_avg_increases.append(avg_increase)
                plot_labels.append(node)

        # Plot
        plt.figure(figsize=(14, 8))
        plt.scatter(plot_btw_cen, plot_avg_increases)

        # Annotate points with telegram_chat_id
        for i, label in enumerate(plot_labels):
            plt.annotate(label, (plot_btw_cen[i], plot_avg_increases[i]))

        plt.xlabel("Betweenness Centrality")
        plt.ylabel("Average Increase")
        plt.title(f"Betweenness Centrality vs Average Increase for {key}")
        plt.show()


def btw_cen_vs_total_return_by_node(processed_signals: pd.DataFrame, graphs: dict):
    # Calculate total return for each group
    def total_return(x):
        return (x + 1).prod() - 1

    # Group by commodity and telegram_chat_id and aggregate
    feature_df = (
        processed_signals.groupby(["commodity", "telegram_chat_id"])
        .agg({"increase_percentage": total_return})
        .reset_index()
    )

    # Rename columns
    feature_df.rename(
        columns={
            "increase_percentage": "total_return",
        },
        inplace=True,
    )

    # Loop through each commodity and its corresponding graph
    for key, g in graphs.items():
        # Compute betweenness centrality
        btw_cen = nx.betweenness_centrality(g)

        # Initialize lists to store values for plotting
        plot_btw_cen = []
        plot_total_returns = []
        plot_labels = []

        for node in g.nodes():
            # Check if node has a corresponding entry in feature_df
            node_data = feature_df[
                (feature_df["commodity"] == key)
                & (feature_df["telegram_chat_id"] == node)
            ]
            if not node_data.empty:
                total_return_val = node_data["total_return"].values[0]
                plot_btw_cen.append(btw_cen[node])
                plot_total_returns.append(total_return_val)
                plot_labels.append(node)

        # Plot
        plt.figure(figsize=(14, 8))
        plt.scatter(plot_btw_cen, plot_total_returns)

        # Annotate points with telegram_chat_id
        for i, label in enumerate(plot_labels):
            plt.annotate(label, (plot_btw_cen[i], plot_total_returns[i]))

        plt.xlabel("Betweenness Centrality")
        plt.ylabel("Total Return")
        plt.title(f"Total Return vs Betweenness Centrality for {key}")
        plt.show()


def draw_graph_with_avg_return(processed_signals: pd.DataFrame, graphs: dict):
    # Group by commodity and telegram_chat_id
    grouped = processed_signals.groupby(["commodity", "telegram_chat_id"])

    # Define aggregation functions
    aggregations = {
        "increase_percentage": "mean",  # average increase
    }

    # Perform groupby and aggregation
    feature_df = grouped.agg(aggregations).reset_index()

    # Rename columns
    feature_df.rename(
        columns={
            "increase_percentage": "average_increase",
        },
        inplace=True,
    )

    # Loop through each commodity and its corresponding graph
    for key, g in graphs.items():
        # Initialize lists to store values for plotting
        node_sizes = []
        labels = {}

        for node in g.nodes():
            # Check if node has a corresponding entry in feature_df
            node_data = feature_df[
                (feature_df["commodity"] == key)
                & (feature_df["telegram_chat_id"] == node)
            ]
            if not node_data.empty:
                avg_increase = node_data["average_increase"].values[0]
                node_sizes.append(
                    avg_increase * 100000
                )  # Multiply by a factor to make it visible
                labels[node] = node
            else:
                node_sizes.append(50)  # Default size for nodes without data

        # Draw the graph
        plt.figure(figsize=(14, 8))
        # pos = nx.spring_layout(g)  # Define the layout for the graph
        pos = nx.circular_layout(g)
        nx.draw(g, pos, with_labels=False, node_size=node_sizes, alpha=0.6)
        nx.draw_networkx_labels(g, pos, labels, font_size=10)

        plt.title(f"Graph for {key} with Node Size by Average Increase")
        plt.show()


def in_out_cen_vs_avg_return_by_node(processed_signals: pd.DataFrame, graphs: dict):
    # Group by commodity and telegram_chat_id
    grouped = processed_signals.groupby(["commodity", "telegram_chat_id"])

    # Define aggregation functions
    aggregations = {
        "increase_percentage": "mean",  # average increase
    }

    # Perform groupby and aggregation
    feature_df = grouped.agg(aggregations).reset_index()

    # Rename columns
    feature_df.rename(
        columns={
            "increase_percentage": "average_increase",
        },
        inplace=True,
    )

    # Loop through each commodity and its corresponding graph
    for key, g in graphs.items():
        # Compute in_degree_centrality and out_degree_centrality
        in_cen = nx.in_degree_centrality(g)
        out_cen = nx.out_degree_centrality(g)

        # Initialize lists to store values for plotting
        plot_in_cen = []
        plot_out_cen = []
        plot_sizes = []
        plot_labels = []

        for node in g.nodes():
            # Check if node has a corresponding entry in feature_df
            node_data = feature_df[
                (feature_df["commodity"] == key)
                & (feature_df["telegram_chat_id"] == node)
            ]
            if not node_data.empty:
                avg_increase = node_data["average_increase"].values[0]
                plot_in_cen.append(in_cen[node])
                plot_out_cen.append(out_cen[node])
                plot_sizes.append(
                    avg_increase * 10000
                )  # Multiply by a factor to make it visible
                plot_labels.append(node)

        # Plot
        plt.figure(figsize=(14, 8))
        plt.scatter(plot_in_cen, plot_out_cen, s=plot_sizes, alpha=0.6)

        # Annotate points with telegram_chat_id
        for i, label in enumerate(plot_labels):
            plt.annotate(label, (plot_in_cen[i], plot_out_cen[i]))

        plt.xlabel("In-Degree Centrality")
        plt.ylabel("Out-Degree Centrality")
        plt.title(f"In-Degree Centrality vs Out-Degree Centrality for {key}")
        plt.show()


if __name__ == "__main__":
    signals = get_scored_signals()
    processed_signals = process_dataframe(signals)
    ided_signals = assign_event_ids(processed_signals)
    cascade, no_nodes, id_mapping = aggregate_data(ided_signals)
    gs = get_graphs(cascade, no_nodes, id_mapping)
    # btw_cen_vs_avg_return_by_node(ided_signals, gs)
    # in_out_cen_vs_avg_return_by_node(processed_signals,gs)
    draw_graph_with_avg_return(processed_signals, gs)
