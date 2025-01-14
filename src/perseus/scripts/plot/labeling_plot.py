"""
This script is used to plot the graph with communities. It is used to visualize the communities detected in the graph.
"""

from os import path
import pickle
import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms.community import louvain_communities
import matplotlib.pyplot as plt
from perseus.dataset.preprocess.groudtruth_labeling import (
    read_labeling_csv_back_to_dict,
)
from perseus.dataset.preprocess.process import (
    aggregate_data,
    assign_event_ids,
    get_graphs,
    process_dataframe,
    features_engineer,
    compute_weighted_graph_features,
)
from perseus.dataset.preprocess.train_test_validate import (
    get_test_scored_signals,
    get_train_scored_signals,
    get_valid_scored_signals,
)
from perseus.dataset.dataset_preparation import combine_features, graph_features
from perseus.settings import PROJECT_ROOT


def community_detection_weighted(P_dict: dict):
    """
    This function detects communities in a graph using the Louvain method.
    """

    communities_dict = {}

    # Iterate over each key in the P_dict to process its edges and weights
    for key in P_dict.keys():
        G = nx.Graph()

        # Populate edge index and weights
        for (source_node, target_node), weight in P_dict[key].items():
            if weight > 0:  # Check if the weight is positive
                G.add_edge(source_node, target_node, weight=weight)
        communities = louvain_communities(G)
        communities_dict[key] = communities

    return communities_dict


def draw_graph_with_communities(
    gs: dict,
    key: str,
    communities_dict: dict,
    id_to_username: dict,
    labels: dict,
    base_node_size=1200,
    font_size=40,
    arrow_size=40,
    label_distance=0.2,  # Distance of the label from the node center
):
    """
    Draw the graph with communities and labels with adjustable text layout.
    Ensuring that labels are not cut off or overlapping.
    """
    G = gs[key]  # Retrieve the graph for the given key
    communities = communities_dict[key]  # Retrieve the communities for this graph

    # Create a mapping of node to community index
    node_to_community = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_to_community[node] = idx

    # Assign a unique color for each community
    colors = [node_to_community[node] for node in G.nodes()]

    # Map node sizes based on labels
    node_sizes = [
        base_node_size * (3 if labels.get(node, 0) == 1 else 1) for node in G.nodes()
    ]

    # Define the plot size (larger)
    plt.figure(figsize=(20, 20))

    # Draw the graph
    pos = nx.circular_layout(G)  # Circular layout to avoid overlap as much as possible
    nx.draw(
        G,
        pos,
        node_color=colors,
        cmap=plt.cm.tab20,
        node_size=node_sizes,
        arrowsize=arrow_size,  # Increased arrow size
    )

    # Adjusting node labels to prevent overlap and ensure visibility
    label_pos = {}
    for node, (x, y) in pos.items():
        angle = np.arctan2(y, x)
        offset_x = 4 * label_distance * np.cos(angle)
        offset_y = 3 * label_distance * np.sin(angle)
        label_pos[node] = (x + offset_x, y + offset_y)

    adjusted_labels = {node: id_to_username.get(node, str(node)) for node in G.nodes()}
    nx.draw_networkx_labels(
        G,
        pos=label_pos,
        labels=adjusted_labels,
        font_size=font_size,
        font_color="black",
    )

    # Ensure that all labels and elements are within the plot area
    plt.axis("equal")  # Ensure the aspect ratio does not distort distances
    plt.xlim(
        min(pos[x][0] for x in pos) - 2, max(pos[x][0] for x in pos) + 2
    )  # Adjust the x limits
    plt.ylim(
        min(pos[x][1] for x in pos) - 2, max(pos[x][1] for x in pos) + 2
    )  # Adjust the y limits

    # Save the graph to a PDF
    plt.tight_layout()  # Adjust layout to prevent cutoff
    plt.savefig(
        path.join(PROJECT_ROOT, "data", "community.pdf")
    )  # Adjust path as necessary
    plt.show()


def draw_graph_with_communities_prediction(
    gs: dict,
    key: str,
    communities_dict: dict,
    id_to_username: dict,
    predictions: dict,
    base_node_size=1200,
    font_size=40,
    arrow_size=40,
    label_distance=0.2,  # Distance of the label from the node center
):
    """
    Draw the graph with communities and labels with adjustable text layout.
    Ensuring that labels are not cut off or overlapping.
    """
    G = gs[key]  # Retrieve the graph for the given key
    communities = communities_dict[key]  # Retrieve the communities for this graph

    # Create a mapping of node to community index
    node_to_community = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_to_community[node] = idx

    # Assign a unique color for each community
    colors = [node_to_community[node] for node in G.nodes()]

    node_sizes = [
        (
            0.1 * base_node_size
            if predictions.get(node, 0) == (1, 0)
            else (
                10 * base_node_size
                if predictions.get(node, 0) == (0, 1)
                else (
                    3 * base_node_size
                    if predictions.get(node, 0) == (1, 1)
                    else base_node_size
                )
            )
        )
        for node in G.nodes()
    ]

    # Define the plot size (larger)
    plt.figure(figsize=(20, 20))

    # Draw the graph
    pos = nx.circular_layout(G)  # Circular layout to avoid overlap as much as possible
    nx.draw(
        G,
        pos,
        node_color=colors,
        cmap=plt.cm.tab20,
        node_size=node_sizes,
        arrowsize=arrow_size,  # Increased arrow size
    )

    # Adjusting node labels to prevent overlap and ensure visibility
    label_pos = {}
    for node, (x, y) in pos.items():
        angle = np.arctan2(y, x)
        offset_x = 4 * label_distance * np.cos(angle)
        offset_y = 3 * label_distance * np.sin(angle)
        label_pos[node] = (x + offset_x, y + offset_y)

    adjusted_labels = {node: id_to_username.get(node, str(node)) for node in G.nodes()}
    nx.draw_networkx_labels(
        G,
        pos=label_pos,
        labels=adjusted_labels,
        font_size=font_size,
        font_color="black",
    )

    # Ensure that all labels and elements are within the plot area
    plt.axis("equal")  # Ensure the aspect ratio does not distort distances
    plt.xlim(
        min(pos[x][0] for x in pos) - 2, max(pos[x][0] for x in pos) + 2
    )  # Adjust the x limits
    plt.ylim(
        min(pos[x][1] for x in pos) - 2, max(pos[x][1] for x in pos) + 2
    )  # Adjust the y limits

    # Save the graph to a PDF
    plt.tight_layout()  # Adjust layout to prevent cutoff
    # plt.savefig(
    #     path.join(PROJECT_ROOT, "data", "community.pdf")
    # )  # Adjust path as necessary
    plt.show()


def draw_graph_with_communities_wrong_case(
    gs: dict,
    key: str,
    communities_dict: dict,
    id_to_username: dict,
    predictions: dict,
    base_node_size=1200,
    font_size=40,
    arrow_size=40,
    label_distance=0.4,  # Distance of the label from the node center
):
    """
    Draw the graph with communities and labels with adjustable text layout.
    Ensuring that labels are not cut off or overlapping.
    """
    G = gs[key]  # Retrieve the graph for the given key
    communities = communities_dict[key]  # Retrieve the communities for this graph

    # Create a mapping of node to community index
    node_to_community = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_to_community[node] = idx

    # Assign a unique color for each community
    colors = [node_to_community[node] for node in G.nodes()]

    # Define node sizes and shapes based on predictions
    node_sizes = []
    node_shapes = {}
    for node in G.nodes():
        prediction = predictions.get(node, 0)
        size = (
            3 * base_node_size
            if prediction in [(1, 0), (0, 1), (1, 1)]
            else base_node_size
        )
        node_sizes.append(size)
        if prediction == (1, 0):
            node_shapes[node] = "s"  # Triangle false positive
        elif prediction == (0, 1):
            node_shapes[node] = "^"  # Square false negative
        else:
            node_shapes[node] = "o"  # Circle

    # Define the plot size (larger)
    plt.figure(figsize=(20, 20))

    # Draw the graph with a circular layout
    pos = nx.circular_layout(G)  # Circular layout to avoid overlap as much as possible

    # Draw nodes based on their shapes
    for shape in set(node_shapes.values()):
        nodes_with_shape = [node for node in G.nodes() if node_shapes[node] == shape]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes_with_shape,
            node_size=[
                node_sizes[list(G.nodes()).index(node)] for node in nodes_with_shape
            ],
            node_color=[
                colors[list(G.nodes()).index(node)] for node in nodes_with_shape
            ],
            cmap=plt.cm.tab20,
            node_shape=shape,
        )

    # Draw the edges
    nx.draw_networkx_edges(G, pos, arrowsize=arrow_size)

    # Adjusting node labels to prevent overlap and ensure visibility
    label_pos = {}
    for node, (x, y) in pos.items():
        angle = np.arctan2(y, x)
        offset_x = 5 * label_distance * np.cos(angle)
        offset_y = 3 * label_distance * np.sin(angle)
        label_pos[node] = (x + offset_x, y + offset_y)

    adjusted_labels = {node: id_to_username.get(node, str(node)) for node in G.nodes()}
    nx.draw_networkx_labels(
        G,
        pos=label_pos,
        labels=adjusted_labels,
        font_size=font_size,
        font_color="black",
    )

    # Ensure that all labels and elements are within the plot area
    plt.axis("equal")  # Ensure the aspect ratio does not distort distances

    # Hiding the spines (the box around the plot)
    ax = plt.gca()  # Get the current axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    plt.xlim(
        min(pos[x][0] for x in pos) - 4, max(pos[x][0] for x in pos) + 4
    )  # Adjust the x limits
    plt.ylim(
        min(pos[x][1] for x in pos) - 4, max(pos[x][1] for x in pos) + 4
    )  # Adjust the y limits

    # Save the graph to a PDF
    plt.tight_layout()  # Adjust layout to prevent cutoff
    plt.savefig(
        path.join(PROJECT_ROOT, "data", "community_DIA.pdf")
    )  # Adjust path as necessary

    plt.show()


# Update the function call in the main block to include id_to_username:
if __name__ == "__main__":
    signals = get_train_scored_signals()
    processed_signals = process_dataframe(signals)
    ided_signals = assign_event_ids(processed_signals)
    cascade_buffer, no_nodes_buffer, id_mapping_buffer, cascade_labeling = (
        aggregate_data(ided_signals)
    )
    gs, results, As, P_dict, P_com = get_graphs(
        cascade_buffer, no_nodes_buffer, id_mapping_buffer
    )
    graph_feature = graph_features(gs)
    market_feature = features_engineer(processed_signals)
    weighted_feature = compute_weighted_graph_features(P_com)
    combine_feature = combine_features(market_feature, graph_feature, weighted_feature)

    communities_dict = community_detection_weighted(P_dict)
    # Assume id_to_username is available here

    telegram_id = pd.read_csv(path.join(PROJECT_ROOT, "data", "telegram_id.csv"))
    id_to_username = dict(zip(telegram_id.telegram_chat_id, telegram_id.username))

    coin = "NEBL"

    # Replacing the first element of each tuple in each sublist with the corresponding username
    cascade_labeling[coin] = [
        [
            tuple(
                [id_to_username[t[0]] if t[0] in id_to_username else t[0]] + list(t[1:])
            )
            for t in sublist
        ]
        for sublist in cascade_labeling[coin]
    ]

    # draw_graph_with_communities(gs, "ACA", communities_dict, id_to_username)
    # draw("ACA", gs, communities_dict, id_to_username)

    labels = read_labeling_csv_back_to_dict("train")
    # labels["ACA"] = {id_to_username[k]: v for k, v in labels["ACA"].items()}
    draw_graph_with_communities(
        gs, coin, communities_dict, id_to_username, labels[coin]
    )

    labeling_df = []
    for i, v in enumerate(cascade_labeling[coin]):
        for j in v:
            labeling_df.append((i, j[0], j[7], j[10], j[2]))

    labeling_df = pd.DataFrame(
        labeling_df,
        columns=["Events", "Telegram Channels", "Timestamps", "Messages", "Returns"],
    )
    labeling_df.to_csv(
        path.join(PROJECT_ROOT, "data", "labeling_NEBL.csv"), index=False
    )

    # load the prediction labels pickle file from the data folder
    with open(path.join(PROJECT_ROOT, "data", "predictions.pkl"), "rb") as file:
        predictions = pickle.load(file)

    # for k,v in predictions.items():
    #     print(k)
    #     draw_graph_with_communities_prediction(gs, k, communities_dict, id_to_username, predictions[k])

    coin = "DIA"

    true_false_labels = {}

    for coin in labels.keys():
        true_false_labels[coin] = {}  # Initialize dictionary for each coin
        for i in labels[coin].keys():
            try:
                true_false_labels[coin][i] = (labels[coin][i], predictions[coin][i][0])
            except KeyError:
                # Skip if prediction for this label doesn't exist
                pass

    # go over each coin and node in each coin and output the key with at least (1,0) and (0,1) and (1,1)
    for coin in true_false_labels.keys():
        # for node in true_false_labels[coin].keys():
        # Convert true_false_labels[coin][node] to a set for easy checking
        label_set = set(v for i, v in true_false_labels[coin].items())

        # Check if all required tuples are in the set
        if {(1, 0), (0, 1), (1, 1)}.issubset(label_set):
            print(coin)

    draw_graph_with_communities_wrong_case(
        gs, coin, communities_dict, id_to_username, true_false_labels
    )
