"""
This script is used to plot the graph with communities. It is used to visualize the communities detected in the graph.
"""

from os import path
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
        min(pos[x][0] for x in pos) - 1.5, max(pos[x][0] for x in pos) + 1.5
    )  # Adjust the x limits
    plt.ylim(
        min(pos[x][1] for x in pos) - 1.5, max(pos[x][1] for x in pos) + 1.5
    )  # Adjust the y limits

    # Save the graph to a PDF
    plt.tight_layout()  # Adjust layout to prevent cutoff
    plt.savefig(
        path.join(PROJECT_ROOT, "data", "community.pdf")
    )  # Adjust path as necessary
    plt.show()


# Update the function call in the main block to include id_to_username:
if __name__ == "__main__":
    signals = get_train_scored_signals()
    processed_signals = process_dataframe(signals)
    ided_signals = assign_event_ids(processed_signals)
    cascade, no_nodes, id_mapping, cascade_labeling = aggregate_data(ided_signals)
    gs, results, As, P_dict = get_graphs(cascade, no_nodes, id_mapping)
    graph_feature = graph_features(gs)
    market_feature = features_engineer(processed_signals)
    combine_feature = combine_features(market_feature, graph_feature)
    communities_dict = community_detection_weighted(P_dict)
    # Assume id_to_username is available here

    telegram_id = pd.read_csv(path.join(PROJECT_ROOT, "data", "telegram_id.csv"))
    id_to_username = dict(zip(telegram_id.telegram_chat_id, telegram_id.username))

    # Replacing the first element of each tuple in each sublist with the corresponding username
    cascade_labeling["ACA"] = [
        [
            tuple(
                [id_to_username[t[0]] if t[0] in id_to_username else t[0]] + list(t[1:])
            )
            for t in sublist
        ]
        for sublist in cascade_labeling["ACA"]
    ]

    # draw_graph_with_communities(gs, "ACA", communities_dict, id_to_username)
    # draw("ACA", gs, communities_dict, id_to_username)

    labels = read_labeling_csv_back_to_dict("train")
    # labels["ACA"] = {id_to_username[k]: v for k, v in labels["ACA"].items()}
    draw_graph_with_communities(
        gs, "ACA", communities_dict, id_to_username, labels["ACA"]
    )
