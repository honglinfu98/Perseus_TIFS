from os import path
import pandas as pd
import networkx as nx
from networkx.algorithms.community import louvain_communities
import matplotlib.pyplot as plt
from perseus.dataset.preprocess.process import aggregate_data, assign_event_ids, get_graphs, process_dataframe, features_engineer
from perseus.dataset.preprocess.train_test_validate import get_test_scored_signals, get_train_scored_signals, get_valid_scored_signals
from perseus.dataset.compare_paper_graph import combine_features, graph_features
from perseus.settings import PROJECT_ROOT


def community_detection_weighted(P_dict: dict):
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



def draw_graph_with_communities(gs, key, communities_dict, id_to_username, node_size=500, font_size=8):
    G = gs[key]  # Retrieve the graph for the given key
    communities = communities_dict[key]  # Retrieve the communities for this graph

    # Create a mapping of node to community index
    node_to_community = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_to_community[node] = idx

    # Assign a unique color for each community
    colors = [node_to_community[node] for node in G.nodes()]

    # Define the plot size (larger)
    plt.figure(figsize=(12, 12))  # Increase this to give more room

    # Draw the graph
    pos = nx.circular_layout(G)  # Change layout to kamada_kawai layout
    # pos = nx.spring_layout(G, k=0.5)  # Change layout to spring layout with more space between nodes
    labels = {node: id_to_username[node] if node in id_to_username else str(node) for node in G.nodes()}  # Map nodes to usernames

    nx.draw(G, pos, node_color=colors, with_labels=True, labels=labels, cmap=plt.cm.tab20, node_size=node_size, font_size=font_size)

    # Save the graph to a PDF
    plt.savefig(path.join(PROJECT_ROOT, "data", 'embedding.pdf'))
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


    a = pd.read_csv(path.join(PROJECT_ROOT, "data", "telegram_id.csv"))
    id_to_username = dict(zip(a.telegram_chat_id, a.username))

    # Replacing the first element of each tuple in each sublist with the corresponding username
    cascade_labeling["POWR"] = [
        [tuple([id_to_username[t[0]] if t[0] in id_to_username else t[0]] + list(t[1:])) for t in sublist]
        for sublist in cascade_labeling["POWR"]
    ]

    draw_graph_with_communities(gs, "POWR", communities_dict, id_to_username)









