import networkx as nx
from networkx.algorithms.community import louvain_communities
import matplotlib.pyplot as plt
from perseus.dataset.preprocess.process import aggregate_data, assign_event_ids, get_graphs, process_dataframe, features_engineer
from perseus.dataset.preprocess.train_test_validate import get_test_scored_signals, get_train_scored_signals, get_valid_scored_signals
from perseus.dataset.compare_paper_graph import combine_features, graph_features


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


def draw_graph_with_communities(gs, key, communities_dict):
    # for key in gs.keys():
    #     print(key)
    G = gs[key]  # Retrieve the graph for the given key
    communities = communities_dict[key]  # Retrieve the communities for this graph

    # Create a mapping of node to community index
    node_to_community = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_to_community[node] = idx

    # Assign a unique color for each community
    colors = [node_to_community[node] for node in G.nodes()]

    # Draw the graph
    pos = nx.shell_layout(G)  # For circular layout
    nx.draw(G, pos, node_color=colors, with_labels=True, cmap=plt.cm.tab20, node_size=500)
    plt.show()




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
    draw_graph_with_communities(gs, "BTC", communities_dict)


