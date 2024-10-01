from os import path
import matplotlib.pyplot as plt
import networkx as nx
from perseus.settings import PROJECT_ROOT


def draw_custom_edge_labels(G, pos, offset_factor=0.9, weight_font_size=12):
    """
    Draw edge labels with a custom offset from the edge midpoint
    """
    for u, v, data in G.edges(data=True):
        weight = data["weight"]
        x_midpoint, y_midpoint = (pos[u][0] + pos[v][0]) / 2, (
            pos[u][1] + pos[v][1]
        ) / 2
        dx, dy = pos[v][0] - pos[u][0], pos[v][1] - pos[u][1]
        norm = (dx**2 + dy**2) ** 0.5
        if norm == 0:
            norm = 1
        dx, dy = dx / norm, dy / norm
        offset_x, offset_y = dy * offset_factor, -dx * offset_factor

        plt.text(
            x_midpoint + offset_x,
            y_midpoint + offset_y,
            str(weight),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=weight_font_size,
            color="black",  # Changed from conditional coloring to always black
            fontweight="bold",  # Bold font
        )


if __name__ == "__main__":

    pos = {"m": (0, 0), "n1": (0.5, 0.86), "n2": (1, 0), "n3": (0.5, -0.86)}
    node_color = "lightblue"
    node_size = 9500
    font_size = 80
    weight_font_size = 60
    arrow_size = 160

    # Initialize directed graph G1
    G1 = nx.DiGraph()
    edge_1 = [("m", "n1", 1), ("n1", "n2", 1), ("n2", "n3", 1)]
    for u, v, _ in edge_1:
        G1.add_edge(u, v)

    # Plotting
    plt.figure(figsize=(10, 10))
    nx.draw_networkx_edges(
        G1, pos, edge_color="gray", arrowstyle="-|>", arrows=True, arrowsize=arrow_size
    )
    nx.draw_networkx_nodes(G1, pos, node_color=node_color, node_size=node_size)
    nx.draw_networkx_labels(
        G1, pos, font_size=font_size, font_color="black", font_weight="bold"
    )  # Add labels to the nodes
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path.join(PROJECT_ROOT, "data", "graph1.pdf"), format="pdf")
    plt.show()
    plt.close()

    # Initialize Graph G2
    G2 = nx.MultiDiGraph()
    edges_2 = [
        ("m", "n1", 0.5),
        ("n1", "m", 0.5),
        ("m", "n2", 0.8),
        ("n2", "m", 0.8),
        ("m", "n3", 0.6),
        ("n3", "m", 0.6),
        ("n1", "n2", 0.7),
        ("n2", "n1", 0.7),
        ("n1", "n3", 0.9),
        ("n3", "n1", 0.9),
        ("n2", "n3", 0.4),
        ("n3", "n2", 0.4),
    ]
    for u, v, w in edges_2:
        G2.add_edge(u, v, weight=w)

    # Plot settings for G2
    plt.figure(figsize=(10, 10))
    nx.draw_networkx_edges(
        G2,
        pos,
        edge_color="gray",
        arrowstyle="-|>",
        arrows=True,
        arrowsize=arrow_size,
        connectionstyle="arc3,rad=0.25",
    )
    nx.draw_networkx_nodes(G2, pos, node_color=node_color, node_size=node_size)
    nx.draw_networkx_labels(
        G2, pos, font_size=font_size, font_color="black", font_weight="bold"
    )
    draw_custom_edge_labels(
        G2, pos, offset_factor=0.17, weight_font_size=weight_font_size
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path.join(PROJECT_ROOT, "data", "graph2.pdf"), format="pdf")
    plt.show()
    plt.close()

    G3 = nx.MultiDiGraph()
    edges_3 = [
        ("m", "n1", 1.0),
        ("n1", "m", 0.8),
        ("n1", "n2", 0.8),
        ("n2", "n1", 0.6),
        ("n2", "n3", 0.6),
        ("n3", "n2", 0.4),
    ]
    for u, v, w in edges_3:
        G3.add_edge(u, v, weight=w)
    # Plot settings for G2
    plt.figure(figsize=(10, 10))
    nx.draw_networkx_edges(
        G3,
        pos,
        edge_color="gray",
        arrowstyle="-|>",
        arrows=True,
        arrowsize=arrow_size,
        connectionstyle="arc3,rad=0.2",
    )
    nx.draw_networkx_nodes(G3, pos, node_color=node_color, node_size=node_size)
    nx.draw_networkx_labels(
        G3, pos, font_size=font_size, font_color="black", font_weight="bold"
    )
    draw_custom_edge_labels(
        G3, pos, offset_factor=0.1, weight_font_size=weight_font_size
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path.join(PROJECT_ROOT, "data", "graph3.pdf"), format="pdf")
    plt.show()
    plt.close()
