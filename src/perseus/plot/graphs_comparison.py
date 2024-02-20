import matplotlib.pyplot as plt
import networkx as nx


# Updated function to include font size for edge weights
def draw_custom_edge_labels(
    G, pos, edge_weights, offset_factor=0.1, weight_font_size=12
):
    for (u, v), weight in edge_weights.items():
        x_midpoint, y_midpoint = (pos[u][0] + pos[v][0]) / 2, (
            pos[u][1] + pos[v][1]
        ) / 2
        dx, dy = pos[v][0] - pos[u][0], pos[v][1] - pos[u][1]
        norm = (dx**2 + dy**2) ** 0.5
        if norm == 0:  # Avoid division by zero
            norm = 1
        dx, dy = dx / norm, dy / norm  # Normalize
        offset_x, offset_y = (
            dy * offset_factor,
            -dx * offset_factor,
        )  # Perpendicular offset

        plt.text(
            x_midpoint + offset_x,
            y_midpoint + offset_y,
            str(weight),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=weight_font_size,
            color="blue",
        )  # Use the passed font size
        reverse_weight = edge_weights.get((v, u), "")
        if reverse_weight:
            plt.text(
                x_midpoint - offset_x,
                y_midpoint - offset_y,
                str(reverse_weight),
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=weight_font_size,
                color="red",
            )  # Use the passed font size


# Settings
node_color = "lightblue"
node_size = 6000
mastermind_size = 8000
edge_width = 60
arrow_size = 60
font_size = 60
weight_font_size = 20  # Define a variable for weight font size for easy adjustment

# Define positions for nodes
pos = {"m": (0, 0), "p1": (0.5, 0.86), "p2": (1, 0), "p3": (0.5, -0.86)}

# GRAPH 1: Dominant DANI (No title)
G1 = nx.DiGraph()
G1.add_edges_from([("m", "p1"), ("p1", "p2"), ("p2", "p3")])
plt.figure(figsize=(8, 6))
nx.draw(
    G1,
    pos,
    with_labels=True,
    node_color=node_color,
    node_size=[mastermind_size if node == "m" else node_size for node in G1],
    arrowsize=arrow_size,
    font_size=font_size,
    edge_color="gray",
)
plt.axis("off")
plt.tight_layout()
plt.savefig("graph1.pdf", format="pdf")

plt.show()
plt.close()

# GRAPH 2: Cosine Matrix (No title)
G2 = nx.DiGraph()
edges_2 = [
    ("m", "p1", 0.5),
    ("p1", "m", 0.5),
    ("m", "p2", 0.8),
    ("p2", "m", 0.8),
    ("m", "p3", 0.6),
    ("p3", "m", 0.6),
    ("p1", "p2", 0.7),
    ("p2", "p1", 0.7),
    ("p1", "p3", 0.9),
    ("p3", "p1", 0.9),
    ("p2", "p3", 0.4),
    ("p3", "p2", 0.4),
]
G2.add_weighted_edges_from(edges_2)
plt.figure(figsize=(8, 6))
nx.draw(
    G2,
    pos,
    with_labels=True,
    node_color=node_color,
    node_size=[mastermind_size if node == "m" else node_size for node in G2],
    arrowsize=arrow_size,
    font_size=font_size,
    edge_color="gray",
)
edge_weights_2 = nx.get_edge_attributes(G2, "weight")
draw_custom_edge_labels(
    G2, pos, edge_weights_2, offset_factor=0.2, weight_font_size=weight_font_size
)
plt.axis("off")
plt.tight_layout()
plt.savefig("graph2.pdf", format="pdf")

plt.show()
plt.close()

# GRAPH 3: DANI Diffusion (No title)
G3 = nx.DiGraph()
edges_3 = [
    ("m", "p1", 1.0),
    ("p1", "m", 0.8),
    ("p1", "p2", 0.8),
    ("p2", "p1", 0.6),
    ("p2", "p3", 0.6),
    ("p3", "p2", 0.4),
]
G3.add_weighted_edges_from(edges_3)
plt.figure(figsize=(8, 6))
nx.draw(
    G3,
    pos,
    with_labels=True,
    node_color=node_color,
    node_size=[mastermind_size if node == "m" else node_size for node in G3],
    arrowsize=arrow_size,
    font_size=font_size,
    edge_color="gray",
)
edge_weights_3 = nx.get_edge_attributes(G3, "weight")
draw_custom_edge_labels(
    G3, pos, edge_weights_3, offset_factor=0.2, weight_font_size=weight_font_size
)
plt.axis("off")
plt.tight_layout()
plt.savefig("graph3.pdf", format="pdf")

plt.show()
plt.close()
