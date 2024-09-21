from os import path
import pickle
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from perseus.settings import PROJECT_ROOT

datasets = ["DDINA", "COSS", "DDM"]
dataset_colors = {"DDINA": "blue", "COSS": "green", "DDM": "red"}
models = ["GCN", "GAT", "GraphSAGE"]
label = 1
dataset_labels = {
    "DDINA": "Directed Diffusion",
    "COSS": "Cosine Similarity",
    "DDM": "Weighted Diffusion",
}

with open(path.join(PROJECT_ROOT, "data", "results_com.pkl"), "rb") as file:
    results = pickle.load(file)

with open(path.join(PROJECT_ROOT, "data", "results1_com.pkl"), "rb") as file:
    results1 = pickle.load(file)


def plot_batch_time_vs_nodes(
    results,
    ax,
    model_name,
    dataset_colors,
    title,
    show_legend=True,
    show_ylabel=True,
    show_xlabel=True,
):
    """
    Plot batch time versus number of nodes for each dataset using line plots with spline interpolation,
    handling duplicates by averaging batch times for the same number of nodes.
    """
    for (
        dataset
    ) in results:  # This should reference the keys of 'results', which was missing
        batch_times = np.array(results[dataset][model_name]["batch_times"])
        num_nodes = np.array(results[dataset][model_name]["num_nodes"])
        unique_nodes, indices = np.unique(num_nodes, return_inverse=True)
        average_batch_times = np.zeros_like(unique_nodes, dtype=float)
        for i, node in enumerate(unique_nodes):
            average_batch_times[i] = np.mean(batch_times[indices == i])
        if len(unique_nodes) > 3:
            spline = make_interp_spline(unique_nodes, average_batch_times, k=3)
            fine_x = np.linspace(unique_nodes.min(), unique_nodes.max(), 500)
            fine_y = spline(fine_x)
            ax.plot(
                fine_x,
                fine_y,
                label=dataset_labels[dataset],  # Ensure 'dataset_labels' is defined
                color=dataset_colors[dataset],
            )
        else:
            ax.plot(
                unique_nodes,
                average_batch_times,
                "o-",
                label=dataset_labels[dataset],  # Ensure 'dataset_labels' is defined
                color=dataset_colors[dataset],
            )
    ax.set_title(title, fontsize=fontsize_title)
    if show_xlabel:
        ax.set_xlabel("Number of Nodes", fontsize=fontsize_labels)
    if show_ylabel:
        ax.set_ylabel("Inference Speed (sec)", fontsize=fontsize_labels)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.5f}"))
    if show_legend:
        ax.legend(
            fontsize=fontsize_legend, title_fontsize=fontsize_legend, loc="lower right"
        )
    ax.tick_params(axis="both", which="major", labelsize=fontsize_ticks)


def save_batch_time_vs_nodes_plot(
    fontsize_title, fontsize_labels, fontsize_legend, fontsize_ticks
):
    fig, axs = plt.subplots(
        len(models), 1, figsize=(12, 24)
    )  # Adjusted to vertical layout
    for i, model_name in enumerate(models):
        plot_batch_time_vs_nodes(
            results,
            axs[i],
            model_name,
            dataset_colors,
            title=f"{model_name}",
            show_legend=False,  # Legend only on the last plot
            show_ylabel=True,  # Include Y-axis label for 'results'
            show_xlabel=(i == len(models) - 1),  # X-axis label only on the last plot
        )
    plt.tight_layout()
    plt.savefig(path.join(PROJECT_ROOT, "data", "batch_time_vs_nodes_results.pdf"))
    plt.show()
    plt.close(fig)

    fig, axs = plt.subplots(
        len(models), 1, figsize=(12, 24)
    )  # Adjusted to vertical layout
    for i, model_name in enumerate(models):
        plot_batch_time_vs_nodes(
            results1,
            axs[i],
            model_name,
            dataset_colors,
            title=f"{model_name}",
            show_legend=(i == len(models) - 1),
            show_ylabel=False,  # Remove Y-axis label for 'results1'
            show_xlabel=(i == len(models) - 1),  # X-axis label only on the last plot
        )
    plt.tight_layout()
    plt.savefig(path.join(PROJECT_ROOT, "data", "batch_time_vs_nodes_results1.pdf"))
    plt.show()
    plt.close(fig)


# Example usage:
fontsize_title = 50
fontsize_labels = 45
fontsize_legend = 35
fontsize_ticks = 35

save_batch_time_vs_nodes_plot(
    fontsize_title=fontsize_title,
    fontsize_labels=fontsize_labels,
    fontsize_legend=fontsize_legend,
    fontsize_ticks=fontsize_ticks,
)
