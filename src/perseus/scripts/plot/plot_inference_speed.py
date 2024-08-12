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


with open(path.join(PROJECT_ROOT, "data", "results.pkl"), "rb") as file:
    results = pickle.load(file)


with open(path.join(PROJECT_ROOT, "data", "results1.pkl"), "rb") as file:
    results1 = pickle.load(file)


def plot_batch_time_vs_nodes(
    results,
    ax,
    model_name,
    dataset_colors,
    title,
    show_legend=True,
    show_title=True,
    show_ylabel=True,
):
    """
    Plot batch time versus number of nodes for each dataset using line plots with spline interpolation,
    handling duplicates by averaging batch times for the same number of nodes.
    """
    for dataset in datasets:
        batch_times = np.array(results[dataset][model_name]["batch_times"])
        num_nodes = np.array(results[dataset][model_name]["num_nodes"])

        # Handling duplicates: Average batch times for the same number of nodes
        unique_nodes, indices = np.unique(num_nodes, return_inverse=True)
        average_batch_times = np.zeros_like(unique_nodes, dtype=float)
        for i, node in enumerate(unique_nodes):
            average_batch_times[i] = np.mean(batch_times[indices == i])

        # Spline interpolation
        if len(unique_nodes) > 3:  # Ensure enough points for cubic spline
            spline = make_interp_spline(unique_nodes, average_batch_times, k=3)
            fine_x = np.linspace(unique_nodes.min(), unique_nodes.max(), 500)
            fine_y = spline(fine_x)
            ax.plot(fine_x, fine_y, label=f"{dataset}", color=dataset_colors[dataset])
        else:
            ax.plot(
                unique_nodes,
                average_batch_times,
                "o-",
                label=f"{dataset}",
                color=dataset_colors[dataset],
            )

    ax.set_title(title if show_title else "", fontsize=fontsize_title)
    ax.set_xlabel("Number of Nodes", fontsize=fontsize_labels)
    ax.set_ylabel(
        "Inference Speed (sec)" if show_ylabel else "", fontsize=fontsize_labels
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.5f}"))
    if show_legend:
        ax.legend(
            title="Dataset",
            fontsize=fontsize_legend,
            title_fontsize=fontsize_legend,
            loc="lower right",
        )
    ax.tick_params(axis="x", which="major", labelsize=fontsize_ticks)
    ax.tick_params(axis="y", which="major", labelsize=fontsize_ticks)


def save_batch_time_vs_nodes_plot(
    fontsize_title, fontsize_labels, fontsize_legend, fontsize_ticks
):
    """
    Save the batch time vs number of nodes plot for all models with conditional styling.
    """
    fig, axs = plt.subplots(1, len(models), figsize=(36, 8))
    # Plotting for results
    for i, model_name in enumerate(models):
        plot_batch_time_vs_nodes(
            results,
            axs[i],
            model_name,
            dataset_colors,
            title=f"{model_name}",
            show_legend=(i == 2),
            show_title=True,
            show_ylabel=(i == 0),
        )
        axs[i].set_xlabel("")  # Remove x-titles for results

    plt.tight_layout()
    plt.savefig(path.join(PROJECT_ROOT, "data", "batch_time_vs_nodes_results.pdf"))
    plt.show()
    plt.close(fig)

    # Plotting for results1
    fig, axs = plt.subplots(1, len(models), figsize=(36, 8))
    for i, model_name in enumerate(models):
        plot_batch_time_vs_nodes(
            results1,
            axs[i],
            model_name,
            dataset_colors,
            title="",
            show_legend=False,
            show_title=False,
            show_ylabel=(i == 0),
        )
        # x-title is kept for results1

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
