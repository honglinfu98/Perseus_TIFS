from os import path
import pickle
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from perseus.settings import PROJECT_ROOT

datasets = ["DDINA", "DDM"]
dataset_colors = {"DDINA": "blue", "DDM": "red"}
models = ["GAT", "GraphSAGE"]
label = 1
dataset_labels = {
    "DDINA": "Directed Diffusion",
    "COSS": "Cosine Similarity",
    "DDM": "Weighted Diffusion",
}


with open(path.join(PROJECT_ROOT, "data", "saved", "results_l.pkl"), "rb") as file:
    results_m = pickle.load(file)

from matplotlib.ticker import MaxNLocator, ScalarFormatter


def plot_combined_batch_time_vs_nodes(
    results,
    ax,
    models,
    title,
    show_legend=True,
    show_ylabel=True,
    show_xlabel=True,
    fontsize=12,
    ticksize=10,
):
    line_styles = {"Directed": "dashed", "Weighted": "solid"}
    model_colors = {"GAT": "blue", "GraphSAGE": "red"}
    dataset_name_mapping = {"DDINA": "Directed", "DDM": "Weighted"}

    combined_handles = {}  # To store combined handles for legend

    # Plot each dataset and model combination
    for model_name in models:
        for dataset_key in dataset_name_mapping:
            dataset = dataset_name_mapping[dataset_key]
            if model_name in results[dataset_key]:
                batch_times = np.array(results[dataset_key][model_name]["batch_times"])
                num_nodes = np.array(results[dataset_key][model_name]["num_nodes"])
                unique_nodes, indices = np.unique(num_nodes, return_inverse=True)
                average_batch_times = np.zeros_like(unique_nodes, dtype=float)
                for i, node in enumerate(unique_nodes):
                    average_batch_times[i] = np.mean(batch_times[indices == i])

                # Check if enough points for spline interpolation
                if len(unique_nodes) > 3:
                    spline = make_interp_spline(unique_nodes, average_batch_times, k=3)
                    fine_x = np.linspace(unique_nodes.min(), unique_nodes.max(), 500)
                    fine_y = spline(fine_x)
                    ax.plot(
                        fine_x,
                        fine_y,
                        color=model_colors[model_name],
                        linestyle=line_styles[dataset],
                    )
                else:
                    ax.plot(
                        unique_nodes,
                        average_batch_times,
                        "o-",
                        color=model_colors[model_name],
                        linestyle=line_styles[dataset],
                    )

                # Combine dataset and model in one label
                combined_label = f"{dataset} ({model_name})"

                # Create and store unique combined handles for legend
                if combined_label not in combined_handles:
                    combined_handles[combined_label] = plt.Line2D(
                        [0],
                        [0],
                        color=model_colors[model_name],
                        linestyle=line_styles[dataset],
                        label=combined_label,
                    )

    # Set labels with fontsize
    if show_xlabel:
        ax.set_xlabel("Number of Nodes", fontsize=fontsize)
    if show_ylabel:
        ax.set_ylabel("Inference Speed (sec)", fontsize=fontsize)

    # Set y-axis to scientific notation
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(ticksize)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # Add combined legend if requested
    if show_legend:
        ax.legend(
            handles=list(combined_handles.values()),
            fontsize=fontsize,
            title_fontsize=fontsize,
            loc="center left",
        )

    # Set tick parameters
    ax.tick_params(axis="both", which="major", labelsize=ticksize)
    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.5)


# Example usage for the combined plot:
fig, ax = plt.subplots(figsize=(8, 8))  # Single plot for all results
plot_combined_batch_time_vs_nodes(
    results_m,  # Assuming results_m is the relevant dataset
    ax,
    models,
    title="Batch Time vs Nodes for Directed Diffusion and Weighted Diffusion (GAT vs GraphSAGE)",
    show_legend=True,
    show_ylabel=True,
    show_xlabel=True,
    fontsize=22,  # Set the fontsize as desired
    ticksize=22,  # Set the ticksize as desired
)
plt.tight_layout()
plt.savefig(
    path.join(PROJECT_ROOT, "data", "oct_combined_batch_time_vs_nodes_results.pdf")
)
plt.show()
plt.close(fig)
