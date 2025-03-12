from os import path
import pickle
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from perseus.settings import PROJECT_ROOT


with open(path.join(PROJECT_ROOT, "data", "results_wn.pkl"), "rb") as file:
    results_m = pickle.load(file)


datasets = ["DDINA", "DDM"]
models = ["GAT", "GraphSAGE"]
label_map = {"DDINA": "Directed", "DDM": "Weighted"}
line_styles = {"Directed": "dashed", "Weighted": "solid"}
model_colors = {"GAT": "#1f77b4", "GraphSAGE": "#ff7f0e"}


def plot_combined_batch_time_vs_nodes(
    results,
    models,
    label_map,
    fontsize=12,
    model_colors=model_colors,
    line_styles=line_styles,
):

    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    combined_handles = {}  # To store combined handles for legend

    # Plot each dataset and model combination
    for model_name in models:
        for dataset_key, dataset in label_map.items():
            if model_name in results[dataset_key]:
                batch_times = np.array(results[dataset_key][model_name]["batch_times"])
                num_nodes = np.array(results[dataset_key][model_name]["num_nodes"])
                unique_nodes, indices = np.unique(num_nodes, return_inverse=True)
                average_batch_times = np.zeros_like(unique_nodes, dtype=float)
                for i in range(len(unique_nodes)):
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
                combined_label = f"{dataset} {model_name}"

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
    ax.set_xlabel("Number of Nodes", fontsize=fontsize)
    ax.set_ylabel("Inference Speed (sec)", fontsize=fontsize)

    # Set y-axis to scientific notation
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(fontsize)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # Add combined legend if requested
    ax.legend(
        handles=list(combined_handles.values()),
        fontsize=fontsize,
        title_fontsize=fontsize,
        loc="center left",
    )

    # Set tick parameters
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(path.join(PROJECT_ROOT, "data", "feb_inference_plot.pdf"))
    plt.show()
    plt.close()


# Plot the combined batch time vs nodes on the current axes
plot_combined_batch_time_vs_nodes(
    results_m,  # Assuming results_m is the relevant dataset
    models,
    label_map,
    fontsize=26,  # Set the fontsize as desired
    model_colors=model_colors,
    line_styles=line_styles,
)
