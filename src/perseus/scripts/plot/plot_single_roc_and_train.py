import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.metrics import auc
from os import path
import pickle

from perseus.settings import PROJECT_ROOT

datasets = ["DDINA", "DDM"]
dataset_colors = {"DDINA": "blue", "DDM": "red"}
methods = ["GAT", "GraphSAGE"]


def load_results():
    with open(path.join(PROJECT_ROOT, "data", "saved", "results_l.pkl"), "rb") as file:
        return pickle.load(file)


results_t = load_results()


# Function to get the global minimum and maximum training times
def get_global_min_max(results, methods, datasets):
    all_times = []
    for method in methods:
        for dataset in datasets:
            all_times.extend(results[dataset][method]["train_times"])
    global_min = min(all_times)
    global_max = max(all_times)
    return global_min, global_max


global_min, global_max = get_global_min_max(results_t, methods, datasets)


def plot_combined_cdf(data, ax, colors, global_min, global_max, fontsize, ticksize):
    label_map = {
        "DDINA": "Directed",
        "COSS": "Cosine Similarity",
        "DDM": "Weighted",
    }
    # line_styles = {"DDINA": "dashed", "DDM": "solid", "COSS":"dotted"}  # Dataset line styles
    # method_colors = {"GAT": "blue", "GraphSAGE": "red" , "GCN":"green"}  # Method colors
    line_styles = {"DDINA": "dashed", "DDM": "solid"}  # Dataset line styles
    method_colors = {"GAT": "blue", "GraphSAGE": "red"}  # Method colors

    dataset_handles = {}
    method_handles = {}

    # Loop over methods
    for method in data:
        for dataset in data[method]:
            sorted_times = np.sort(data[method][dataset])
            cdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)

            # Plot CDF
            sns.lineplot(
                x=sorted_times,
                y=cdf,
                ax=ax,
                color=method_colors[method],  # Color by method
                drawstyle="steps-post",
                linestyle=line_styles.get(dataset, "solid"),  # Line style by dataset
            )

            # Store handles only once for dataset and method
            if dataset not in dataset_handles:
                dataset_handles[dataset] = plt.Line2D(
                    [0],
                    [0],
                    color="black",
                    linestyle=line_styles.get(dataset, "solid"),
                    label=label_map.get(dataset, dataset),
                )
            if method not in method_handles:
                method_handles[method] = plt.Line2D(
                    [0],
                    [0],
                    color=method_colors[method],
                    label=method,
                )

    # Set log scale for x-axis
    ax.set_xscale("log")
    ax.set_xlim(global_min, global_max)

    # Set major ticks explicitly with corresponding grid lines for x-axis
    major_ticks = np.logspace(np.log10(global_min), np.log10(global_max), num=5)
    ax.xaxis.set_major_locator(ticker.FixedLocator(major_ticks))

    # Set formatter to ensure clean labeling for log axis
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{x:.2f}" if x < 1 else f"{int(x)}")
    )

    # Disable minor ticks and minor grid lines on x-axis
    ax.minorticks_off()

    # Set the tick parameters for x-axis
    ax.tick_params(axis="x", which="major", labelsize=ticksize)

    # Major ticks for the y-axis with equal intervals
    y_ticks = np.linspace(0, 1, 6)
    ax.set_yticks(y_ticks)

    # Set labels for the y-axis ticks
    ax.set_yticklabels([f"{y:.1f}" for y in y_ticks])

    # Set the tick parameters for y-axis and disable minor ticks
    ax.tick_params(axis="y", which="major", labelsize=ticksize)
    ax.minorticks_off()  # Disable minor ticks for y-axis as well

    # Set limits for y-axis
    ax.set_ylim(0, 1)

    # Set axis labels and title
    ax.set_xlabel("Time per Epoch (seconds)", fontsize=fontsize)
    ax.set_ylabel("CDF", fontsize=fontsize)

    # Add legends once
    dataset_legend = ax.legend(
        handles=list(dataset_handles.values()),
        title="Graphs",
        fontsize=fontsize,
        title_fontsize=fontsize,
        loc="lower left",
    )
    method_legend = ax.legend(
        handles=list(method_handles.values()),
        title="Architectures",
        fontsize=fontsize,
        title_fontsize=fontsize,
        loc="lower right",
    )

    # Add both legends to the plot
    ax.add_artist(dataset_legend)

    # Set grid for major ticks only, reduce the density of grid lines
    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.5)

    # Set major ticks and grid lines to correspond on x-axis
    ax.xaxis.set_major_locator(ticker.FixedLocator(major_ticks))


def plot_combined_roc(results, ax, colors, fontsize, ticksize):
    label_map = {
        "DDINA": "Directed",
        "DDM": "Weighted",
    }
    line_styles = {"DDINA": "dashed", "DDM": "solid"}  # Dataset line styles
    method_colors = {"GAT": "blue", "GraphSAGE": "red"}  # Method colors

    # line_styles = {"DDINA": "dashed", "DDM": "solid", "COSS":"dotted"}  # Dataset line styles
    # method_colors = {"GAT": "blue", "GraphSAGE": "red" , "GCN":"green"}  # Method colors

    combined_handles = {}  # Store combined handles for legend

    for method in results:
        for dataset in datasets:
            fpr = results[method][dataset]["fpr"][0]
            tpr = results[method][dataset]["tpr"][0]
            auc_value = auc(fpr, tpr)

            # Plot ROC without confidence interval
            sns.lineplot(
                x=fpr,
                y=tpr,
                ax=ax,
                color=method_colors[method],  # Color by method
                linestyle=line_styles.get(dataset, "solid"),  # Line style by dataset
                errorbar=None,
            )

            # Combine method and dataset in one label with AUC
            combined_label = (
                f"{label_map.get(dataset, dataset)} {method} - AUC: {auc_value:.2f}"
            )

            # Store handles for combined method-dataset-auc
            combined_handles[combined_label] = plt.Line2D(
                [0],
                [0],
                color=method_colors[method],
                linestyle=line_styles.get(dataset, "solid"),
                label=combined_label,
            )

    # Set y-axis limits
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    # Set axis labels and title
    ax.set_xlabel("False Positive Rate", fontsize=fontsize)
    ax.set_ylabel("True Positive Rate", fontsize=fontsize)

    # Add combined legend with dataset, method, and AUC
    combined_legend = ax.legend(
        handles=list(combined_handles.values()),
        # title="Models and Datasets (AUC)",
        fontsize=fontsize,
        title_fontsize=fontsize,
        loc="lower right",
    )

    # Set tick parameters
    ax.tick_params(axis="both", which="major", labelsize=ticksize)

    # Add grid for major ticks
    ax.grid(True)


def save_combined_plots(plot_function, results, plot_type, settings):
    fig, ax = plt.subplots(figsize=settings["figsize"])
    global_min, global_max = (
        get_global_min_max(results, methods, datasets)
        if plot_type == "cdf"
        else (None, None)
    )
    data = {
        method: {
            dataset: (
                results[dataset][method]["train_times"]
                if plot_type == "cdf"
                else results[dataset][method]
            )
            for dataset in datasets
        }
        for method in methods
    }

    if plot_type == "cdf":
        plot_function(
            data,
            ax,
            dataset_colors,
            global_min,
            global_max,
            settings["fontsize"],
            settings["ticksize"],
        )
    else:
        plot_function(
            data,
            ax,
            dataset_colors,
            settings["fontsize"],
            settings["ticksize"],
        )

    plt.tight_layout()
    plt.savefig(path.join(PROJECT_ROOT, "data", f"oct_combined_{plot_type}_plot.pdf"))
    plt.show()
    plt.close(fig)


# Save combined CDF and ROC plots

plot_settings = {"fontsize": 22, "ticksize": 22, "figsize": (8, 8)}

save_combined_plots(plot_combined_cdf, results_t, "cdf", plot_settings)
save_combined_plots(plot_combined_roc, results_t, "roc", plot_settings)
