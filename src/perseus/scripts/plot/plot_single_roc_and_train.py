import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.metrics import auc
from os import path
import pickle
import numpy as np
from perseus.settings import PROJECT_ROOT

# Load the results
with open(path.join(PROJECT_ROOT, "data", "results_wn.pkl"), "rb") as file:
    results_t = pickle.load(file)

# Use the same label_map and line_styles as in the first script
label_map = {"DDINA": "Directed", "DDM": "Weighted"}
line_styles = {"Directed": "dashed", "Weighted": "solid"}
model_colors = {"GAT": "#1f77b4", "GraphSAGE": "#ff7f0e"}


def plot_combined_cdf(
    results_t, fontsize, ticksize, label_map, line_styles, model_colors
):
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    data = {
        method: {
            dataset: results_t[dataset][method]["train_times"]
            for dataset in label_map.keys()
        }
        for method in model_colors.keys()
    }

    all_times = []
    for method in model_colors.keys():
        for dataset in label_map.keys():
            all_times.extend(results_t[dataset][method]["train_times"])
    global_min = min(all_times)
    global_max = max(all_times)
    combined_handles = {}  # Store combined handles for legend

    # Loop over models and datasets to plot CDF and create handles
    for method in data:
        for dataset in data[method]:
            sorted_times = np.sort(data[method][dataset])
            cdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)

            # Look up the mapped label to get the proper line style
            style_key = label_map.get(dataset, dataset)
            current_linestyle = line_styles.get(style_key, "solid")

            # Plot CDF
            sns.lineplot(
                x=sorted_times,
                y=cdf,
                ax=ax,
                color=model_colors[method],  # Color by method
                drawstyle="steps-post",
                linestyle=current_linestyle,  # Line style based on mapped label
            )

            # Combine method and dataset in one label
            combined_label = f"{style_key} {method}"

            # Store unique combined handles for legend
            if combined_label not in combined_handles:
                combined_handles[combined_label] = plt.Line2D(
                    [0],
                    [0],
                    color=model_colors[method],
                    linestyle=current_linestyle,
                    label=combined_label,
                )

    # Set log scale for x-axis
    ax.set_xscale("log")
    ax.set_xlim(global_min, global_max)

    # Set major ticks explicitly with corresponding grid lines for x-axis
    major_ticks = np.logspace(np.log10(global_min), np.log10(global_max), num=5)
    ax.xaxis.set_major_locator(ticker.FixedLocator(major_ticks))

    # Set formatter for log axis labels
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{x:.2f}" if x < 1 else f"{int(x)}")
    )

    # Disable minor ticks and minor grid lines on x-axis
    ax.minorticks_off()
    ax.tick_params(axis="x", which="major", labelsize=ticksize)

    # Major ticks for the y-axis with equal intervals
    y_ticks = np.linspace(0, 1, 6)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y:.1f}" for y in y_ticks])
    ax.tick_params(axis="y", which="major", labelsize=ticksize)
    ax.minorticks_off()

    # Set limits and labels for the axes
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time per Epoch (seconds)", fontsize=fontsize)
    ax.set_ylabel("CDF", fontsize=fontsize)

    # Add combined legend with method and dataset
    ax.legend(
        handles=list(combined_handles.values()),
        fontsize=fontsize,
        title_fontsize=fontsize,
        loc="lower right",
    )

    # Set grid for major ticks only
    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(path.join(PROJECT_ROOT, "data", "feb_combined_cdf_plot.pdf"))
    plt.show()
    plt.close()


def plot_combined_roc(
    results, fontsize, ticksize, label_map, line_styles, model_colors
):
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    combined_handles = {}  # Store combined handles for legend

    for dataset in label_map.keys():
        for method in model_colors.keys():
            fpr = results[dataset][method]["fpr"][0]
            tpr = results[dataset][method]["tpr"][0]
            auc_value = auc(fpr, tpr)

            # Look up the mapped label to get the proper line style
            style_key = label_map.get(dataset, dataset)
            current_linestyle = line_styles.get(style_key, "solid")

            # Plot ROC without confidence interval
            sns.lineplot(
                x=fpr,
                y=tpr,
                ax=ax,
                color=model_colors[method],  # Color by method
                linestyle=current_linestyle,  # Line style based on mapped label
                errorbar=None,
            )

            # Combine dataset and method in one label with AUC
            combined_label = f"{style_key} {method} AUC: {auc_value:.2f}"

            # Store handle for the combined label
            combined_handles[combined_label] = plt.Line2D(
                [0],
                [0],
                color=model_colors[method],
                linestyle=current_linestyle,
                label=combined_label,
            )

    # Set axis limits and labels
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("False Positive Rate", fontsize=fontsize)
    ax.set_ylabel("True Positive Rate", fontsize=fontsize)

    # Add combined legend
    ax.legend(
        handles=list(combined_handles.values()),
        fontsize=fontsize,
        title_fontsize=fontsize,
        loc="lower right",
    )

    ax.tick_params(axis="both", which="major", labelsize=ticksize)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(path.join(PROJECT_ROOT, "data", "feb_combined_roc_plot.pdf"))
    plt.show()
    plt.close()


# Plot the combined CDF and ROC using the updated definitions
plot_combined_cdf(
    results_t,
    fontsize=26,
    ticksize=26,
    label_map=label_map,
    line_styles=line_styles,
    model_colors=model_colors,
)

plot_combined_roc(
    results_t,
    fontsize=26,
    ticksize=26,
    label_map=label_map,
    line_styles=line_styles,
    model_colors=model_colors,
)
