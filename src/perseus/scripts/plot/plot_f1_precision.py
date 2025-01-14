from os import path
import pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import precision_score, f1_score
from perseus.settings import PROJECT_ROOT

datasets = ["DDINA", "DDM"]
dataset_colors = {"DDINA": "blue", "DDM": "red"}
models = ["GAT", "GraphSAGE"]


with open(path.join(PROJECT_ROOT, "data", "saved", "results_l.pkl"), "rb") as file:
    results_m = pickle.load(file)

from matplotlib.lines import Line2D


def plot_precision_f1_separately(
    results_t,
    models,
    datasets,
    thresholds=np.linspace(0, 1, 100),
    fontsize=10,
    ticksize=10,
    save_plots=True,
):
    # Color and linestyle for clarity
    colors = ["b", "r"]
    linestyles = ["dashed", "-"]

    # Label map for datasets
    label_map = {
        "DDINA": "Directed",
        "DDM": "Weighted",
    }

    # Combined handles for legend
    combined_handles = {}

    # Function to plot a specific metric
    def plot_metric(metric_name, ylabel, file_suffix):
        plt.figure(figsize=(8, 8))

        # Plotting precision or F1 score for each model and dataset
        for i, model in enumerate(models):
            for j, dataset in enumerate(datasets):
                labels = results_m[dataset][model]["metrics"]["labels"]
                probs = results_m[dataset][model]["metrics"]["probs"]

                # Calculate scores based on the metric type
                if metric_name == "Precision":
                    scores = [
                        precision_score(labels, probs > threshold, zero_division=0)
                        for threshold in thresholds
                    ]
                elif metric_name == "F1 Score":
                    scores = [
                        f1_score(labels, probs > threshold, zero_division=0)
                        for threshold in thresholds
                    ]

                # Plot scores against thresholds
                plt.plot(thresholds, scores, color=colors[i], linestyle=linestyles[j])

                # Create combined label for each dataset-model pair
                combined_label = f"{label_map[dataset]} {model}"

                # Create and store unique combined handles for legend
                if combined_label not in combined_handles:
                    combined_handles[combined_label] = Line2D(
                        [0],
                        [0],
                        color=colors[i],
                        linestyle=linestyles[j],
                        label=combined_label,
                    )

        # Configure axis labels and title
        plt.xlabel("Threshold", fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.grid(True)
        plt.tick_params(axis="both", which="major", labelsize=ticksize)

        # Add combined legend with dataset-model pairs
        plt.legend(
            handles=list(combined_handles.values()),
            fontsize=fontsize - 2,
            # title="Models and Graphs",
            title_fontsize=fontsize,
            loc="lower left",
        )

        # Save the plot if requested
        if save_plots:
            plt.savefig(
                path.join(PROJECT_ROOT, "data", f"nov_{file_suffix}_plot.pdf"),
                bbox_inches="tight",
                format="pdf",
            )

        plt.show()

    # Plot Precision
    plot_metric("Precision", "Precision", "precision")

    # Plot F1 Score
    plot_metric("F1 Score", "F1 Score", "f1")


# Call the function with desired parameters
plot_precision_f1_separately(results_m, models, datasets, fontsize=22, ticksize=22)
