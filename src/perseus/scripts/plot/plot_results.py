"""
This function is used to get the global minimum and maximum training times
"""

from os import path
import pickle
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

from perseus.settings import PROJECT_ROOT

datasets = ["DDINA", "COSS", "DDM"]
dataset_colors = {"DDINA": "blue", "COSS": "green", "DDM": "red"}
models = ["GCN", "GAT", "GraphSAGE"]
label = 1


with open(path.join(PROJECT_ROOT, "data", "results.pkl"), "rb") as file:
    results = pickle.load(file)


with open(path.join(PROJECT_ROOT, "data", "results1.pkl"), "rb") as file:
    results1 = pickle.load(file)


def get_global_min_max(results: dict, results1: dict, method: str):
    """
    This function is used to get the global minimum and maximum training times
    """
    all_times = []
    for res in [results, results1]:
        for dataset in datasets:
            all_times.extend(res[dataset][method]["train_times"])
    return min(all_times), max(all_times)


global_min_max = {}
for method in ["GCN", "GAT", "GraphSAGE"]:
    global_min_max[method] = get_global_min_max(results, results1, method)


def plot_cdf(data: dict, ax, title, colors, global_min, global_max):
    """
    This function is used to plot the CDF of the training times for each method
    """
    label_map = {
        "DDINA": "Directed Diffusion",
        "COSS": "Cosine Similarity",
        "DDM": "Weighted Diffusion",
    }
    for dataset, times in data.items():
        sorted_times = np.sort(times)
        cdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
        dataset_label = label_map.get(dataset, dataset)

        sns.lineplot(
            x=sorted_times,
            y=cdf,
            ax=ax,
            label=f"{dataset_label}",
            color=colors[dataset],
            drawstyle="steps-post",
        )

    ax.set_xscale("log")
    ax.set_xlim(global_min, global_max)

    if global_max / global_min > 1000:
        numticks = 3
    else:
        numticks = 5

    ax.xaxis.set_major_locator(
        ticker.LogLocator(base=10.0, subs="auto", numticks=numticks)
    )
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{x:.3f}" if x < 1 else f"{int(x)}")
    )

    ax.tick_params(axis="x", which="major", labelrotation=45)

    ax.set_ylim(0, 1)  # This ensures the y-axis starts at 0 and ends at 1
    ax.set_xlabel("Time per Epoch (seconds)")
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.legend()


def plot_fpr_tpr(
    results: dict,
    model_name: str,
    ax,
    datasets: str,
    label: int,
    dataset_colors: str,
    title,
    xlabel,
    ylabel,
):
    """
    This function is used to plot the FPR vs TPR for each model
    """
    label_map = {
        "DDINA": "Directed Diffusion",
        "COSS": "Cosine Similarity",
        "DDM": "Weighted Diffusion",
    }

    for dataset in datasets:
        fpr = results[dataset][model_name]["fpr"][label]
        tpr = results[dataset][model_name]["tpr"][label]

        dataset_label = label_map.get(dataset, dataset)

        sns.lineplot(
            x=fpr, y=tpr, ax=ax, label=f"{dataset_label}", color=dataset_colors[dataset]
        )

    ax.set_ylim(0, 1)  # This ensures the y-axis starts at 0 and ends at 1
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()


def save_individual_subfigures(
    fontsize_title, fontsize_labels, fontsize_legend, fontsize_ticks
):
    """
    This function is used to save the individual subfigures for the cdf and roc plots
    """

    def set_style(ax, show_title=True, show_xlabel=True, show_ylabel=True):
        if show_title:
            ax.set_title(ax.get_title(), fontsize=fontsize_title)
        else:
            ax.set_title("")
        if show_xlabel:
            ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize_labels)
        else:
            ax.set_xlabel("")
        if show_ylabel:
            ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize_labels)
        else:
            ax.set_ylabel("")
        ax.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
        if ax.get_legend():
            ax.legend(
                title=ax.get_legend().get_title().get_text(),
                fontsize=fontsize_legend,
                title_fontsize=fontsize_legend,
                loc="lower right",
            )

    # Adjusting to a vertical layout (1 column, 3 rows)
    fig_a, axs_a = plt.subplots(3, 1, figsize=(12, 24))  # Adjusted for vertical layout
    for j, method in enumerate(["GCN", "GAT", "GraphSAGE"]):
        method_train_times = {
            dataset: results[dataset][method]["train_times"] for dataset in datasets
        }
        plot_cdf(
            method_train_times,
            axs_a[j],
            method,
            dataset_colors,
            *global_min_max[method],
        )
        axs_a[j].get_legend().remove()

    for j, ax in enumerate(axs_a):
        set_style(
            ax, show_title=True, show_xlabel=(j == 2), show_ylabel=True
        )  # x-titles removed for first two plots
    plt.tight_layout()
    plt.savefig(path.join(PROJECT_ROOT, "data", "subfigure_a.pdf"))
    plt.show()
    plt.close(fig_a)

    fig_b, axs_b = plt.subplots(3, 1, figsize=(12, 24))  # Adjusted for vertical layout
    for j, method in enumerate(["GCN", "GAT", "GraphSAGE"]):
        method_train_times = {
            dataset: results1[dataset][method]["train_times"] for dataset in datasets
        }
        plot_cdf(
            method_train_times,
            axs_b[j],
            method,
            dataset_colors,
            *global_min_max[method],
        )
        if j < 2:
            axs_b[j].get_legend().remove()

    for j, ax in enumerate(axs_b):
        set_style(
            ax, show_title=True, show_xlabel=(j == 2), show_ylabel=False
        )  # x-titles removed for first two plots
    plt.tight_layout()
    plt.savefig(path.join(PROJECT_ROOT, "data", "subfigure_b.pdf"))
    plt.show()
    plt.close(fig_b)

    fig_c, axs_c = plt.subplots(
        len(models), 1, figsize=(12, 24)
    )  # Adjusted for vertical layout
    for i, model_name in enumerate(models):
        plot_fpr_tpr(
            results,
            model_name,
            axs_c[i],
            datasets,
            label,
            dataset_colors,
            title=f"{model_name}",
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
        )
        axs_c[i].get_legend().remove()

    for i, ax in enumerate(axs_c):
        set_style(
            ax, show_title=True, show_xlabel=(i == len(models) - 1), show_ylabel=True
        )  # x-titles removed for first two plots
    plt.tight_layout()
    plt.savefig(path.join(PROJECT_ROOT, "data", "subfigure_c.pdf"))
    plt.show()
    plt.close(fig_c)

    fig_d, axs_d = plt.subplots(
        len(models), 1, figsize=(12, 24)
    )  # Adjusted for vertical layout
    for i, model_name in enumerate(models):
        plot_fpr_tpr(
            results1,
            model_name,
            axs_d[i],
            datasets,
            label,
            dataset_colors,
            title=f"{model_name}",
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
        )
        if i < 2:
            axs_d[i].get_legend().remove()

    for i, ax in enumerate(axs_d):
        set_style(
            ax, show_title=True, show_xlabel=(i == len(models) - 1), show_ylabel=False
        )  # x-titles removed for first two plots
    plt.tight_layout()
    plt.savefig(path.join(PROJECT_ROOT, "data", "subfigure_d.pdf"))
    plt.show()
    plt.close(fig_d)


save_individual_subfigures(
    fontsize_title=50, fontsize_labels=45, fontsize_legend=35, fontsize_ticks=35
)
